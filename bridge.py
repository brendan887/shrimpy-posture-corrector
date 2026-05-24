"""Robot <-> Pose UI status bridge.

Newline-delimited JSON over TCP. The robot runs a StatusServer and broadcasts
phase events; the UI runs a StatusClient that maintains a RobotState snapshot
in a background thread. Stdlib-only so neither side needs extra deps.
"""

from __future__ import annotations

import dataclasses
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# Phase strings broadcast by the robot.
PHASE_DISCONNECTED = "disconnected"
PHASE_AT_HOME = "at_home"
PHASE_MOVING_TO_START = "moving_to_start"
PHASE_AT_START = "at_start"
PHASE_EXECUTING = "executing"
PHASE_AT_END = "at_end"
PHASE_RETURNING_HOME = "returning_home"
PHASE_ABORTED = "aborted"


@dataclass
class RobotState:
    connected: bool = False
    phase: str = PHASE_DISCONNECTED
    capture_key: Optional[str] = None
    capture_name: Optional[str] = None
    step: Optional[str] = None
    target: Optional[str] = None
    detail: Optional[str] = None
    sequence_steps: tuple = field(default_factory=tuple)
    last_event_at: float = 0.0


class StatusServer:
    """Robot-side broadcaster. Accepts UI clients; sends events to all of them."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="status-server-accept"
        )
        self._accept_thread.start()
        print(f"[bridge] Status server listening on {self._host}:{self._port}")

    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(1.0)
            with self._lock:
                self._clients.append(client)
            print(f"[bridge] UI client connected from {addr}")

    def send(self, event: str, **data) -> None:
        payload = {"event": event, "ts": time.time(), **data}
        line = (json.dumps(payload) + "\n").encode("utf-8")
        dead: list[socket.socket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(line)
                except OSError:
                    dead.append(client)
            for d in dead:
                self._clients.remove(d)
                try:
                    d.close()
                except OSError:
                    pass

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()


class StatusClient:
    """UI-side subscriber. Background thread keeps state in sync."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self._state = RobotState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="status-client"
        )
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            try:
                sock.connect((self._host, self._port))
            except OSError:
                sock.close()
                with self._lock:
                    self._state.connected = False
                    self._state.phase = PHASE_DISCONNECTED
                time.sleep(1.0)
                continue

            sock.settimeout(None)
            with self._lock:
                self._state.connected = True

            buf = b""
            try:
                while self._running:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, _, buf = buf.partition(b"\n")
                        self._handle_line(line)
            except OSError:
                pass
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
                with self._lock:
                    self._state.connected = False
                    self._state.phase = PHASE_DISCONNECTED

    def _handle_line(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

        event = payload.get("event")
        with self._lock:
            self._state.last_event_at = float(payload.get("ts", time.time()))
            if event == "capture_selected":
                self._state.capture_key = payload.get("capture")
                self._state.capture_name = payload.get("name")
                steps = payload.get("sequence", [])
                self._state.sequence_steps = tuple(steps)
            elif event in (
                PHASE_AT_HOME,
                PHASE_MOVING_TO_START,
                PHASE_AT_START,
                PHASE_EXECUTING,
                PHASE_AT_END,
                PHASE_RETURNING_HOME,
                PHASE_ABORTED,
            ):
                self._state.phase = event
                self._state.step = payload.get("step")
                self._state.target = payload.get("target")
                self._state.detail = payload.get("detail")

    def snapshot(self) -> RobotState:
        with self._lock:
            return dataclasses.replace(self._state)

    def stop(self) -> None:
        self._running = False
