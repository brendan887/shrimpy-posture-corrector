"""Demo variant of piper_sequence.py — right arm only, driven by the UI
workflow over the status bridge instead of blocking input() prompts.

Command flow (UI -> robot):
  1. start_session  -> arm moves from home to the right-arm 'start' pose
  2. begin_workout  -> arm runs through middle -> end (the stretch)
  3. end_session    -> arm returns to home

Each command is broadcast by live_pose_full.py when its 5-step workflow
crosses the corresponding boundary. The robot publishes phase events back
(at_home / moving_to_start / at_start / executing / at_end / returning_home /
aborted) so the UI can react — most importantly, the workflow uses 'at_end'
as the cue to start the 3-second peak-hold capture.

Run after starting live_pose_full.py (or before — both sides reconnect).
"""

import math
import os
import sys
import time

# bridge.py lives in the project root, one level above piper-sequencing/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import StatusServer
from piper_sdk import C_PiperInterface_V2

RAD_TO_MDEG = 1000 * 180 / math.pi

SAFE_MODE = True
SAFE_SPEED_PCT = 10
SAFE_MAX_JOINT_SPD = 300
NORMAL_SPEED_PCT = 30
NORMAL_MAX_JOINT_SPD = 3000

FORCE_MONITORING = False
FORCE_THRESHOLDS_NM = [15.0, 24.0, 17.0, 3.0, 6.8, 2.5]
FORCE_POLL_HZ = 50
LEGACY_J123_TORQUE_FIX = True
MOVE_DURATION_S = 3.0

# Demo only ships the right arm. Joint poses copied verbatim from
# piper_sequence.CAPTURES["right_arm"].
RIGHT_ARM_CAPTURE = {
    "key": "right_arm",
    "name": "Right Arm",
    "description": "Sweep across the right side of the workspace.",
    "sequence": [
        ("start",  [-0.4227, -0.0644, -3.0429, -0.1525, -0.4191, -2.9603]),
        ("middle", [-0.4826, -0.0632, -2.0577,  0.0176, -1.3199, -2.9609], True),
        ("end",    [-2.6961,  0.2407, -1.3574,  1.7632, -0.3218, -2.4452]),
    ],
    "home": [0.0465, -0.2051, -3.0447, 0.0326, -1.3196, -2.9607],
}


def _current_joints_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    return [r / 1000.0 for r in raw]


def move_to(piper: C_PiperInterface_V2, joints_rad: list[float]) -> None:
    speed_pct = SAFE_SPEED_PCT if SAFE_MODE else NORMAL_SPEED_PCT
    target_deg = [math.degrees(a) for a in joints_rad]
    current_deg = _current_joints_deg(piper)
    delta_deg = [t - c for t, c in zip(target_deg, current_deg)]
    print(f"  current: {[f'{d:+7.2f}' for d in current_deg]} deg")
    print(f"  target:  {[f'{d:+7.2f}' for d in target_deg]} deg")
    print(f"  delta:   {[f'{d:+7.2f}' for d in delta_deg]} deg")
    piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
    piper.JointCtrl(*(int(round(a * RAD_TO_MDEG)) for a in joints_rad))


def read_joint_efforts_nm(piper: C_PiperInterface_V2) -> list[float]:
    msgs = piper.GetArmHighSpdInfoMsgs()
    motors = [msgs.motor_1, msgs.motor_2, msgs.motor_3,
              msgs.motor_4, msgs.motor_5, msgs.motor_6]
    efforts = [m.effort / 1000.0 for m in motors]
    if LEGACY_J123_TORQUE_FIX:
        efforts[0] *= 4
        efforts[1] *= 4
        efforts[2] *= 4
    return [abs(e) for e in efforts]


def wait_with_force_check(piper: C_PiperInterface_V2,
                          duration_s: float) -> tuple[bool, int, float]:
    deadline = time.time() + duration_s
    interval = 1.0 / FORCE_POLL_HZ
    while time.time() < deadline:
        efforts = read_joint_efforts_nm(piper)
        for i, (e, thresh) in enumerate(zip(efforts, FORCE_THRESHOLDS_NM)):
            if e > thresh:
                piper.MotionCtrl_1(0x00, 0x06, 0x00)
                return False, i, e
        time.sleep(interval)
    return True, -1, 0.0


def wait_for_command(bus: StatusServer, expected: str) -> None:
    """Block until the UI sends a command. Logs and discards anything else."""
    print(f"[demo] waiting for UI command '{expected}'…")
    while True:
        msg = bus.wait_for_command(timeout=1.0)
        if msg is None:
            continue
        cmd = msg.get("command")
        if cmd == expected:
            print(f"[demo] received '{expected}'")
            return
        print(f"[demo] ignoring unexpected command: {msg}")


def main() -> None:
    cap = RIGHT_ARM_CAPTURE
    sequence = cap["sequence"]
    print(f"\nDemo mode — hardcoded {cap['name']} sequence "
          f"({len(sequence)} step(s)).")

    bus = StatusServer()
    bus.start()
    bus.send(
        "capture_selected",
        capture=cap["key"],
        name=cap["name"],
        sequence=[step[0] for step in sequence],
    )

    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    print("Enabling arm…")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Arm enabled.")

    # Clear residual trajectory state (track_ctrl=0x04).
    piper.MotionCtrl_1(0x00, 0x04, 0x00)
    time.sleep(0.05)

    status = piper.GetArmStatus().arm_status
    print(f"ctrl_mode={status.ctrl_mode}, arm_status={status.arm_status}, "
          f"motion_status={status.motion_status}")

    max_spd = SAFE_MAX_JOINT_SPD if SAFE_MODE else NORMAL_MAX_JOINT_SPD
    print(f"Setting per-joint max speed to {max_spd / 1000:.2f} rad/s "
          f"(SAFE_MODE={SAFE_MODE}).")
    for motor_num in range(1, 7):
        piper.MotorMaxSpdSet(motor_num, max_spd)

    speed_pct = SAFE_SPEED_PCT if SAFE_MODE else NORMAL_SPEED_PCT
    piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
    time.sleep(0.1)

    if FORCE_MONITORING:
        print(f"Force monitoring ENABLED — per-joint thresholds (N·m): "
              f"{FORCE_THRESHOLDS_NM}")
    else:
        print("Force monitoring DISABLED.")

    bus.send("at_home", detail="Arm enabled, awaiting start_session.")

    def run_move(step_name: str, pose: list[float], monitor: bool) -> bool:
        print(f"Moving to {step_name} pose: {pose}")
        move_to(piper, pose)
        if monitor:
            ok, joint, value = wait_with_force_check(piper, MOVE_DURATION_S)
            if not ok:
                thresh = FORCE_THRESHOLDS_NM[joint]
                print("\n!!! SAFETY ABORT !!!")
                print(f"Joint J{joint + 1} torque {value:.2f} N·m exceeded "
                      f"threshold {thresh:.2f} N·m at step '{step_name}'.")
                print("Motion terminated; motors still powered, arm holding position.")
                return False
        else:
            time.sleep(MOVE_DURATION_S)
        return True

    # 1) start_session: UI->robot when workflow enters step 3 (grab handle).
    wait_for_command(bus, "start_session")
    first_name, first_pose = sequence[0][0], sequence[0][1]
    bus.send("moving_to_start", target=first_name)
    if not run_move(first_name, first_pose, monitor=False):
        bus.send("aborted", step=first_name, detail="Start move failed.")
        return
    bus.send("at_start", step=first_name)

    # 2) begin_workout: UI->robot when workflow enters step 4 (stretch).
    wait_for_command(bus, "begin_workout")
    for step in sequence[1:]:
        step_name, pose = step[0], step[1]
        bus.send("executing", step=step_name)
        if not run_move(step_name, pose, FORCE_MONITORING):
            bus.send("aborted", step=step_name, detail="Workout step failed.")
            return
    last_name = sequence[-1][0]
    bus.send("at_end", step=last_name)
    # ^ This 'at_end' is the cue the UI uses to start the 3-second peak-hold
    # capture for the assisted (post-stretch) angle.

    # 3) end_session: UI->robot when workflow enters step 5 (results).
    wait_for_command(bus, "end_session")
    home = cap.get("home")
    if home is not None:
        bus.send("returning_home")
        if not run_move("home", home, FORCE_MONITORING):
            bus.send("aborted", step="home", detail="Home return failed.")
            return
        print("Session ended. Arm at home position.")
        bus.send("at_home", detail="Session ended.")
    else:
        print("Sequence complete.")


if __name__ == "__main__":
    main()
