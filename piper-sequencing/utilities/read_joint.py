"""Capture Piper arm poses by hand (motors-off teach).

The arm motors are left disabled so you can move the arm freely. Press Enter
to capture the current joint angles, press 'q' + Enter when done. Output is
a Python list ready to paste into piper_sequence.py's SEQUENCES.

WARNING: with motors off, the arm has no holding torque — support it before
running this, or rest it on the bench so it can't fall.
"""
import math
import time

from piper_sdk import C_PiperInterface_V2, LogLevel

MDEG_TO_RAD = math.pi / 180_000  # joint feedback is in 0.001 degrees


def read_joints_rad(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    return [j * MDEG_TO_RAD for j in raw]


def main() -> None:
    piper = C_PiperInterface_V2(
        can_name="can0",
        dh_is_offset=1,
        start_sdk_joint_limit=False,
        start_sdk_gripper_limit=False,
        logger_level=LogLevel.WARNING,
    )
    piper.ConnectPort()
    time.sleep(0.1)

    # Make sure motors are unpowered. If a prior script left them enabled,
    # the arm will be rigid; disabling cuts power so you can move it by hand.
    input("Support the arm now — pressing Enter will cut motor power. > ")
    piper.DisablePiper()
    time.sleep(0.2)

    # Wait until joint feedback is actually streaming (first frame is all zeros).
    print("Waiting for joint feedback...")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        msg = piper.GetArmJointMsgs()
        if msg.Hz > 0:
            break
        time.sleep(0.05)
    else:
        print("WARNING: no joint feedback received — check arm power and CAN.")

    captured: list[list[float]] = []
    print("\nCommands:")
    print("  Enter      -> capture current pose")
    print("  q + Enter  -> finish and print sequence\n")

    while True:
        cmd = input(f"[{len(captured)} captured] > ").strip().lower()
        if cmd == "q":
            break
        pose = read_joints_rad(piper)
        captured.append(pose)
        pose_deg = [round(math.degrees(a), 2) for a in pose]
        print(f"  captured #{len(captured)}: {pose_deg} deg")

    if not captured:
        print("No poses captured.")
        return

    print("\n# --- paste into piper_sequence.py ---")
    print("SEQUENCES = [")
    for i, pose in enumerate(captured):
        name = ["start", "middle", "end"][i] if i < 3 else f"step_{i}"
        rounded = [round(a, 4) for a in pose]
        print(f'    ("{name}", {rounded}),')
    print("]")


if __name__ == "__main__":
    main()
