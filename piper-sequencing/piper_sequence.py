import math
import time

from piper_sdk import C_PiperInterface_V2

RAD_TO_MDEG = 1000 * 180 / math.pi

# Bench-safe speeds: 10% rate + 0.3 rad/s hard per-joint cap.
# Flip to False once you're confident in the trajectory.
SAFE_MODE = True
SAFE_SPEED_PCT = 10        # MotionCtrl_2 move_spd_rate_ctrl (0-100)
SAFE_MAX_JOINT_SPD = 300   # MotorMaxSpdSet, 0.001 rad/s (300 = 0.3 rad/s)
NORMAL_SPEED_PCT = 30
NORMAL_MAX_JOINT_SPD = 3000  # 3.0 rad/s, the firmware ceiling

# Safety: abort the move if any joint's torque (effort) exceeds its threshold
# during the middle/end steps (i.e. when the patient is engaged). Tune
# FORCE_THRESHOLDS_NM empirically — start high and lower until normal motion
# doesn't false-trip. Set FORCE_MONITORING = False to disable entirely.
FORCE_MONITORING = False
FORCE_THRESHOLDS_NM = [15.0, 24.0, 17.0, 3.0, 6.8, 2.5]  # J1..J6, in N·m
FORCE_POLL_HZ = 50
# Firmware ≤ S-V1.8-2 reports J1-J3 effort scaled 4× too low; flip to False
# on newer firmware. Check with piper.GetPiperFirmwareVersion().
LEGACY_J123_TORQUE_FIX = True
# Max time to wait for each move to complete (or trip a safety abort).
MOVE_DURATION_S = 3.0

# A "capture" is a named series of joint poses (radians, j1..j6) to play back
# in order. Record new ones with test_scripts/read_joint.py and paste them in.
#
# Each step is (name, joints_rad). A trailing element (e.g. skip flag from an
# earlier design) is ignored — the runtime flow has three fixed auth points:
#   1) start session: Enter to move from home to the first step
#   2) begin workout: Enter at the first step to auto-execute the rest
#   3) end session:   Enter at the last step to return to `home`
CAPTURES: dict[str, dict] = {
    "right_arm": {
        "name": "Right Arm",
        "description": "Sweep across the right side of the workspace.",
        "sequence": [
                ("start", [-0.4227, -0.0644, -3.0429, -0.1525, -0.4191, -2.9603]),
                ("middle", [-0.4826, -0.0632, -2.0577, 0.0176, -1.3199, -2.9609], True),
                ("end", [-2.6961, 0.2407, -1.3574, 1.7632, -0.3218, -2.4452]),
        ],
        "home": [0.0465, -0.2051, -3.0447, 0.0326, -1.3196, -2.9607],
    },
    "left_arm": {
        "name": "Left Arm",
        "description": "Sweep across the left side of the workspace.",
        "sequence": [
            ("start",  [0.6595, -0.1974, -3.0425, -0.1212, -1.1313, -2.9385]),
            ("middle", [0.4823, -0.1974, -3.0404, -0.4585, -0.2121, -2.595], True),
            ("end",    [0.419,  -0.1935, -2.5504, -0.6327, -0.2745, -2.5503], True),
            ("step_3", [1.5691,  1.0871, -3.0346, -1.2592, -0.2813, -2.6376]),
        ],
        "home": [0.0465, -0.2051, -3.0447, 0.0326, -1.3196, -2.9607],
    },
}


def choose_capture() -> tuple[str, dict]:
    keys = list(CAPTURES.keys())
    print("Available captures:")
    for i, key in enumerate(keys, 1):
        cap = CAPTURES[key]
        n_steps = len(cap["sequence"])
        print(f"  {i}) {cap['name']} ({n_steps} step{'s' if n_steps != 1 else ''})")
        print(f"       {cap['description']}")
    while True:
        choice = input("Select capture (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            key = keys[int(choice) - 1]
            return key, CAPTURES[key]
        print("Invalid choice — enter a number from the list.")


def _current_joints_deg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    return [r / 1000.0 for r in raw]  # 0.001° → °


def move_to(piper: C_PiperInterface_V2, joints_rad: list[float]) -> None:
    speed_pct = SAFE_SPEED_PCT if SAFE_MODE else NORMAL_SPEED_PCT
    target_deg = [math.degrees(a) for a in joints_rad]
    current_deg = _current_joints_deg(piper)
    delta_deg = [t - c for t, c in zip(target_deg, current_deg)]
    print(f"  current: {[f'{d:+7.2f}' for d in current_deg]} deg")
    print(f"  target:  {[f'{d:+7.2f}' for d in target_deg]} deg")
    print(f"  delta:   {[f'{d:+7.2f}' for d in delta_deg]} deg")
    piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)  # CAN ctrl, MOVE J, pos-vel
    piper.JointCtrl(*(int(round(a * RAD_TO_MDEG)) for a in joints_rad))


def read_joint_efforts_nm(piper: C_PiperInterface_V2) -> list[float]:
    """Per-joint |torque| in N·m. Effort is reported in 0.001 N·m."""
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
    """Wait up to duration_s; return (ok, tripped_joint_index, tripped_value).

    On over-threshold reading, send track-terminate (0x150 track_ctrl=0x06)
    to halt motion with motors still powered, and return False.
    """
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


def main() -> None:
    _, cap = choose_capture()
    sequence = cap["sequence"]
    if not sequence:
        print(f"\n'{cap['name']}' has no poses recorded yet. "
              f"Capture some with test_scripts/read_joint.py first.")
        return
    print(f"\nSelected: {cap['name']} — {len(sequence)} step(s).")

    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    print("Enabling arm...")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Arm enabled.")

    # Clear any residual trajectory state from a prior force-abort
    # (track_ctrl=0x04 = "clear all trajectories"). No effect on motors,
    # so the arm stays held in place — unlike ResetPiper which cuts power.
    piper.MotionCtrl_1(0x00, 0x04, 0x00)
    time.sleep(0.05)

    status = piper.GetArmStatus().arm_status
    print(f"ctrl_mode={status.ctrl_mode}, arm_status={status.arm_status}, "
      f"motion_status={status.motion_status}")


    max_spd = SAFE_MAX_JOINT_SPD if SAFE_MODE else NORMAL_MAX_JOINT_SPD
    print(f"Setting per-joint max speed to {max_spd / 1000:.2f} rad/s "
          f"(SAFE_MODE={SAFE_MODE}).")
    for motor_num in range(1, 7):  # MotorMaxSpdSet rejects 7=all; loop 1..6
        piper.MotorMaxSpdSet(motor_num, max_spd)

    # Prime CAN-control + MOVE J mode so the first JointCtrl in the loop
    # isn't dropped while the firmware is still in STANDBY.
    speed_pct = SAFE_SPEED_PCT if SAFE_MODE else NORMAL_SPEED_PCT
    piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)
    time.sleep(0.1)

    if FORCE_MONITORING:
        print(f"Force monitoring ENABLED — per-joint thresholds (N·m): "
              f"{FORCE_THRESHOLDS_NM}")
    else:
        print("Force monitoring DISABLED.")

    def run_move(step_name: str, pose: list[float], monitor: bool) -> bool:
        """Execute a move + wait/monitor. Returns False on force-abort."""
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

    # Three authorization points: start session, begin workout, end session.
    # 1. Auth → arm moves from its current (home) position to the first step.
    first_name, first_pose = sequence[0][0], sequence[0][1]
    input(f"Press Enter to start session (arm will move to "
          f"'{first_name}' position)...")
    if not run_move(first_name, first_pose, monitor=False):
        return

    # 2. Auth at the first step → auto-execute all remaining steps in order.
    if len(sequence) > 1:
        input(f"At '{first_name}' position. "
              f"Press Enter to begin the workout sequence...")
        for step in sequence[1:]:
            step_name, pose = step[0], step[1]
            if not run_move(step_name, pose, FORCE_MONITORING):
                return

    # 3. Auth at the last step → arm returns to home, ending the session.
    last_name = sequence[-1][0]
    home = cap.get("home")
    if home is not None:
        input(f"At '{last_name}' position. "
              f"Press Enter to end session and return to home...")
        if not run_move("home", home, FORCE_MONITORING):
            return
        print("Session ended. Arm at home position.")
    else:
        print("Sequence complete.")


if __name__ == "__main__":
    main()
