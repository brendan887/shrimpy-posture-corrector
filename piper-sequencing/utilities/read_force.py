"""Continuously print per-joint torques to calibrate FORCE_THRESHOLDS_NM.

The arm is enabled (so motors actively hold position). Push or pull on the
end-effector with the patient-resistance level you want to safely tolerate,
and watch each joint's torque. Set the thresholds in piper_sequence.py
slightly above the peaks you observe.

Ctrl-C to stop. The final peaks-per-joint are printed on exit.
"""
import math
import time

from piper_sdk import C_PiperInterface_V2, LogLevel

# Keep these in sync with piper_sequence.py.
LEGACY_J123_TORQUE_FIX = True
THRESHOLDS_NM = [20.0, 20.0, 20.0, 3.0, 3.0, 3.0]  # for the "!" exceedance marker

PRINT_HZ = 10


def read_joint_efforts_nm(piper: C_PiperInterface_V2) -> list[float]:
    """Per-joint torque in N·m (signed). Effort is reported in 0.001 N·m."""
    msgs = piper.GetArmHighSpdInfoMsgs()
    motors = [msgs.motor_1, msgs.motor_2, msgs.motor_3,
              msgs.motor_4, msgs.motor_5, msgs.motor_6]
    efforts = [m.effort / 1000.0 for m in motors]
    if LEGACY_J123_TORQUE_FIX:
        efforts[0] *= 4
        efforts[1] *= 4
        efforts[2] *= 4
    return efforts


def main() -> None:
    piper = C_PiperInterface_V2("can0", logger_level=LogLevel.WARNING)
    piper.ConnectPort()
    print("Enabling arm so motors hold position...")
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Arm enabled.\n")

    print(f"Thresholds (N·m):  J1={THRESHOLDS_NM[0]:.1f}  J2={THRESHOLDS_NM[1]:.1f}  "
          f"J3={THRESHOLDS_NM[2]:.1f}  J4={THRESHOLDS_NM[3]:.1f}  "
          f"J5={THRESHOLDS_NM[4]:.1f}  J6={THRESHOLDS_NM[5]:.1f}")
    print(f"Push/pull on the arm to see how torque responds. "
          f"'!' marks values above threshold. Ctrl-C to stop.\n")

    peak = [0.0] * 6
    interval = 1.0 / PRINT_HZ
    try:
        while True:
            efforts = read_joint_efforts_nm(piper)
            for i, e in enumerate(efforts):
                if abs(e) > peak[i]:
                    peak[i] = abs(e)
            row_parts = []
            for i, e in enumerate(efforts):
                marker = "!" if abs(e) > THRESHOLDS_NM[i] else " "
                row_parts.append(f"J{i+1}:{e:+6.2f}{marker}")
            peak_str = " ".join(f"{p:5.2f}" for p in peak)
            print("\r" + "  ".join(row_parts) + f"   peak: {peak_str}",
                  end="", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nFinal peaks (N·m, abs):")
        for i, p in enumerate(peak):
            thresh = THRESHOLDS_NM[i]
            flag = " (exceeded threshold!)" if p > thresh else ""
            print(f"  J{i+1}: {p:5.2f}   threshold: {thresh:5.2f}{flag}")
        print("\nTune FORCE_THRESHOLDS_NM in piper_sequence.py to be modestly above "
              "these peaks — high enough to not false-trip on normal interaction, "
              "low enough to abort before harm.")


if __name__ == "__main__":
    main()
