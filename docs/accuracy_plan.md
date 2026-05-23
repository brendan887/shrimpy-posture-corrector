# Flexion and Abduction Accuracy Plan

This MVP starts with a front-facing single-camera setup. The goal is to improve useful shoulder flexion and abduction feedback in small, testable steps before considering angled or multi-camera setups.

## Step 1: Improve Current Single-Camera Robustness

- Smooth underlying torso and arm vectors before computing angles, not just the final displayed numbers.
- Use confidence-aware updates so low-confidence landmarks do not overwrite stable measurements.
- Keep the last good angle briefly when landmarks are missing instead of flickering to empty values.
- Show simple UI status when measurements are waiting for visibility or calibration.

## Step 2: Front-View Calibration Poses

- Let the user press `c` to start calibration.
- Capture three poses: neutral arms down, forward raise, and side raise.
- Use visibility and stillness checks before capture.
- Show instructions and a countdown in the UI.
- Allow `Space` as a manual capture fallback for each calibration pose.
- Use averaged landmarks from each captured pose to define body-relative down, forward, and side axes for the session.

## Step 3: 45-Degree View Guidance

- Keep front view as the default MVP camera angle.
- Test `left-45` and `right-45` camera angles for improved flexion.
- Use front-view mode for abduction-heavy checks.
- Use 45-degree modes when flexion is the priority.
- Keep the math torso-relative first; only add view-specific geometry if repeated tests show a consistent bias.

## Step 4: Later Hardware Expansion

- Try iPhone Continuity Camera as a better single external camera.
- Consider two-camera simple fusion before true stereo triangulation.
- Only move to calibrated stereo if the simpler improvements are not accurate enough.
