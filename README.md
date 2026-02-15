Virtual Mouse using Computer Vision

A real-time AI-powered virtual mouse that allows users to control their computer cursor using hand gestures through a webcam.
Built using Python, OpenCV, and MediaPipe, this system enables touchless human-computer interaction with low latency and smooth tracking.

Features

👆 Cursor movement using index finger tracking
🖱️ Left-click gesture detection
🖱️ Right-click gesture detection
👆👆 Double-click gesture
📸 Screenshot gesture
⚡ Low-latency real-time tracking
🎯 Gesture smoothing for stable pointer movement


How It Works

Hand Detection
MediaPipe detects 21 hand landmarks in real time.

Landmark Tracking
The system tracks specific landmarks (e.g., index fingertip) to determine gesture states.

Gesture Recognition Logic
Finger distance calculations determine click events.
Multi-hand detection differentiates left and right hands.
Temporal smoothing reduces jitter.

Cursor Mapping
Camera frame coordinates are mapped to screen resolution.
Interpolation improves pointer smoothness.


Tech Stack:

Python,
OpenCV,
MediaPi,pe
PyAutoGU,I
NumPy


Future Improvements:

Scroll gesture implementation,
Custom gesture configuration,
ML-based adaptive gesture sensitivity,
Cross-platform packaging
