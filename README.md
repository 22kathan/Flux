# ⚡ FLUX

**Real-time interactive hand tracking with neon laser beams, air writing, and OS control.**

Built with **MediaPipe Hands**, **WebSockets**, and **Python (PyAutoGUI)**, **Flux** transforms your webcam into a powerful input device, allowing you to interact with your computer using nothing but gestures.

---

## ✨ Key Features

-   **⚡ Laser Beams:** Dynamic, neon-colored laser beams that track your fingertips in real-time with zero lag.
-   **✏️ Air Writing:** Switch to writing mode and draw directly on your screen by pointing your index finger.
-   **🖱️ OS Control:** Control your mouse cursor, perform left clicks, and scroll through pages using intuitive hand gestures.
-   **🎨 Premium UI:** A polished, "Dark Tech" aesthetic featuring CRT scanlines, glassmorphism, and smooth animations.
-   **📸 Screenshot Mode:** Capture your hand-tracking art with a single button or keypress.

---

## 🎮 How to Use

### **Hand Gestures**

| Action | Gesture |
| :--- | :--- |
| **Move Cursor** | Point with your **Index Finger** |
| **Left Click** | **Pinch** Thumb + Index Finger |
| **Scroll** | **Pinch & Hold** Thumb + Middle Finger and move up/down |
| **Draw (Writing Mode)** | Point with **Index Finger** (tracks stroke) |
| **Clear Drawing** | Close your hand into a **Fist** or press `C` |

### **Keyboard Shortcuts**

-   `1`: Laser Mode
-   `2`: Writing Mode
-   `3`: OS Control Mode
-   `F`: Toggle Fullscreen
-   `S`: Take Screenshot
-   `C`: Clear Canvas (Writing Mode)

---

## 🚀 Getting Started

**Flux** is designed to be easy to set up on any Windows machine.

### **Quick Setup (Recommended)**

1.  Download the latest [Setup-and-Launch-Flux.bat](https://github.com/22kathan/laser-hands/releases).
2.  Right-click the file and select **"Run as Administrator"**.
3.  The script will automatically install Python (if needed), set up the environment, and launch the application.

> [!IMPORTANT]
> For a detailed walkthrough and troubleshooting, please refer to the [USER-SETUP-GUIDE.md](USER-SETUP-GUIDE.md).

---

## 🛠️ Tech Stack

-   **Frontend:** HTML5, Vanilla CSS, JavaScript (ES6+).
-   **Tracking:** [Google MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands).
-   **Backend (OS Control):** Python 3.11+ using `pyautogui` and `websockets`.
-   **Aesthetics:** Custom HSL color palettes, Glassmorphism, and CRT post-processing effects.

---

## 📂 Project Structure

-   `index.html`: The main web interface and tracking logic.
-   `os_controller.py`: The Python backend that receives tracking data and controls the OS.
-   `USER-SETUP-GUIDE.md`: Comprehensive guide for new users.
-   `Launch Flux.bat`: Quick launcher for returning users.

---

## 📜 License

This project is open-source. Feel free to explore, modify, and build upon it!

---

*Developed with ❤️ by [Kathan Gadhiya](https://22kathan.github.io/portfolio)*
