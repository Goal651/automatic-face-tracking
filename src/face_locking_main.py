"""
Main entry point for Face Locking Application

This is the refactored version that uses the modular framework:
- Face tracking framework for core functionality
- Servo controller for MQTT communication
- Application layer for UI and integration

Usage:
    python -m src.face_locking_main

Controls:
    q: quit
    r: reload database
    l: toggle lock on/off for target identity
    +/-: adjust smile detection threshold
    F1/F2: adjust face detection sensitivity
    m: toggle mirror mode (natural camera view)
    M: toggle landmarks display
    C: toggle confidence display
    d: toggle detailed UI information
    [/]: adjust window scaling
    s: save current action history to ./history/
    p: toggle MQTT publishing
"""

from __future__ import annotations

import cv2
import time
from pathlib import Path

from .face_locking_app import FaceLockingApp
from .servo_controller import ServoConfig


def main():
    """Main application entry point"""
    
    # Configuration
    target_identity = "Wilson"
    db_path = "data/db/face_db.npz"
    model_path = "models/embedder_arcface.onnx"
    window_scale = 1.0
    mirror_mode = True
    
    # MQTT configuration
    mqtt_config = ServoConfig(
        broker="localhost",
        port=1883,
        topic_movement="vision/team351/movement",
        topic_status="robot/status",
        topic_servo_status="servo/status",
        servo_Kp=60.0,
        servo_ema_alpha=0.45,
        auto_stop_on_lock=True,
        scan_enabled=True,
        scan_speed=30.0
    )
    
    # Initialize application
    app = FaceLockingApp(
        target_identity=target_identity,
        db_path=db_path,
        model_path=model_path,
        window_scale=window_scale,
        mirror_mode=mirror_mode,
        mqtt_config=mqtt_config
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Face Locking Application started")
    print("Controls:")
    print("  q: quit")
    print("  r: reload database")
    print("  l: toggle lock on/off")
    print("  +/-: adjust smile detection threshold")
    print("  F1/F2: adjust face detection sensitivity")
    print("  m: toggle mirror mode")
    print("  M: toggle landmarks display")
    print("  C: toggle confidence display")
    print("  d: toggle detailed UI information")
    print("  [/]: adjust window scaling")
    print("  s: save action history")
    print("  p: toggle MQTT publishing")
    
    # Main loop
    last_time = time.time()
    frame_count = 0
    fps = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Process frame
            processed_frame = app.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            
            # Add FPS overlay
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            window_name = "Face Locking System"
            cv2.imshow(window_name, processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reload database
                try:
                    app.tracker.matcher.reload_from(Path(db_path))
                    app.tracker.identity_cache.clear()
                    print("📁 Database reloaded")
                except Exception as e:
                    print(f"❌ Failed to reload database: {e}")
            
            elif key == ord('l'):
                # Toggle lock
                app.toggle_lock()
            
            elif key in (ord('+'), ord('=')):
                # Increase smile detection threshold
                current_threshold = app.tracker.action_detector.smile_ratio_threshold
                new_threshold = min(2.0, current_threshold + 0.1)
                app.tracker.action_detector.smile_ratio_threshold = new_threshold
                print(f"😊 Smile threshold: {new_threshold:.2f}")
            
            elif key == ord('-'):
                # Decrease smile detection threshold
                current_threshold = app.tracker.action_detector.smile_ratio_threshold
                new_threshold = max(0.5, current_threshold - 0.1)
                app.tracker.action_detector.smile_ratio_threshold = new_threshold
                print(f"😊 Smile threshold: {new_threshold:.2f}")
            
            elif key == ord('m'):
                # Toggle mirror mode
                app.tracking_config.mirror_mode = not app.tracking_config.mirror_mode
                status = "ON" if app.tracking_config.mirror_mode else "OFF"
                print(f"🪞 Mirror mode: {status}")
            
            elif key == ord('M'):
                # Toggle landmarks display
                app.show_landmarks = not app.show_landmarks
                status = "ON" if app.show_landmarks else "OFF"
                print(f"📍 Landmarks: {status}")
            
            elif key == ord('C'):
                # Toggle confidence display
                app.show_confidence = not app.show_confidence
                status = "ON" if app.show_confidence else "OFF"
                print(f"📊 Confidence: {status}")
            
            elif key == ord('d'):
                # Toggle detailed UI
                app.show_detailed_info = not app.show_detailed_info
                status = "ON" if app.show_detailed_info else "OFF"
                print(f"📋 Detailed info: {status}")
            
            elif key == ord('['):
                # Decrease window scale
                app.tracking_config.window_scale = max(0.5, app.tracking_config.window_scale - 0.1)
                print(f"🔍 Window scale: {app.tracking_config.window_scale:.1f}x")
            
            elif key == ord(']'):
                # Increase window scale
                app.tracking_config.window_scale = min(2.0, app.tracking_config.window_scale + 0.1)
                print(f"🔍 Window scale: {app.tracking_config.window_scale:.1f}x")
            
            elif key == ord('s'):
                # Save action history
                try:
                    filepath = app.save_action_history()
                    print(f"💾 Action history saved")
                except Exception as e:
                    print(f"❌ Failed to save history: {e}")
            
            elif key == ord('p'):
                # Toggle MQTT publishing
                if app.servo.enabled:
                    app.servo.disable()
                    print("📡 MQTT publishing: OFF")
                else:
                    app.servo.enable()
                    print("📡 MQTT publishing: ON")
            
            # F1/F2 for face detection sensitivity
            elif key == 0xFFBE:  # F1
                # Decrease face detection sensitivity (increase min_size)
                current_min = app.tracker.detector.min_size
                new_min = (max(50, current_min[0] - 10), max(50, current_min[1] - 10))
                app.tracker.detector.min_size = new_min
                print(f"🎯 Face sensitivity: LESS sensitive (min_size={new_min})")
            
            elif key == 0xFFBF:  # F2
                # Increase face detection sensitivity (decrease min_size)
                current_min = app.tracker.detector.min_size
                new_min = (max(30, current_min[0] - 10), max(30, current_min[1] - 10))
                app.tracker.detector.min_size = new_min
                print(f"🎯 Face sensitivity: MORE sensitive (min_size={new_min})")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Cleanup
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        app.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
