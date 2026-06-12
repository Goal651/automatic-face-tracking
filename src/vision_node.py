"""
vision_node.py
Simulated Vision Node for Distributed Vision-Control System.
Tracks face and publishes movement commands via MQTT.
Topic: vision/team313/movement
"""

import time
import argparse
import cv2
import json
import numpy as np
try:
    import paho.mqtt.client as mqtt
except Exception as e:
    mqtt = None
    _MQTT_IMPORT_ERROR = e
from pathlib import Path
import sys
import base64

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import Face Locking modules
from src.haar_5pt import Haar5ptDetector
from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz
from src.face_locking import FaceLockSystem, LockState

# Configuration
DEFAULT_BROKER = "192.168.8.101" 
PORT = 1883
TEAM_ID = "dragonfly"
TOPIC_MOVEMENT = f"vision/{TEAM_ID}/movement"
TOPIC_HEARTBEAT = f"vision/{TEAM_ID}/heartbeat"


def _open_any_camera(indices=(0, 1, 2, 3)) -> cv2.VideoCapture:
    """
    Try several camera indices and return the first opened capture.
    Raises RuntimeError if none can be opened.
    """
    for idx in indices:
        cap = cv2.VideoCapture(2)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError(f"Failed to open camera. Tried indices: {list(indices)}")

class VisionNode:
    def __init__(self, broker, port, target_name):
        if mqtt is None:
            raise RuntimeError(
                f"paho-mqtt import failed: {_MQTT_IMPORT_ERROR}\n"
                "Install dependencies with: pip install -r requirements.txt"
            )
        # MQTT Setup
        self.client = mqtt.Client(client_id=f"{TEAM_ID}_vision_node")
        self.client.on_connect = self.on_connect
        self.client.connect(broker, port, 60)
        self.client.loop_start()
        
        # Face Recognition & Locking Setup
        print("Initializing Face Recognition...")
        self.det = Haar5ptDetector(min_size=(70, 70))
        self.embedder = ArcFaceEmbedderONNX(input_size=(112, 112))
        
        # Load Database
        db_path = Path(__file__).parent.parent / "data/db/face_db.npz"
        if not db_path.exists():
            print(f"ERROR: Face DB not found at {db_path}. Run enroll.py first!")
            sys.exit(1)
            
        db = load_db_npz(db_path)
        if target_name not in db:
            print(f"WARNING: Target '{target_name}' not in database. Available: {list(db.keys())}")
        
        self.matcher = FaceDBMatcher(db, dist_thresh=0.60)
        self.system = FaceLockSystem(target_name, self.matcher, self.det)
        
        self.running = True
        self.last_heartbeat = 0
        self.last_publish_time = 0
        self.mqtt_topic = TOPIC_MOVEMENT
        self.snapshot_sent = False
        self.last_status = "CENTERED"
        self.last_published_status = None  # Track what we last sent for full-stop logic

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT Broker with result code {rc}")
        self.publish_heartbeat()

    def publish_movement(self, status, confidence=1.0, target=None, locked=False, face_image=None):
        payload = {
            "status": status,
            "confidence": confidence,
            "target": target,
            "locked": locked,
            "timestamp": time.time()
        }
        
        # Add face image if available
        if face_image is not None:
            _, buffer = cv2.imencode('.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            payload["face_image"] = base64.b64encode(buffer).decode('utf-8')
        
        self.client.publish(self.mqtt_topic, json.dumps(payload))
        print(f"Published: {status} (image: {'yes' if face_image is not None else 'no'})")

    def publish_heartbeat(self):
        payload = {
            "node": "pc_vision",
            "status": "ONLINE",
            "timestamp": time.time()
        }
        self.client.publish(TOPIC_HEARTBEAT, json.dumps(payload))

    def run(self):
        cap = _open_any_camera()
        
        print(f"Vision Node Started. Tracking target: {self.system.target_name}")
        print(f"Publishing to {TOPIC_MOVEMENT}")
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            
            # Process Frame using FaceLockSystem
            vis, target_face, lock_state = self.system.process_frame(frame, self.embedder)
            
            status = "NO_FACE"
            face_crop = None

            if lock_state == LockState.SEARCHING:
                status = "NO_FACE"
                if self.snapshot_sent:
                    self.snapshot_sent = False
                    print("🔓 Target lost - snapshot flag reset")
            elif lock_state == LockState.LOCKED:
                if target_face:
                    f = target_face

                    # Extract face crop for dashboard
                    if not self.snapshot_sent:
                        x1, y1, x2, y2 = int(f.x1), int(f.y1), int(f.x2), int(f.y2)
                        pad = 20
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(W, x2 + pad)
                        y2 = min(H, y2 + pad)
                        face_crop = frame[y1:y2, x1:x2]
                        self.snapshot_sent = True
                        print("📸 Face snapshot captured and will be sent")

                    # Calculate Center
                    cx = (f.x1 + f.x2) / 2.0
                    cx_norm = cx / W

                    # Movement Logic - Deadband: 0.35 to 0.65 is CENTERED
                    if cx_norm < 0.35:
                        status = "MOVE_LEFT"
                    elif cx_norm > 0.65:
                        status = "MOVE_RIGHT"
                    else:
                        status = "CENTERED"

                    self.last_status = status
                else:
                    if self.last_status == "NO_FACE":
                        status = "CENTERED"
                    else:
                        status = self.last_status
            
            # --- SMART RATE LIMITING (FULL STOP ON CENTERED) ---
            current_time = time.time()
            
            if status == "CENTERED":
                # Only publish CENTERED ONCE when entering centered state
                if self.last_published_status != "CENTERED":
                    self.publish_movement(status, target=self.system.target_name, 
                                        locked=True, face_image=face_crop)
                    self.last_publish_time = current_time
                    self.last_published_status = status
                    print("🛑 CENTERED - Servo FULL STOP")
                # else: COMPLETELY SILENT - no publishing
            else:
                # Moving or searching - publish at 5Hz
                if current_time - self.last_publish_time >= 0.2:
                    is_locked = (status != "NO_FACE")
                    self.publish_movement(status, target=self.system.target_name, 
                                        locked=is_locked, face_image=face_crop)
                    self.last_publish_time = current_time
                    self.last_published_status = status
            
            # Heartbeat every 5s
            if time.time() - self.last_heartbeat > 5:
                self.publish_heartbeat()
                self.last_heartbeat = time.time()
            
            cv2.imshow("Vision Node (Locked)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.client.loop_stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER, help="MQTT Broker Address")
    parser.add_argument("--name", type=str, default="wigo", help="Target name to lock onto")
    args = parser.parse_args()

    node = VisionNode(args.broker, PORT, args.name)
    node.run()