"""
Face Locking Feature Implementation with MQTT Control

This module extends the face recognition system with face locking capabilities:
- Manual face selection for a specific enrolled identity
- Stable face tracking across frames
- Action detection (movement, blinks, smiles)
- MQTT publishing to control servo based on face position
- Action history recording to timestamped files in ./history/
- Display all detected faces with status (LOCKED/UNLOCKED/UNKNOWN)

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

import re
import time
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# Import existing modules
from .recognize import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    FaceDet,
    MatchResult,
    load_db_npz,
    _kps_span_ok,
)
from .haar_5pt import align_face_5pt
from .action_detection import AdvancedActionDetector, ActionClassifier


@dataclass
class MQTTConfig:
    """MQTT configuration"""

    broker: str = "localhost"  # or "127.0.0.1" for local broker
    port: int = 1883
    topic_movement: str = "vision/team351/movement"
    topic_status: str = "robot/status"
    topic_servo_status: str = "servo/status"  # ESP8266 feedback topic
    client_id: str = f"face_locking_{int(time.time())}"
    publish_interval: float = 0.1  # seconds between publishes
    confidence_threshold: float = 0.7
    # Servo angle range (degrees)
    servo_min_angle: int = 0
    servo_max_angle: int = 180
    servo_center_angle: int = 90  # angle sent when no face is detected


@dataclass
class ActionRecord:
    """Single action record with timestamp"""

    timestamp: str
    action_type: str
    description: str
    value: Optional[float] = None


@dataclass
class FaceTracker:
    """Tracks a locked face across frames"""

    identity: str
    last_position: Tuple[int, int]  # center (x, y)
    last_seen_frame: int
    confidence_history: deque  # recent recognition confidences
    position_history: deque  # recent positions for movement detection
    blink_state: str  # "open", "closed", "unknown"
    blink_counter: int
    smile_state: bool
    lock_start_time: float

    def __post_init__(self):
        if not hasattr(self, "confidence_history") or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, "position_history") or self.position_history is None:
            self.position_history = deque(maxlen=5)


class FaceLockingSystem:
    """Main face locking system with MQTT control"""

    def __init__(
        self,
        target_identity: str = "Wilson",
        db_path: str = "data/db/face_db.npz",
        model_path: str = "models/embedder_arcface.onnx",
        window_scale: float = 1.0,
        mirror_mode: bool = True,
        mqtt_config: Optional[MQTTConfig] = None,
    ):
        self.target_identity = target_identity
        self.db_path = Path(db_path)
        self.window_scale = window_scale
        self.mirror_mode = mirror_mode

        # MQTT settings
        self.mqtt_enabled = True
        self.mqtt_config = mqtt_config or MQTTConfig()
        self.last_mqtt_publish = 0
        self.last_mqtt_status = None  # Track last published status to avoid duplicates

        # Servo angle tracking
        self._current_servo_angle: float = float(self.mqtt_config.servo_center_angle)
        self._last_published_angle: int = -1
        self._target_servo_angle: Optional[float] = None
        self._centering_tolerance: float = 0.05  # 5% tolerance for centering
        self._centered_stable_frames: int = 0
        self._centering_stability_threshold: int = 5  # frames to confirm centered

        # P-controller gain: how many degrees to adjust per unit of normalised error
        # error is -1 (face fully left) to +1 (face fully right)
        # Increase Kp if the servo is slow to center; decrease if it overshoots
        self._servo_Kp: float = 60.0
        
        # Auto-stop when locked: immediately center and hold on first lock
        self._auto_stop_on_lock: bool = True
        self._lock_centering_complete: bool = False

        # EMA smoothing on the P-controller output (0 = frozen, 1 = no smoothing)
        self._servo_ema_alpha: float = 0.45

        # Scan-sweep state (active when no face is locked)
        self._scan_angle: float = float(self.mqtt_config.servo_center_angle)
        self._scan_direction: float = 1.0  # +1 = increasing angle, -1 = decreasing
        self._scan_speed: float = 30.0  # degrees per second
        self._scan_min: float = 20.0  # leftmost sweep angle
        self._scan_max: float = 160.0  # rightmost sweep angle
        self._scan_last_time: float = time.time()
        self._scanning: bool = False  # True while no face is locked

        # UI settings
        self.show_detailed_info = True
        self.show_landmarks = True
        self.show_confidence = True
        self.ui_alpha = 0.8

        # Face detection settings  (kept close to align.py defaults for flexibility)
        self.min_face_size = (70, 70)
        self.max_face_size = (800, 800)
        self.face_aspect_ratio_range = (0.5, 2.0)

        # Initialize components
        self.detector = HaarFaceMesh5pt(min_size=self.min_face_size, debug=False)
        self.embedder = ArcFaceEmbedderONNX(model_path=model_path, debug=False)

        # Load database and matcher
        db = load_db_npz(self.db_path)
        self.matcher = FaceDBMatcher(db=db, dist_thresh=0.4)

        # Face tracking & Recognition optimization
        self.tracker: Optional[FaceTracker] = None
        self.is_locked = False
        self.frame_count = 0
        self.lock_timeout = 30  # Restore original timeout
        self.recognition_interval = 10  # Restore original recognition interval
        self.lock_on_first_detection = False  # Disable lock-on-first-detection mode

        # Identity cache
        self.identity_cache = {}
        self.cache_distance_threshold = 80

        # Action detection
        self.action_detector = AdvancedActionDetector()
        self.action_classifier = ActionClassifier()
        self.action_history: List[ActionRecord] = []

        # UI state
        self.last_ui_update = time.time()
        self.ui_update_interval = 0.1

        # Face validation cache
        self.face_validation_cache = {}
        self.cache_timeout = 2.0

        # Initialize MQTT
        self._init_mqtt()

        # Verify target identity exists
        if target_identity not in db:
            available = list(db.keys())
            raise ValueError(
                f"Target identity '{target_identity}' not found in database. "
                f"Available: {available}"
            )

        print(f"Face Locking System initialized for: {target_identity}")
        print(f"Database contains {len(db)} identities: {list(db.keys())}")
        print(f"Window scale: {window_scale}x")
        print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
        print(f"MQTT: {'ENABLED' if self.mqtt_enabled else 'DISABLED'}")
        print(f"MQTT Broker: {self.mqtt_config.broker}:{self.mqtt_config.port}")
        print(f"MQTT Topic: {self.mqtt_config.topic_movement}")

    def _init_mqtt(self):
        """Initialize MQTT connection and subscribe to servo feedback"""
        self.mqtt_client = mqtt.Client(client_id=self.mqtt_config.client_id)

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"✅ Connected to MQTT broker at {self.mqtt_config.broker}")
                self._publish_status("ONLINE", "Face locking system started")
                # Subscribe to servo status so we can track its real angle
                client.subscribe(self.mqtt_config.topic_servo_status)
                print(
                    f"✅ Subscribed to servo feedback: {self.mqtt_config.topic_servo_status}"
                )
            else:
                print(f"❌ Failed to connect to MQTT broker, return code {rc}")
                self.mqtt_enabled = False

        def on_message(client, userdata, msg):
            """Parse servo/status feedback to keep _current_servo_angle in sync."""
            try:
                payload = msg.payload.decode("utf-8")
                # Matches both 'Moved to 95° ...' and 'Current angle: 95° ...'
                m = re.search(r"(\d+)\u00b0", payload)
                if m:
                    new_angle = float(m.group(1))
                    # Only update if we have a significant change to avoid jitter
                    if abs(new_angle - self._current_servo_angle) > 0.5:
                        self._current_servo_angle = new_angle
                        if self.show_detailed_info:
                            print(f"📥 Servo feedback: {new_angle:.1f}°")
            except Exception:
                pass

        def on_disconnect(client, userdata, rc):
            print("⚠️ Disconnected from MQTT broker")
            if rc != 0:
                print("Attempting to reconnect...")
                try:
                    client.reconnect()
                except Exception:
                    pass

        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        self.mqtt_client.on_disconnect = on_disconnect

        try:
            self.mqtt_client.connect(self.mqtt_config.broker, self.mqtt_config.port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"❌ MQTT connection error: {e}")
            self.mqtt_enabled = False

    def _face_position_to_servo_angle(
        self, face_center_x: int, frame_width: int
    ) -> int:
        """
        Enhanced P-controller with centering detection and servo feedback integration.
        
        normalised_error = (face_x - frame_center_x) / half_width
          +1 = face fully at right edge
          -1 = face fully at left edge
           0 = face perfectly centered  → no adjustment needed

        In mirror mode the pixel axis is inverted relative to physical space,
        so we negate the error direction.

        The computed target angle is EMA-smoothed to reduce jitter.
        When face is centered within tolerance, stops sending commands.
        """
        half_w = frame_width / 2.0
        error = (face_center_x - half_w) / half_w  # −1 … +1

        # Mirror: pixel-right = physical-left → flip the sign
        if self.mirror_mode:
            error = -error

        # Check if face is already centered within tolerance
        if abs(error) <= self._centering_tolerance:
            self._centered_stable_frames += 1
            if self._centered_stable_frames >= self._centering_stability_threshold:
                # Face is stably centered - don't adjust servo
                return int(round(self._current_servo_angle))
        else:
            self._centered_stable_frames = 0

        correction = self._servo_Kp * error
        target = self._current_servo_angle + correction

        cfg = self.mqtt_config
        target = float(np.clip(target, cfg.servo_min_angle, cfg.servo_max_angle))

        # EMA smoothing
        smoothed = (
            self._servo_ema_alpha * target
            + (1.0 - self._servo_ema_alpha) * self._current_servo_angle
        )
        angle = int(round(np.clip(smoothed, cfg.servo_min_angle, cfg.servo_max_angle)))
        return angle

    def _publish_movement(
        self,
        status: str,
        confidence: float,
        angle: Optional[int] = None,
        force: bool = False,
    ):
        """Publish face movement + servo angle to MQTT"""
        if not self.mqtt_enabled:
            return

        # Rate limiting (bypass when force=True, e.g. on first lock)
        current_time = time.time()
        if (
            not force
            and current_time - self.last_mqtt_publish
            < self.mqtt_config.publish_interval
        ):
            return

        # Skip duplicate angle unless forced or status changed
        if not force and angle is not None:
            if status == self.last_mqtt_status and angle == self._last_published_angle:
                return
        elif not force:
            if status == self.last_mqtt_status:
                return

        # Mirror-adjust the direction label (P-controller already handles mirror)
        display_status = status
        if self.mirror_mode and status not in ("NO_FACE", "CENTERED", "SCANNING", "PERFECTLY_CENTERED", "AUTO_STOP_COMPLETE"):
            display_status = "MOVE_RIGHT" if status == "MOVE_LEFT" else "MOVE_LEFT"

        servo_angle = (
            angle if angle is not None else self.mqtt_config.servo_center_angle
        )

        # Minimal message - only essential fields
        message = {
            "status": display_status,
            "angle": servo_angle,
            "confidence": float(confidence),
        }

        try:
            self.mqtt_client.publish(
                self.mqtt_config.topic_movement, json.dumps(message)
            )
            self.last_mqtt_publish = current_time
            self.last_mqtt_status = status
            if angle is not None:
                self._last_published_angle = angle
                self._current_servo_angle = float(angle)  # track what we sent

            if self.show_detailed_info:
                print(
                    f"Status: {display_status} | Angle: {servo_angle}° | Confidence: {confidence:.2f}"
                )

        except Exception as e:
            print(f"❌ MQTT publish error: {e}")

    def _publish_status(self, status: str, message: str):
        """Publish system status to MQTT"""
        if not self.mqtt_enabled:
            return

        status_msg = {
            "status": status,
            "message": message,
            "timestamp": int(time.time()),
            "target": self.target_identity if self.is_locked else "none",
        }

        try:
            self.mqtt_client.publish(
                self.mqtt_config.topic_status, json.dumps(status_msg)
            )
        except:
            pass

    def _publish_scan_sweep(self):
        """
        Advance the sweep oscillation and publish the next scan angle.
        Called every frame while not locked.  The servo sweeps smoothly
        between _scan_min and _scan_max at _scan_speed deg/s.
        """
        now = time.time()
        dt = now - self._scan_last_time
        self._scan_last_time = now

        # Advance angle
        self._scan_angle += self._scan_direction * self._scan_speed * dt

        # Bounce at limits
        if self._scan_angle >= self._scan_max:
            self._scan_angle = self._scan_max
            self._scan_direction = -1.0
        elif self._scan_angle <= self._scan_min:
            self._scan_angle = self._scan_min
            self._scan_direction = 1.0

        angle = int(round(self._scan_angle))
        if not self.mqtt_enabled:
            return

        now2 = time.time()
        if now2 - self.last_mqtt_publish < self.mqtt_config.publish_interval:
            return

        # Minimal message for scanning
        message = {
            "status": "SCANNING",
            "angle": angle,
            "confidence": 0.0,
        }
        try:
            self.mqtt_client.publish(
                self.mqtt_config.topic_movement, json.dumps(message)
            )
            self.last_mqtt_publish = now2
            self._last_published_angle = angle
        except Exception as e:
            print(f"❌ MQTT scan publish error: {e}")

    def _get_face_position_status(
        self, face_center: Tuple[int, int], frame_shape: Tuple[int, int]
    ) -> str:
        """
        Determine face position relative to frame with enhanced centering detection.
        Returns: "MOVE_LEFT", "CENTERED", "MOVE_RIGHT", or "PERFECTLY_CENTERED"
        """
        frame_width = frame_shape[1]

        # Calculate relative position (0 to 1, where 0.5 is center)
        rel_position = face_center[0] / frame_width
        center_distance = abs(rel_position - 0.5)

        # Define thresholds (adjust these for sensitivity)
        left_threshold = 0.4  # 40% from left edge
        right_threshold = 0.6  # 60% from left edge
        perfect_center_threshold = 0.02  # 2% from perfect center

        if center_distance <= perfect_center_threshold:
            return "PERFECTLY_CENTERED"
        elif rel_position < left_threshold:
            return "MOVE_LEFT"
        elif rel_position > right_threshold:
            return "MOVE_RIGHT"
        else:
            return "CENTERED"

    def _validate_face_detection(self, face: FaceDet, frame: np.ndarray) -> bool:
        """Validate face detection to filter out false positives like mouths"""
        # ... (keep your existing validation code)
        face_width = face.x2 - face.x1
        face_height = face.y2 - face.y1

        # Size validation
        if face_width < self.min_face_size[0] or face_height < self.min_face_size[1]:
            return False

        if face_width > self.max_face_size[0] or face_height > self.max_face_size[1]:
            return False

        # Aspect ratio validation
        aspect_ratio = face_width / face_height
        if not (
            self.face_aspect_ratio_range[0]
            <= aspect_ratio
            <= self.face_aspect_ratio_range[1]
        ):
            return False

        # Landmark validation
        if hasattr(face, "kps") and face.kps is not None:
            landmarks = face.kps

            if len(landmarks) != 5:
                return False

            # Validate landmark positions
            for lm in landmarks:
                x, y = lm
                if not (
                    face.x1 - 10 <= x <= face.x2 + 10
                    and face.y1 - 10 <= y <= face.y2 + 10
                ):
                    return False

            if not _kps_span_ok(landmarks, min_eye_dist=face_height * 0.1):
                return False

            left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            if mouth_center_y <= eye_center_y:
                return False

            if not (eye_center_y < nose[1] < mouth_center_y):
                return False

            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            if mouth_width > face_width * 0.8:
                return False

        # Position validation
        frame_h, frame_w = frame.shape[:2]
        face_center_x = (face.x1 + face.x2) / 2
        face_center_y = (face.y1 + face.y2) / 2

        edge_margin = 5  # very small — allow faces close to frame edges
        if (
            face_center_x < edge_margin
            or face_center_x > frame_w - edge_margin
            or face_center_y < edge_margin
            or face_center_y > frame_h - edge_margin
        ):
            return False

        return True

    def _get_face_center(self, face: FaceDet) -> Tuple[int, int]:
        """Get center point of face bounding box"""
        center_x = (face.x1 + face.x2) // 2
        center_y = (face.y1 + face.y2) // 2
        return (center_x, center_y)

    def _is_same_face(self, face: FaceDet, tracker: FaceTracker) -> bool:
        """Check if detected face matches tracked face based on position"""
        current_center = self._get_face_center(face)
        last_center = tracker.last_position

        distance = np.sqrt(
            (current_center[0] - last_center[0]) ** 2
            + (current_center[1] - last_center[1]) ** 2
        )

        return distance < 100

    def _get_cached_identity(self, face: FaceDet) -> Optional[MatchResult]:
        """Try to retrieve recognized identity from cache"""
        center = self._get_face_center(face)

        current_frame = self.frame_count
        self.identity_cache = {
            pos: val
            for pos, val in self.identity_cache.items()
            if current_frame - val[1] < 5
        }

        for pos, (result, frame) in self.identity_cache.items():
            dist = np.sqrt((center[0] - pos[0]) ** 2 + (center[1] - pos[1]) ** 2)
            if dist < self.cache_distance_threshold:
                return result
        return None

    def _update_cache(self, face: FaceDet, result: MatchResult):
        """Store recognized identity in cache"""
        center = self._get_face_center(face)
        self.identity_cache[center] = (result, self.frame_count)

    def _record_action(
        self, action_type: str, description: str, value: Optional[float] = None
    ):
        """Record an action to history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        action = ActionRecord(
            timestamp=timestamp,
            action_type=action_type,
            description=description,
            value=value,
        )
        self.action_history.append(action)
        print(f"[ACTION] {action.timestamp}: {action.description}")

    def _save_action_history(self) -> str:
        """Save action history to file in history folder"""
        if not self.action_history:
            print("No actions to save")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{self.target_identity.lower()}_history_{timestamp}.txt"
        filepath = Path("history") / filename

        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, "w") as f:
            f.write("Face Locking Action History\n")
            f.write(f"Target Identity: {self.target_identity}\n")
            f.write(
                f"Session Start: {self.action_history[0].timestamp if self.action_history else 'N/A'}\n"
            )
            f.write(
                f"Session End: {self.action_history[-1].timestamp if self.action_history else 'N/A'}\n"
            )
            f.write(f"Total Actions: {len(self.action_history)}\n")
            f.write("-" * 50 + "\n\n")

            for action in self.action_history:
                line = (
                    f"{action.timestamp} | {action.action_type} | {action.description}"
                )
                if action.value is not None:
                    line += f" | Value: {action.value:.3f}"
                f.write(line + "\n")

        print(f"Action history saved to: {filepath}")
        return str(filepath)

    def _create_ui_overlay(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create a semi-transparent overlay for UI elements"""
        h, w = frame_shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 160), (w, h), (0, 0, 0), -1)

        if self.show_detailed_info:
            cv2.rectangle(overlay, (0, 120), (300, h - 160), (0, 0, 0), -1)

        return overlay

    def _draw_clean_text(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 0.6,
        thickness: int = 1,
    ):
        """Draw text with background for better readability"""
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        cv2.rectangle(
            img,
            (pos[0] - 2, pos[1] - text_h - 2),
            (pos[0] + text_w + 2, pos[1] + baseline + 2),
            (0, 0, 0),
            -1,
        )

        cv2.putText(img, text, pos, font, scale, color, thickness)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for face locking with MQTT publishing"""
        self.frame_count += 1

        if self.mirror_mode:
            frame = cv2.flip(frame, 1)

        if self.window_scale != 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * self.window_scale), int(h * self.window_scale)
            frame = cv2.resize(frame, (new_w, new_h))

        vis = frame.copy()

        current_time = time.time()
        if current_time - self.last_ui_update > self.ui_update_interval:
            self.last_ui_update = current_time

        detected_faces = self.detector.detect(frame, max_faces=5)

        valid_faces = []
        for face in detected_faces:
            if self._validate_face_detection(face, frame):
                valid_faces.append(face)
            elif self.show_detailed_info:
                cv2.rectangle(
                    vis, (face.x1, face.y1), (face.x2, face.y2), (128, 128, 128), 1
                )
                self._draw_clean_text(
                    vis, "REJECTED", (face.x1, face.y1 - 5), (128, 128, 128), 0.3, 1
                )

        if self.is_locked and self.tracker:
            self._scanning = False
            self._update_locked_face(frame, valid_faces, vis)
        else:
            # No lock — sweep the servo to search
            if not self._scanning:
                self._scanning = True
                self._scan_last_time = time.time()  # reset timer to avoid big dt jump
            self._publish_scan_sweep()
            self._search_for_target(frame, valid_faces, vis)

        self._draw_clean_ui(vis)

        return vis

    def _update_locked_face(
        self, frame: np.ndarray, faces: List[FaceDet], vis: np.ndarray
    ):
        """Update tracking for locked face with simplified validation"""
        target_face = None

        # If lock_on_first_detection is enabled, just track by position without re-recognition
        if self.lock_on_first_detection:
            # Find face closest to last tracked position
            if self.tracker and faces:
                best_face = None
                min_distance = float('inf')
                
                for face in faces:
                    current_center = self._get_face_center(face)
                    last_center = self.tracker.last_position
                    distance = np.sqrt(
                        (current_center[0] - last_center[0]) ** 2
                        + (current_center[1] - last_center[1]) ** 2
                    )
                    
                    if distance < min_distance and distance < 150:  # 150px tolerance
                        min_distance = distance
                        best_face = face
                
                if best_face:
                    target_face = best_face
                    # Update tracker position
                    current_center = self._get_face_center(target_face)
                    self.tracker.last_position = current_center
                    self.tracker.last_seen_frame = self.frame_count
                    self.tracker.position_history.append(current_center)
                    
                    # Continue servo control
                    position_status = self._get_face_position_status(
                        current_center, frame.shape
                    )
                    avg_confidence = 0.8  # Default confidence when locked
                    
                    if position_status == "PERFECTLY_CENTERED" or (self._auto_stop_on_lock and self._lock_centering_complete):
                        servo_angle = int(round(self._current_servo_angle))
                        if position_status == "PERFECTLY_CENTERED" and not self._lock_centering_complete:
                            self._lock_centering_complete = True
                            self._publish_movement("AUTO_STOP_COMPLETE", avg_confidence, angle=servo_angle, force=True)
                            print(f"✅ AUTO-STOP COMPLETE - Face centered at {servo_angle}°")
                        else:
                            self._publish_movement("PERFECTLY_CENTERED", avg_confidence, angle=servo_angle)
                    else:
                        servo_angle = self._face_position_to_servo_angle(
                            current_center[0], frame.shape[1]
                        )
                        self._publish_movement(position_status, avg_confidence, angle=servo_angle)
                    
                    self._draw_locked_face(vis, target_face)
                else:
                    # No face found nearby, check timeout
                    frames_since_seen = self.frame_count - self.tracker.last_seen_frame
                    if frames_since_seen > self.lock_timeout:
                        self._release_lock()
                        self._record_action(
                            "lock_lost", f"Face disappeared for {frames_since_seen} frames"
                        )
                        self._current_servo_angle = float(self.mqtt_config.servo_center_angle)
                        self._publish_movement(
                            "NO_FACE", 0.0, angle=self.mqtt_config.servo_center_angle
                        )
            return
        
        # Original recognition-based tracking (when lock_on_first_detection is False)
        for face in faces:
            is_possible_target = self._is_same_face(face, self.tracker)
            needs_recognition = self.frame_count % self.recognition_interval == 0

            match_result = None
            if is_possible_target and not needs_recognition:
                match_result = MatchResult(
                    name=self.target_identity,
                    distance=0.0,
                    similarity=np.mean(list(self.tracker.confidence_history))
                    if self.tracker.confidence_history
                    else 0.8,
                    accepted=True,
                )
            else:
                match_result = self._get_cached_identity(face)

            if match_result is None:
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                emb = self.embedder.embed(aligned)
                match_result = self.matcher.match(emb)
                self._update_cache(face, match_result)

            is_locked_target = (
                is_possible_target
                and match_result.accepted
                and match_result.name == self.target_identity
            )

            if is_locked_target:
                target_face = face
                if needs_recognition:
                    self.tracker.confidence_history.append(match_result.similarity)
            else:
                self._draw_unlocked_face(vis, face, match_result)

        if target_face:
            current_center = self._get_face_center(target_face)
            self.tracker.last_position = current_center
            self.tracker.last_seen_frame = self.frame_count
            self.tracker.position_history.append(current_center)

            # Determine face position + continuous servo angle and publish to MQTT
            position_status = self._get_face_position_status(
                current_center, frame.shape
            )
            
            # Only compute and publish servo angle if not perfectly centered or auto-stop not complete
            avg_confidence = (
                np.mean(list(self.tracker.confidence_history))
                if self.tracker.confidence_history
                else 0.8
            )
            
            if position_status == "PERFECTLY_CENTERED" or (self._auto_stop_on_lock and self._lock_centering_complete):
                # Face is perfectly centered or auto-stop is complete - maintain current angle
                servo_angle = int(round(self._current_servo_angle))
                
                # Mark centering as complete on first perfect center
                if position_status == "PERFECTLY_CENTERED" and not self._lock_centering_complete:
                    self._lock_centering_complete = True
                    self._publish_movement("AUTO_STOP_COMPLETE", avg_confidence, angle=servo_angle, force=True)
                    print(f"✅ AUTO-STOP COMPLETE - Face centered at {servo_angle}°")
                else:
                    self._publish_movement("PERFECTLY_CENTERED", avg_confidence, angle=servo_angle)
                    
                if self.show_detailed_info:
                    print(f"🎯 PERFECTLY CENTERED - Servo holding at {servo_angle}°")
            else:
                # Compute servo adjustment
                servo_angle = self._face_position_to_servo_angle(
                    current_center[0], frame.shape[1]
                )
                self._publish_movement(position_status, avg_confidence, angle=servo_angle)

            self._detect_actions(target_face)
            self._draw_locked_face(vis, target_face)

        else:
            frames_since_seen = self.frame_count - self.tracker.last_seen_frame
            if frames_since_seen > self.lock_timeout:
                self._release_lock()
                self._record_action(
                    "lock_lost", f"Face disappeared for {frames_since_seen} frames"
                )
                # Publish NO_FACE + center angle so servo returns to rest
                self._current_servo_angle = float(self.mqtt_config.servo_center_angle)
                self._publish_movement(
                    "NO_FACE", 0.0, angle=self.mqtt_config.servo_center_angle
                )

    def _draw_unlocked_face(
        self, vis: np.ndarray, face: FaceDet, match_result: MatchResult
    ):
        """Draw unlocked faces"""
        if match_result.accepted:
            if match_result.name == self.target_identity:
                color = (0, 255, 255)
                status = "TARGET"
            else:
                color = (255, 165, 0)
                status = match_result.name.upper()
        else:
            color = (0, 0, 255)
            status = "UNKNOWN"

        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)

        if self.show_detailed_info:
            self._draw_clean_text(vis, status, (face.x1, face.y1 - 10), color, 0.5, 1)

            if self.show_confidence:
                conf_text = f"{match_result.similarity:.2f}"
                self._draw_clean_text(
                    vis, conf_text, (face.x2 - 50, face.y1 - 10), color, 0.4, 1
                )

        if self.show_landmarks and match_result.accepted:
            for x, y in face.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 1, color, -1)

        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(
            face.kps, time.time()
        )
        if smile_detected:
            smile_x, smile_y = face.x2 - 25, face.y1 + 15
            cv2.circle(vis, (smile_x, smile_y), 8, (0, 255, 255), 1)
            cv2.circle(vis, (smile_x - 3, smile_y - 2), 1, (0, 255, 255), -1)
            cv2.circle(vis, (smile_x + 3, smile_y - 2), 1, (0, 255, 255), -1)
            cv2.ellipse(
                vis, (smile_x, smile_y + 2), (4, 2), 0, 0, 180, (0, 255, 255), 1
            )

    def _search_for_target(
        self, frame: np.ndarray, faces: List[FaceDet], vis: np.ndarray
    ):
        """Search for target identity"""
        frame_shape = frame.shape

        for face in faces:
            match_result = self._get_cached_identity(face)

            if match_result is None:
                aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
                emb = self.embedder.embed(aligned)
                match_result = self.matcher.match(emb)
                self._update_cache(face, match_result)

            is_target = (
                match_result.accepted and match_result.name == self.target_identity
            )

            if is_target:
                color = (0, 255, 0)
                status = "TARGET"
            elif match_result.accepted:
                color = (255, 165, 0)
                status = match_result.name.upper()
            else:
                color = (0, 0, 255)
                status = "UNKNOWN"

            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)

            if self.show_detailed_info:
                self._draw_clean_text(
                    vis, status, (face.x1, face.y1 - 10), color, 0.5, 1
                )

                if self.show_confidence:
                    self._draw_clean_text(
                        vis,
                        f"{match_result.similarity:.2f}",
                        (face.x2 - 50, face.y1 - 10),
                        color,
                        0.4,
                        1,
                    )

            if self.show_landmarks and match_result.accepted:
                for x, y in face.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 1, color, -1)

            smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(
                face.kps, time.time()
            )
            if smile_detected:
                smile_x, smile_y = face.x2 - 20, face.y1 + 15
                cv2.circle(vis, (smile_x, smile_y), 6, (0, 255, 255), 1)
                cv2.circle(vis, (smile_x - 2, smile_y - 1), 1, (0, 255, 255), -1)
                cv2.circle(vis, (smile_x + 2, smile_y - 1), 1, (0, 255, 255), -1)
                cv2.ellipse(
                    vis, (smile_x, smile_y + 1), (3, 2), 0, 0, 180, (0, 255, 255), 1
                )

            if is_target and match_result.similarity > 0.5:
                self._initiate_lock(face, match_result.similarity, frame)

    def _initiate_lock(self, face: FaceDet, confidence: float, frame: np.ndarray):
        """Initiate face lock on target and immediately center servo with auto-stop"""
        center = self._get_face_center(face)
        frame_w = frame.shape[1]

        # Stop scanning and seed current angle from scan position
        self._scanning = False
        self._current_servo_angle = self._scan_angle
        self._lock_centering_complete = False  # Reset centering status

        # Compute the centering angle RIGHT NOW and force-publish it
        center_angle = self._face_position_to_servo_angle(center[0], frame_w)
        self._publish_movement("CENTERING", confidence, angle=center_angle, force=True)
        print(f"🎯 CENTERING servo to {center_angle}° for {self.target_identity}")

        self.tracker = FaceTracker(
            identity=self.target_identity,
            last_position=center,
            last_seen_frame=self.frame_count,
            confidence_history=deque([confidence], maxlen=10),
            position_history=deque([center], maxlen=5),
            blink_state="unknown",
            blink_counter=0,
            smile_state=False,
            lock_start_time=time.time(),
        )

        self.is_locked = True
        self._record_action(
            "lock_initiated", f"Locked onto {self.target_identity} at position {center}"
        )
        self._publish_status("LOCKED", f"Locked onto {self.target_identity}")
        print(f"🔒 LOCKED onto {self.target_identity} - Auto-centering enabled")

    def _release_lock(self):
        """Release face lock and reset auto-stop status"""
        if self.tracker:
            lock_duration = time.time() - self.tracker.lock_start_time
            self._record_action(
                "lock_released", f"Lock held for {lock_duration:.1f} seconds"
            )
            self._publish_status(
                "UNLOCKED", f"Lock released after {lock_duration:.1f}s"
            )

        self.tracker = None
        self.is_locked = False
        self._lock_centering_complete = False  # Reset auto-stop status
        print("🔓 Lock RELEASED - Auto-stop disabled")

    def _detect_actions(self, face: FaceDet):
        """Detect actions on locked face"""
        if not self.tracker:
            return

        current_time = time.time()
        current_center = self._get_face_center(face)

        # Movement detection (already handled separately for MQTT)
        movement_info = self.action_detector.detect_movement_advanced(
            current_center, current_time
        )
        if movement_info and self.action_classifier.should_record_action(
            movement_info["direction"], current_time
        ):
            if self.mirror_mode:
                direction = movement_info["direction"]
                if "left" in direction:
                    direction = direction.replace("left", "right")
                elif "right" in direction:
                    direction = direction.replace("right", "left")
                movement_info["direction"] = direction

            action_type = self.action_classifier.classify_movement(movement_info)
            description = f"Face moved {movement_info['distance']:.1f}px in {movement_info['direction'].split('_')[2]} direction"
            self._record_action(action_type, description, movement_info["speed"])

        # Blink detection
        blink_detected, eye_metrics = self.action_detector.detect_blink_advanced(
            face.kps, current_time
        )
        if blink_detected and self.action_classifier.should_record_action(
            "eye_blink", current_time
        ):
            self.tracker.blink_counter += 1
            description = f"Blink detected (EAR: {eye_metrics.avg_eye_ratio:.3f}, count: {self.tracker.blink_counter})"
            self._record_action("eye_blink", description, eye_metrics.avg_eye_ratio)

        # Smile detection
        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(
            face.kps, current_time
        )
        if smile_detected != self.tracker.smile_state:
            if smile_detected and self.action_classifier.should_record_action(
                "smile", current_time
            ):
                description = (
                    f"Smile detected (mouth ratio: {mouth_metrics.mouth_ratio:.2f})"
                )
                self._record_action("smile", description, mouth_metrics.mouth_ratio)
            elif not smile_detected and self.action_classifier.should_record_action(
                "smile_end", current_time
            ):
                self._record_action("smile_end", "Smile ended")
            self.tracker.smile_state = smile_detected

    def _draw_locked_face(self, vis: np.ndarray, face: FaceDet):
        """Draw locked face with highlighting and centering indicator"""
        # Get current position status for visual feedback
        current_center = self._get_face_center(face)
        position_status = self._get_face_position_status(current_center, vis.shape)
        
        # Choose color based on centering status
        if position_status == "PERFECTLY_CENTERED":
            box_color = (0, 255, 0)  # Green for perfectly centered
            status_text = "🎯 LOCKED & CENTERED"
        else:
            box_color = (0, 165, 255)  # Orange for locked but not centered
            status_text = f"🔒 LOCKED: {self.target_identity}"
        
        cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), box_color, 3)

        lock_x, lock_y = face.x1 + 5, face.y1 + 5
        cv2.rectangle(vis, (lock_x, lock_y), (lock_x + 15, lock_y + 12), box_color, 2)
        cv2.circle(vis, (lock_x + 7, lock_y + 4), 4, box_color, 2)

        self._draw_clean_text(
            vis,
            status_text,
            (face.x1, face.y1 - 15),
            box_color,
            0.7,
            2,
        )
        
        # Draw centering indicator
        if position_status == "PERFECTLY_CENTERED":
            # Draw crosshair at face center when perfectly centered
            center_x, center_y = current_center
            cv2.drawMarker(vis, (center_x, center_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(vis, (center_x, center_y), 25, (0, 255, 0), 2)
            
            # Add "CENTERED" text below face
            self._draw_clean_text(
                vis,
                "✓ CENTERED",
                (face.x1, face.y2 + 35),
                (0, 255, 0),
                0.6,
                2,
            )

        if self.show_confidence and self.tracker and self.tracker.confidence_history:
            avg_conf = np.mean(list(self.tracker.confidence_history))
            self._draw_clean_text(
                vis,
                f"{avg_conf:.2f}",
                (face.x2 - 50, face.y1 - 15),
                (0, 255, 0),
                0.5,
                1,
            )

        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(
            face.kps, time.time()
        )

        if smile_detected:
            smile_x, smile_y = face.x1 + 30, face.y1 + 25
            cv2.circle(vis, (smile_x, smile_y), 12, (0, 255, 255), 2)
            cv2.circle(vis, (smile_x - 4, smile_y - 3), 2, (0, 255, 255), -1)
            cv2.circle(vis, (smile_x + 4, smile_y - 3), 2, (0, 255, 255), -1)
            cv2.ellipse(
                vis, (smile_x, smile_y + 2), (6, 4), 0, 0, 180, (0, 255, 255), 2
            )

            if self.show_detailed_info:
                self._draw_clean_text(
                    vis,
                    f"SMILE {mouth_metrics.mouth_ratio:.1f}",
                    (face.x1, face.y2 + 15),
                    (0, 255, 255),
                    0.5,
                    1,
                )
        elif self.show_detailed_info:
            self._draw_clean_text(
                vis,
                f"Score: {mouth_metrics.mouth_ratio:.1f}",
                (face.x1, face.y2 + 15),
                (150, 150, 150),
                0.4,
                1,
            )

        if self.show_detailed_info and self.tracker:
            self._draw_clean_text(
                vis,
                f"Blinks: {self.tracker.blink_counter}",
                (face.x2 - 80, face.y2 + 15),
                (0, 255, 0),
                0.4,
                1,
            )

        if self.show_landmarks:
            if len(face.kps) >= 5:
                left_mouth = face.kps[3].astype(int)
                right_mouth = face.kps[4].astype(int)
                cv2.circle(vis, tuple(left_mouth), 3, (0, 255, 255), -1)
                cv2.circle(vis, tuple(right_mouth), 3, (0, 255, 255), -1)
                cv2.line(vis, tuple(left_mouth), tuple(right_mouth), (0, 255, 255), 1)

                for i, (x, y) in enumerate(face.kps.astype(int)):
                    if i not in [3, 4]:
                        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

    def _draw_clean_ui(self, vis: np.ndarray):
        """Draw clean user interface with MQTT status"""
        h, w = vis.shape[:2]

        overlay = vis.copy()
        status_bg = np.zeros((50, w, 3), dtype=np.uint8)
        status_bg[:] = (0, 0, 0)

        status = "🔒 LOCKED" if self.is_locked else "🔍 SEARCHING"
        status_color = (0, 255, 0) if self.is_locked else (0, 255, 255)

        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

        mirror_indicator = "🪞" if self.mirror_mode else "📷"
        mqtt_indicator = "📡" if self.mqtt_enabled else "🚫"

        self._draw_clean_text(
            vis,
            f"Face Locking: {status} | Target: {self.target_identity} {mirror_indicator} {mqtt_indicator}",
            (10, 25),
            status_color,
            0.7,
            2,
        )

        if len(self.action_history) > 0:
            self._draw_clean_text(
                vis,
                f"Actions: {len(self.action_history)}",
                (w - 150, 25),
                (255, 255, 255),
                0.5,
                1,
            )

        # Show current face position if locked
        if self.is_locked and self.tracker and self.tracker.position_history:
            last_pos = self.tracker.position_history[-1]
            pos_status = self._get_face_position_status(last_pos, (h, w))
            pos_colors = {
                "MOVE_LEFT": (0, 0, 255),
                "CENTERED": (255, 255, 0),
                "MOVE_RIGHT": (255, 0, 0),
                "PERFECTLY_CENTERED": (0, 255, 0),
            }
            pos_color = pos_colors.get(pos_status, (255, 255, 255))
            
            # Show position status with special indicator for perfect centering
            display_text = "Position: " + ("🎯 CENTERED" if pos_status == "PERFECTLY_CENTERED" else pos_status)
            self._draw_clean_text(
                vis,
                display_text,
                (w - 300, 25),
                pos_color,
                0.5,
                1,
            )
            
            # Show current servo angle when locked
            if self.show_detailed_info:
                self._draw_clean_text(
                    vis,
                    f"Servo: {self._current_servo_angle:.1f}°",
                    (w - 300, 45),
                    (200, 200, 200),
                    0.4,
                    1,
                )

        if self.show_detailed_info:
            cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

            controls = "L:Lock | S:Save | R:Reload | +/-:Smile | F1/F2:Detection | m:Mirror | M:Landmarks | D:Details | P:MQTT | Q:Quit"
            self._draw_clean_text(vis, controls, (10, h - 50), (200, 200, 200), 0.45, 1)

            mqtt_status = "MQTT:ON" if self.mqtt_enabled else "MQTT:OFF"
            mirror_text = "Mirror:ON" if self.mirror_mode else "Mirror:OFF"
            settings = f"Smile: {self.action_detector.smile_ratio_threshold:.1f} | Scale: {self.window_scale:.1f}x | Face: {self.min_face_size[0]}px | {mirror_text} | {mqtt_status}"
            self._draw_clean_text(vis, settings, (10, h - 25), (150, 150, 150), 0.4, 1)
        else:
            cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            self._draw_clean_text(
                vis,
                "D:Show Details | P:Toggle MQTT | Q:Quit",
                (10, h - 10),
                (200, 200, 200),
                0.4,
                1,
            )

        if self.show_detailed_info and not self.is_locked:
            legend_x = 10
            legend_y = 70

            legend_items = [
                ("●", (0, 255, 0), "Locked"),
                ("●", (0, 255, 255), "Target"),
                ("●", (255, 165, 0), "Known"),
                ("●", (0, 0, 255), "Unknown"),
            ]

            for i, (symbol, color, label) in enumerate(legend_items):
                y_pos = legend_y + (i * 20)
                self._draw_clean_text(
                    vis, f"{symbol} {label}", (legend_x, y_pos), color, 0.4, 1
                )

    def toggle_ui_details(self):
        """Toggle detailed UI information"""
        self.show_detailed_info = not self.show_detailed_info
        print(f"UI Details: {'ON' if self.show_detailed_info else 'OFF'}")

    def toggle_landmarks(self):
        """Toggle landmark display"""
        self.show_landmarks = not self.show_landmarks
        print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")

    def toggle_confidence(self):
        """Toggle confidence display"""
        self.show_confidence = not self.show_confidence
        print(f"Confidence: {'ON' if self.show_confidence else 'OFF'}")

    def adjust_window_scale(self, increase: bool):
        """Adjust window scaling"""
        if increase:
            self.window_scale = min(2.0, self.window_scale + 0.1)
        else:
            self.window_scale = max(0.5, self.window_scale - 0.1)
        print(f"Window scale: {self.window_scale:.1f}x")

    def toggle_mirror_mode(self):
        """Toggle mirror mode for natural camera view"""
        self.mirror_mode = not self.mirror_mode
        print(f"Mirror mode: {'ON' if self.mirror_mode else 'OFF'}")
        self._record_action(
            "mirror_toggle",
            f"Mirror mode {'enabled' if self.mirror_mode else 'disabled'}",
        )

    def toggle_mqtt(self):
        """Toggle MQTT publishing on/off"""
        self.mqtt_enabled = not self.mqtt_enabled
        status = "ENABLED" if self.mqtt_enabled else "DISABLED"
        print(f"MQTT Publishing: {status}")
        self._record_action("mqtt_toggle", f"MQTT {status}")

        if self.mqtt_enabled:
            self._publish_status("ONLINE", "MQTT publishing enabled")
        else:
            try:
                self.mqtt_client.publish(
                    self.mqtt_config.topic_status,
                    json.dumps({"status": "OFFLINE", "message": "MQTT disabled"}),
                )
            except Exception:
                pass

    def adjust_face_detection_sensitivity(self, increase: bool):
        """Adjust face detection sensitivity"""
        if increase:
            self.min_face_size = (
                max(40, self.min_face_size[0] - 10),
                max(40, self.min_face_size[1] - 10),
            )
            self.face_aspect_ratio_range = (
                max(0.4, self.face_aspect_ratio_range[0] - 0.1),
                min(2.5, self.face_aspect_ratio_range[1] + 0.1),
            )
        else:
            self.min_face_size = (
                min(200, self.min_face_size[0] + 10),
                min(200, self.min_face_size[1] + 10),
            )
            self.face_aspect_ratio_range = (
                min(0.8, self.face_aspect_ratio_range[0] + 0.1),
                max(1.3, self.face_aspect_ratio_range[1] - 0.1),
            )

        print(
            f"Face detection - Min size: {self.min_face_size}, Aspect ratio: {self.face_aspect_ratio_range}"
        )
        self._record_action("detection_adjust", "Face detection sensitivity adjusted")

    def adjust_smile_threshold(self, increase: bool):
        """Adjust smile detection threshold"""
        if increase:
            self.action_detector.smile_ratio_threshold += 0.1
        else:
            self.action_detector.smile_ratio_threshold = max(
                0.5, self.action_detector.smile_ratio_threshold - 0.1
            )

        print(
            f"Smile threshold adjusted to: {self.action_detector.smile_ratio_threshold:.1f}"
        )
        self._record_action(
            "threshold_adjust",
            f"Smile threshold set to {self.action_detector.smile_ratio_threshold:.1f}",
        )

    def toggle_lock(self):
        """Manually toggle lock state"""
        if self.is_locked:
            self._release_lock()
        else:
            print(f"Manual lock toggle - searching for {self.target_identity}")

    def reload_database(self):
        """Reload face database"""
        db = load_db_npz(self.db_path)
        self.matcher.reload_from(self.db_path)
        self.identity_cache = {}  # Clear cache on reload
        print(f"Database reloaded: {len(db)} identities")
        self._publish_status("RELOADED", f"Database reloaded: {len(db)} identities")


def main():
    """Main face locking demo with MQTT control"""

    # Configure MQTT for local broker
    mqtt_config = MQTTConfig(
        broker="localhost",  # or "127.0.0.1"
        port=1883,
        topic_movement="vision/team351/movement",
        topic_status="robot/status",
        publish_interval=0.2,  # Reduced frequency from 0.1 to 0.2 seconds
    )

    # Initialize system
    try:
        system = FaceLockingSystem(
            target_identity="Wilson",
            window_scale=1.0,
            mirror_mode=True,
            mqtt_config=mqtt_config,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Open camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n" + "=" * 60)
    print("FACE LOCKING SYSTEM with MQTT Robot Control")
    print("=" * 60)
    print(f"Target Identity: {system.target_identity}")
    print(f"Window Scale: {system.window_scale}x")
    print(f"Mirror Mode: {'ON' if system.mirror_mode else 'OFF'}")
    print(f"MQTT Broker: {system.mqtt_config.broker}:{system.mqtt_config.port}")
    print(f"MQTT Movement Topic: {system.mqtt_config.topic_movement}")
    print(f"MQTT Status Topic: {system.mqtt_config.topic_status}")
    print(f"Face Detection: Min size {system.min_face_size}")
    print("\nControls:")
    print("  L - Toggle lock on/off")
    print("  S - Save action history to ./history/")
    print("  R - Reload database")
    print("  +/- - Adjust smile threshold")
    print("  F1/F2 - Adjust face detection sensitivity")
    print("  m - Toggle mirror mode")
    print("  M - Toggle landmarks display")
    print("  C - Toggle confidence display")
    print("  D - Toggle detailed UI")
    print("  P - Toggle MQTT publishing")
    print("  [ / ] - Adjust window scale")
    print("  Q - Quit")
    print("=" * 60)

    cv2.namedWindow("Face Locking System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Locking System", 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vis = system.process_frame(frame)
            cv2.imshow("Face Locking System", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("l") or key == ord("L"):
                system.toggle_lock()
            elif key == ord("s") or key == ord("S"):
                filepath = system._save_action_history()
                if filepath:
                    print(f"History saved to: {filepath}")
            elif key == ord("r") or key == ord("R"):
                system.reload_database()
            elif key == ord("+") or key == ord("="):
                system.adjust_smile_threshold(increase=True)
            elif key == ord("-") or key == ord("_"):
                system.adjust_smile_threshold(increase=False)
            elif key == ord("d") or key == ord("D"):
                system.toggle_ui_details()
            elif key == ord("["):
                system.adjust_window_scale(increase=False)
            elif key == ord("]"):
                system.adjust_window_scale(increase=True)
            elif key == ord("c") or key == ord("C"):
                system.toggle_confidence()
            elif key == ord("m"):
                system.toggle_mirror_mode()
            elif key == ord("M"):
                system.toggle_landmarks()
            elif key == ord("p") or key == ord("P"):
                system.toggle_mqtt()
            elif key == 65470:  # F1
                system.adjust_face_detection_sensitivity(increase=False)
            elif key == 65471:  # F2
                system.adjust_face_detection_sensitivity(increase=True)

    finally:
        if system.action_history:
            system._save_action_history()

        if system.mqtt_enabled:
            system._publish_status("OFFLINE", "System shutting down")
            time.sleep(0.1)
            system.mqtt_client.loop_stop()
            system.mqtt_client.disconnect()

        cap.release()
        cv2.destroyAllWindows()
        print("\nFace Locking System terminated")
        print(f"Total actions recorded: {len(system.action_history)}")


if __name__ == "__main__":
    main()
