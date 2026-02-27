#!/usr/bin/env python3
"""
Face Locking System - Consolidated Main Application

This is a consolidated version of the refactored face locking system
that combines all modules into a single file for easy reading and deployment,
while maintaining the modular architecture internally.

Features:
- Face detection and recognition using Haar + FaceMesh + ArcFace
- MQTT servo control with P-controller and auto-scan
- Action detection (movement, blinks, smiles)
- Real-time face locking and tracking
- UI visualization and overlays
- Action history management

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
from collections import deque

import cv2
import numpy as np
import paho.mqtt.client as mqtt

# Import existing modules
from src.recognize import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    FaceDet,
    MatchResult,
    load_db_npz,
    _kps_span_ok,
)
from src.haar_5pt import align_face_5pt
from src.action_detection import AdvancedActionDetector, ActionClassifier


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ServoConfig:
    """Configuration for servo control system"""
    broker: str = "localhost"
    port: int = 1883
    topic_movement: str = "vision/team351/movement"
    topic_status: str = "robot/status"
    topic_servo_status: str = "servo/status"
    client_id: str = f"servo_controller_{int(time.time())}"
    publish_interval: float = 0.1
    confidence_threshold: float = 0.7
    
    # Servo angle range
    servo_min_angle: int = 0
    servo_max_angle: int = 180
    servo_center_angle: int = 90
    
    # P-controller settings
    servo_Kp: float = 60.0
    servo_ema_alpha: float = 0.45
    
    # Auto-scan settings
    scan_enabled: bool = True
    scan_speed: float = 30.0
    scan_min_angle: float = 20.0
    scan_max_angle: float = 160.0
    
    # Auto-stop on lock
    auto_stop_on_lock: bool = True
    centering_tolerance: float = 0.05
    centering_stability_threshold: int = 5


@dataclass
class TrackingConfig:
    """Configuration for face tracking system"""
    target_identity: str = "Wilson"
    db_path: str = "data/db/face_db.npz"
    model_path: str = "models/embedder_arcface.onnx"
    window_scale: float = 1.0
    mirror_mode: bool = True
    
    # Face detection settings
    min_face_size: Tuple[int, int] = (70, 70)
    max_face_size: Tuple[int, int] = (800, 800)
    face_aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    
    # Tracking settings
    lock_timeout: int = 30
    recognition_interval: int = 10
    lock_on_first_detection: bool = False
    cache_distance_threshold: int = 80


@dataclass
class ActionRecord:
    """Single action record with timestamp"""
    timestamp: str
    action_type: str
    description: str
    value: Optional[float] = None


@dataclass
class FaceTrackingState:
    """Current state of face tracking"""
    identity: str
    last_position: Tuple[int, int]
    last_seen_frame: int
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    position_history: deque = field(default_factory=lambda: deque(maxlen=5))
    blink_state: str = "unknown"
    blink_counter: int = 0
    smile_state: bool = False
    lock_start_time: float = 0.0
    is_locked: bool = False


# ============================================================================
# SERVO CONTROLLER
# ============================================================================

class ServoController:
    """MQTT-based servo controller with face tracking capabilities"""
    
    def __init__(self, config: ServoConfig):
        self.config = config
        self.enabled = True
        self.last_mqtt_publish = 0
        self.last_mqtt_status = None
        
        # Servo angle tracking
        self._current_servo_angle: float = float(config.servo_center_angle)
        self._last_published_angle: int = -1
        self._target_servo_angle: Optional[float] = None
        self._centered_stable_frames: int = 0
        self._lock_centering_complete: bool = False
        
        # Scan-sweep state
        self._scan_angle: float = float(config.servo_center_angle)
        self._scan_direction: float = 1.0
        self._scan_last_time: float = time.time()
        self._scanning: bool = False
        self._scan_speed: float = config.scan_speed
        self._scan_min_angle: float = config.scan_min_angle
        self._scan_max_angle: float = config.scan_max_angle
        
        # Initialize MQTT
        self._init_mqtt()
        
        # Initialize action detector for advanced movement tracking
        self.action_detector = AdvancedActionDetector()
        
        print(f"Servo Controller initialized")
        print(f"MQTT Broker: {config.broker}:{config.port}")
        print(f"Movement Topic: {config.topic_movement}")
    
    def _init_mqtt(self):
        """Initialize MQTT connection"""
        self.mqtt_client = mqtt.Client(client_id=self.config.client_id)
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"✅ Connected to MQTT broker at {self.config.broker}")
                self._publish_status("ONLINE", "Servo controller started")
                client.subscribe(self.config.topic_servo_status)
            else:
                print(f"❌ Failed to connect to MQTT broker: {rc}")
        
        def on_message(client, userdata, msg):
            try:
                payload = msg.payload.decode().strip()
                if msg.topic == self.config.topic_servo_status:
                    if payload.startswith("ANGLE:") or payload.startswith("POSITION:"):
                        angle = float(payload.split(":")[1])
                        self._current_servo_angle = angle
            except Exception as e:
                print(f"MQTT message error: {e}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        
        try:
            self.mqtt_client.connect(self.config.broker, self.config.port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"❌ Failed to connect to MQTT broker: {e}")
            self.enabled = False
    
    def _publish_status(self, status: str, message: str):
        """Publish status message"""
        if not self.enabled:
            return
        try:
            payload = f"{status}: {message}"
            self.mqtt_client.publish(self.config.topic_status, payload)
        except Exception as e:
            print(f"Failed to publish status: {e}")
    
    def _publish_angle(self, angle: int):
        """Publish servo angle command"""
        if not self.enabled:
            return
        try:
            if abs(angle - self._last_published_angle) >= 1:
                payload = str(angle)
                self.mqtt_client.publish(self.config.topic_movement, payload)
                self._last_published_angle = angle
                self.last_mqtt_publish = time.time()
        except Exception as e:
            print(f"Failed to publish angle: {e}")
    
    def enable(self):
        """Enable servo control"""
        self.enabled = True
        self._publish_status("ENABLED", "Servo control enabled")
    
    def disable(self):
        """Disable servo control"""
        self.enabled = False
        self._publish_status("DISABLED", "Servo control disabled")
    
    def center_servo(self):
        """Move servo to center position"""
        center_angle = self.config.servo_center_angle
        self._target_servo_angle = float(center_angle)
        self._publish_angle(center_angle)
        self._scanning = False
    
    def update_face_tracking(self, face_center: Tuple[int, int], frame_size: Tuple[int, int], 
                           is_locked: bool, confidence: float) -> Optional[int]:
        """Update servo position based on face tracking using advanced movement detection"""
        if not self.enabled or confidence < self.config.confidence_threshold:
            return None
        
        frame_width, frame_height = frame_size
        face_x, face_y = face_center
        
        # Use advanced movement detection for smoother control
        movement_info = self.action_detector.detect_movement_advanced(face_center, time.time())
        
        if movement_info:
            # Extract movement metrics
            velocity_x = movement_info['velocity']
            speed = movement_info['speed']
            direction = movement_info['direction']
            displacement = movement_info['displacement']
            
            # Enhanced control using velocity and speed
            # Normalize velocity to servo control range
            max_velocity = 200.0  # pixels per second
            normalized_velocity = np.clip(velocity_x / max_velocity, -1.0, 1.0)
            
            # Apply speed factor for more responsive control
            speed_factor = min(2.0, 1.0 + speed / 100.0)
            control_signal = self.config.servo_Kp * normalized_velocity * speed_factor
            
            # Auto-stop on lock: center immediately on first lock
            if self.config.auto_stop_on_lock and is_locked and not self._lock_centering_complete:
                self.center_servo()
                self._lock_centering_complete = True
                return self.config.servo_center_angle
            
            # Check if face is centered using displacement
            if abs(displacement[0]) <= self.config.centering_tolerance * frame_width:
                self._centered_stable_frames += 1
                if self._centered_stable_frames >= self.config.centering_stability_threshold:
                    self._lock_centering_complete = True
            else:
                self._centered_stable_frames = 0
                self._lock_centering_complete = False
            
        else:
            # Fallback to basic position tracking if no movement detected
            normalized_x = (face_x - frame_width / 2) / (frame_width / 2)
            error = -normalized_x
            control_signal = self.config.servo_Kp * error
            
            # Auto-stop on lock
            if self.config.auto_stop_on_lock and is_locked and not self._lock_centering_complete:
                self.center_servo()
                self._lock_centering_complete = True
                return self.config.servo_center_angle
            
            # Check if face is centered
            if abs(normalized_x) <= self.config.centering_tolerance:
                self._centered_stable_frames += 1
                if self._centered_stable_frames >= self.config.centering_stability_threshold:
                    self._lock_centering_complete = True
            else:
                self._centered_stable_frames = 0
                self._lock_centering_complete = False
        
        # Apply EMA smoothing
        if self._target_servo_angle is not None:
            smoothed_signal = (self.config.servo_ema_alpha * control_signal + 
                             (1 - self.config.servo_ema_alpha) * 
                             (self._target_servo_angle - self._current_servo_angle))
        else:
            smoothed_signal = control_signal
        
        # Calculate target angle
        target_angle = self._current_servo_angle + smoothed_signal
        
        # Clamp to servo limits
        target_angle = max(self.config.servo_min_angle, 
                          min(self.config.servo_max_angle, target_angle))
        
        self._target_servo_angle = target_angle
        self._scanning = False
        
        # Publish angle command
        angle_int = int(round(target_angle))
        self._publish_angle(angle_int)
        
        return angle_int
    
    def update_scan_mode(self, current_time: float) -> Optional[int]:
        """Update servo position in scan mode"""
        if not self.enabled or not self.config.scan_enabled:
            return None
        
        if not self._scanning:
            self._scanning = True
            self._scan_angle = self._current_servo_angle
            self._scan_last_time = current_time
        
        dt = current_time - self._scan_last_time
        if dt < 0.001:
            return int(round(self._scan_angle))
        
        angle_change = self._scan_speed * dt * self._scan_direction
        self._scan_angle += angle_change
        
        # Check bounds and reverse direction
        if self._scan_angle >= self._scan_max_angle:
            self._scan_angle = self._scan_max_angle
            self._scan_direction = -1.0
        elif self._scan_angle <= self._scan_min_angle:
            self._scan_angle = self._scan_min_angle
            self._scan_direction = 1.0
        
        self._scan_last_time = current_time
        self._target_servo_angle = self._scan_angle
        
        angle_int = int(round(self._scan_angle))
        self._publish_angle(angle_int)
        return angle_int
    
    def process_tracking_update(self, tracking_result: dict, frame_size: Tuple[int, int]) -> Optional[int]:
        """Process tracking update and update servo accordingly"""
        current_time = tracking_result.get('frame_time', time.time())
        is_locked = tracking_result.get('is_locked', False)
        
        if is_locked and tracking_result.get('target_face') and tracking_result.get('match_result'):
            face = tracking_result['target_face']
            face_center = ((face.x1 + face.x2) // 2, (face.y1 + face.y2) // 2)
            confidence = tracking_result['match_result'].similarity
            return self.update_face_tracking(face_center, frame_size, is_locked, confidence)
        else:
            return self.update_scan_mode(current_time)
    
    def get_current_angle(self) -> float:
        return self._current_servo_angle
    
    def is_scanning(self) -> bool:
        return self._scanning
    
    def reset_centering(self):
        self._lock_centering_complete = False
        self._centered_stable_frames = 0
    
    def shutdown(self):
        self._publish_status("OFFLINE", "Servo controller shutting down")
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


# ============================================================================
# FACE TRACKER
# ============================================================================

class FaceTracker:
    """Base face tracking system with callback architecture"""
    
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.frame_count = 0
        
        # Initialize core components
        self.detector = HaarFaceMesh5pt(min_size=config.min_face_size, debug=False)
        self.embedder = ArcFaceEmbedderONNX(model_path=config.model_path, debug=False)
        
        # Load database and matcher
        db = load_db_npz(Path(config.db_path))
        self.matcher = FaceDBMatcher(db=db, dist_thresh=0.4)
        
        # Verify target identity exists
        if config.target_identity not in db:
            available = list(db.keys())
            raise ValueError(
                f"Target identity '{config.target_identity}' not found in database. "
                f"Available: {available}"
            )
        
        # Tracking state
        self.tracking_state: Optional[FaceTrackingState] = None
        self.is_locked = False
        
        # Recognition optimization
        self.identity_cache = {}
        self.cache_distance_threshold = config.cache_distance_threshold
        
        # Action detection
        self.action_detector = AdvancedActionDetector()
        self.action_classifier = ActionClassifier()
        self.action_history: List[ActionRecord] = []
        
        # Callback system
        self.callbacks: Dict[str, List[Callable]] = {
            'on_face_detected': [],
            'on_face_lost': [],
            'on_action_detected': [],
            'on_lock_acquired': [],
            'on_lock_lost': [],
            'on_frame_processed': []
        }
        
        print(f"Face Tracker initialized for: {config.target_identity}")
        print(f"Database contains {len(db)} identities: {list(db.keys())}")
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for specific events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """Trigger all callbacks for an event"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error for {event}: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDet]:
        """Detect all faces in frame"""
        return self.detector.detect(frame, max_faces=5)
    
    def recognize_face(self, frame: np.ndarray, face_det: FaceDet) -> MatchResult:
        """Recognize a specific face with caching optimization"""
        center = ((face_det.x1 + face_det.x2) // 2, (face_det.y1 + face_det.y2) // 2)
        
        mr = None
        needs_recognition = self.frame_count % self.config.recognition_interval == 0
        
        # Clean old cache entries
        self.identity_cache = {
            pos: val
            for pos, val in self.identity_cache.items()
            if self.frame_count - val[1] < 5
        }
        
        if not needs_recognition:
            for pos, (cached_mr, last_fidx) in self.identity_cache.items():
                dist = np.sqrt((center[0] - pos[0])**2 + (center[1] - pos[1])**2)
                if dist < self.cache_distance_threshold:
                    mr = cached_mr
                    break
        
        if mr is None:
            aligned, _ = align_face_5pt(frame, face_det.kps, out_size=(112, 112))
            emb = self.embedder.embed(aligned)
            mr = self.matcher.match(emb)
            self.identity_cache[center] = (mr, self.frame_count)
        
        return mr
    
    def find_target_face(self, frame: np.ndarray, faces: List[FaceDet]) -> Optional[Tuple[FaceDet, MatchResult]]:
        """Find the target identity among detected faces"""
        for face in faces:
            mr = self.recognize_face(frame, face)
            if mr.accepted and mr.name == self.config.target_identity:
                return face, mr
        return None
    
    def update_tracking_state(self, face: FaceDet, mr: MatchResult):
        """Update the tracking state with new face data"""
        center = ((face.x1 + face.x2) // 2, (face.y1 + face.y2) // 2)
        current_time = time.time()
        
        if self.tracking_state is None:
            self.tracking_state = FaceTrackingState(
                identity=mr.name,
                last_position=center,
                last_seen_frame=self.frame_count,
                lock_start_time=current_time
            )
            self.is_locked = True
            self._trigger_callbacks('on_lock_acquired', self.tracking_state)
        else:
            self.tracking_state.last_position = center
            self.tracking_state.last_seen_frame = self.frame_count
            self.tracking_state.confidence_history.append(mr.similarity)
            self.tracking_state.position_history.append(center)
        
        self._trigger_callbacks('on_face_detected', face, mr, self.tracking_state)
    
    def detect_actions(self, face: FaceDet, frame_time: float) -> List[ActionRecord]:
        """Detect actions on the tracked face"""
        if self.tracking_state is None:
            return []
        
        actions = []
        
        # Movement detection
        movement_info = self.action_detector.detect_movement_advanced(
            self.tracking_state.last_position, frame_time
        )
        if movement_info:
            action_type = self.action_classifier.classify_movement(movement_info)
            if self.action_classifier.should_record_action(action_type, frame_time):
                description = self.action_classifier.get_action_description(action_type, movement_info)
                actions.append(ActionRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    action_type=action_type,
                    description=description,
                    value=movement_info.get('speed', 0)
                ))
        
        # Blink detection
        blink_detected, eye_metrics = self.action_detector.detect_blink_advanced(face.kps, frame_time)
        if blink_detected:
            action_type = "eye_blink"
            if self.action_classifier.should_record_action(action_type, frame_time):
                description = self.action_classifier.get_action_description(action_type, eye_metrics)
                actions.append(ActionRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    action_type=action_type,
                    description=description,
                    value=eye_metrics.avg_eye_ratio
                ))
        
        # Smile detection
        smile_detected, mouth_metrics = self.action_detector.detect_smile_advanced(face.kps, frame_time)
        if smile_detected and not self.tracking_state.smile_state:
            action_type = "smile"
            if self.action_classifier.should_record_action(action_type, frame_time):
                description = self.action_classifier.get_action_description(action_type, mouth_metrics)
                actions.append(ActionRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    action_type=action_type,
                    description=description,
                    value=mouth_metrics.mouth_ratio
                ))
            self.tracking_state.smile_state = True
        elif not smile_detected and self.tracking_state.smile_state:
            action_type = "smile_end"
            if self.action_classifier.should_record_action(action_type, frame_time):
                description = self.action_classifier.get_action_description(action_type)
                actions.append(ActionRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    action_type=action_type,
                    description=description
                ))
            self.tracking_state.smile_state = False
        
        for action in actions:
            self.action_history.append(action)
            self._trigger_callbacks('on_action_detected', action)
        
        return actions
    
    def check_lock_timeout(self) -> bool:
        """Check if lock should be released due to timeout"""
        if self.tracking_state is None:
            return False
        
        frames_since_seen = self.frame_count - self.tracking_state.last_seen_frame
        if frames_since_seen > self.config.lock_timeout:
            self.is_locked = False
            old_state = self.tracking_state
            self.tracking_state = None
            self._trigger_callbacks('on_lock_lost', old_state)
            self._trigger_callbacks('on_face_lost', old_state)
            return True
        
        return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and return tracking results"""
        self.frame_count += 1
        frame_time = time.time()
        
        faces = self.detect_faces(frame)
        target_result = None
        
        if faces:
            target_result = self.find_target_face(frame, faces)
        
        actions = []
        if target_result:
            face, mr = target_result
            self.update_tracking_state(face, mr)
            actions = self.detect_actions(face, frame_time)
        else:
            self.check_lock_timeout()
        
        result = {
            'faces': faces,
            'target_face': target_result[0] if target_result else None,
            'match_result': target_result[1] if target_result else None,
            'tracking_state': self.tracking_state,
            'actions': actions,
            'is_locked': self.is_locked,
            'frame_time': frame_time
        }
        
        self._trigger_callbacks('on_frame_processed', result)
        return result
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracking_state = None
        self.is_locked = False
        self.identity_cache.clear()
        self.action_detector.reset_state()
        self.action_classifier.last_actions.clear()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class FaceLockingApp:
    """Main face locking application"""
    
    def __init__(self, target_identity: str = "Wilson", db_path: str = "data/db/face_db.npz",
                 model_path: str = "models/embedder_arcface.onnx", window_scale: float = 1.0,
                 mirror_mode: bool = True, mqtt_config: Optional[ServoConfig] = None):
        
        # Initialize configs
        self.tracking_config = TrackingConfig(
            target_identity=target_identity, db_path=db_path, model_path=model_path,
            window_scale=window_scale, mirror_mode=mirror_mode
        )
        
        # Initialize components
        self.tracker = FaceTracker(self.tracking_config)
        self.servo = ServoController(mqtt_config or ServoConfig())
        
        # UI settings
        self.show_detailed_info = True
        self.show_landmarks = True
        self.show_confidence = True
        self.ui_alpha = 0.8
        
        # History management
        self.history_dir = Path("history")
        self.history_dir.mkdir(exist_ok=True)
        
        # Register callbacks
        self._setup_callbacks()
        
        print("Face Locking Application initialized")
        print(f"Target: {target_identity}")
        print(f"Window scale: {window_scale}x")
        print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
    
    def _setup_callbacks(self):
        """Setup callbacks between tracker and servo"""
        def on_lock_acquired(tracking_state):
            self.servo.reset_centering()
            print(f"🔒 Locked onto {tracking_state.identity}")
        
        def on_lock_lost(tracking_state):
            print(f"🔓 Lost track of {tracking_state.identity}")
        
        def on_action_detected(action: ActionRecord):
            print(f"🎯 {action.description}")
        
        self.tracker.register_callback('on_lock_acquired', on_lock_acquired)
        self.tracker.register_callback('on_lock_lost', on_lock_lost)
        self.tracker.register_callback('on_action_detected', on_action_detected)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return visualization"""
        # Store frame size for servo calculations
        self.frame_size = (frame.shape[1], frame.shape[0])
        
        # Mirror if enabled
        if self.tracking_config.mirror_mode:
            frame = cv2.flip(frame, 1)
        
        # Process frame through tracker
        result = self.tracker.process_frame(frame)
        
        # Create visualization
        vis_frame = self._create_visualization(frame, result)
        
        # Update servo with correct frame size
        self.servo.process_tracking_update(result, self.frame_size)
        
        return vis_frame
    
    def _create_visualization(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Create visualization with overlays"""
        vis = frame.copy()
        faces = result['faces']
        target_face = result.get('target_face')
        match_result = result.get('match_result')
        tracking_state = result.get('tracking_state')
        actions = result.get('actions', [])
        is_locked = result.get('is_locked', False)
        
        # Draw all detected faces
        for i, face in enumerate(faces):
            if face == target_face and match_result and match_result.accepted:
                if match_result.name == self.tracking_config.target_identity:
                    color = (0, 255, 0)  # Green for locked target
                    status = "LOCKED"
                else:
                    color = (0, 255, 255)  # Yellow for known but not target
                    status = match_result.name or "UNKNOWN"
            else:
                color = (0, 0, 255)  # Red for unknown
                status = "UNKNOWN"
            
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
            
            if self.show_landmarks:
                for x, y in face.kps.astype(int):
                    cv2.circle(vis, (x, y), 2, color, -1)
            
            label = f"{i+1}: {status}"
            cv2.putText(vis, label, (face.x1, face.y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if self.show_confidence and match_result and face == target_face:
                conf_text = f"conf: {match_result.similarity:.3f}"
                cv2.putText(vis, conf_text, (face.x1, face.y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw tracking info
        if is_locked and tracking_state:
            self._draw_tracking_info(vis, tracking_state)
        
        # Draw servo info
        self._draw_servo_info(vis)
        
        # Draw action indicators
        self._draw_action_indicators(vis, actions)
        
        # Draw UI header
        self._draw_ui_header(vis, result)
        
        return vis
    
    def _draw_tracking_info(self, vis: np.ndarray, tracking_state):
        """Draw tracking information overlay"""
        h, w = vis.shape[:2]
        
        info_lines = [
            f"Locked: {tracking_state.identity}",
            f"Position: ({tracking_state.last_position[0]}, {tracking_state.last_position[1]})",
            f"Frames seen: {tracking_state.last_seen_frame}",
            f"Lock time: {time.time() - tracking_state.lock_start_time:.1f}s"
        ]
        
        if self.show_detailed_info:
            info_lines.extend([
                f"Blink state: {tracking_state.blink_state}",
                f"Smiling: {tracking_state.smile_state}",
                f"Avg confidence: {np.mean(list(tracking_state.confidence_history)):.3f}" if tracking_state.confidence_history else "Avg confidence: N/A"
            ])
        
        overlay = vis.copy()
        box_height = len(info_lines) * 20 + 10
        cv2.rectangle(overlay, (10, 60), (250, 60 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.ui_alpha, vis, 1 - self.ui_alpha, 0, vis)
        
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (15, 75 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _draw_servo_info(self, vis: np.ndarray):
        """Draw servo status information"""
        h, w = vis.shape[:2]
        
        current_angle = self.servo.get_current_angle()
        is_scanning = self.servo.is_scanning()
        
        servo_lines = [
            f"Servo: {current_angle:.1f}°",
            f"Mode: {'SCAN' if is_scanning else 'TRACK'}",
        ]
        
        overlay = vis.copy()
        box_height = len(servo_lines) * 20 + 10
        cv2.rectangle(overlay, (w - 200, 60), (w - 10, 60 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.ui_alpha, vis, 1 - self.ui_alpha, 0, vis)
        
        for i, line in enumerate(servo_lines):
            cv2.putText(vis, line, (w - 195, 75 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _draw_action_indicators(self, vis: np.ndarray, actions):
        """Draw action detection indicators"""
        h, w = vis.shape[:2]
        
        if not actions:
            return
        
        recent_actions = actions[-3:]
        for i, action in enumerate(recent_actions):
            if "blink" in action.action_type:
                color = (255, 255, 0)  # Yellow
            elif "smile" in action.action_type:
                color = (0, 255, 255)  # Cyan
            elif "moved" in action.action_type:
                color = (255, 165, 0)  # Orange
            else:
                color = (255, 255, 255)  # White
            
            text = f"🎯 {action.action_type}"
            cv2.putText(vis, text, (10, h - 30 - i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_ui_header(self, vis: np.ndarray, result: Dict[str, Any]):
        """Draw main UI header with status"""
        h, w = vis.shape[:2]
        
        is_locked = result.get('is_locked', False)
        frame_count = self.tracker.frame_count
        action_count = len(self.tracker.action_history)
        
        header_text = (
            f"Face Locking | Target: {self.tracking_config.target_identity} | "
            f"Status: {'LOCKED' if is_locked else 'SEARCHING'} | "
            f"Frame: {frame_count} | Actions: {action_count}"
        )
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
        
        status_color = (0, 255, 0) if is_locked else (0, 0, 255)
        cv2.putText(vis, header_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    def save_action_history(self, filename: Optional[str] = None):
        """Save action history to timestamped file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"actions_{timestamp}.json"
        
        filepath = self.history_dir / filename
        
        history_data = []
        for action in self.tracker.action_history:
            action_dict = {
                'timestamp': action.timestamp,
                'action_type': action.action_type,
                'description': action.description,
                'value': action.value
            }
            history_data.append(action_dict)
        
        with open(filepath, 'w') as f:
            json.dump({
                'target_identity': self.tracking_config.target_identity,
                'session_start': datetime.now().isoformat(),
                'total_actions': len(history_data),
                'actions': history_data
            }, f, indent=2)
        
        print(f"Action history saved to {filepath}")
        return filepath
    
    def toggle_lock(self):
        """Toggle face locking on/off"""
        if self.is_locked:
            self.tracker.reset_tracking()
            print("🔓 Face locking disabled")
        else:
            print("🔒 Face locking enabled - waiting for target face...")
    
    @property
    def is_locked(self) -> bool:
        return self.tracker.is_locked
    
    def shutdown(self):
        """Shutdown the application"""
        print("Shutting down Face Locking Application...")
        
        if self.tracker.action_history:
            self.save_action_history("final_actions.json")
        
        self.servo.shutdown()
        print("Shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

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
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            processed_frame = app.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
            
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Face Locking System", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                try:
                    app.tracker.matcher.reload_from(Path(db_path))
                    app.tracker.identity_cache.clear()
                    print("📁 Database reloaded")
                except Exception as e:
                    print(f"❌ Failed to reload database: {e}")
            
            elif key == ord('l'):
                app.toggle_lock()
            
            elif key in (ord('+'), ord('=')):
                current_threshold = app.tracker.action_detector.smile_ratio_threshold
                new_threshold = min(2.0, current_threshold + 0.1)
                app.tracker.action_detector.smile_ratio_threshold = new_threshold
                print(f"😊 Smile threshold: {new_threshold:.2f}")
            
            elif key == ord('-'):
                current_threshold = app.tracker.action_detector.smile_ratio_threshold
                new_threshold = max(0.5, current_threshold - 0.1)
                app.tracker.action_detector.smile_ratio_threshold = new_threshold
                print(f"😊 Smile threshold: {new_threshold:.2f}")
            
            elif key == ord('m'):
                app.tracking_config.mirror_mode = not app.tracking_config.mirror_mode
                status = "ON" if app.tracking_config.mirror_mode else "OFF"
                print(f"🪞 Mirror mode: {status}")
            
            elif key == ord('M'):
                app.show_landmarks = not app.show_landmarks
                status = "ON" if app.show_landmarks else "OFF"
                print(f"📍 Landmarks: {status}")
            
            elif key == ord('C'):
                app.show_confidence = not app.show_confidence
                status = "ON" if app.show_confidence else "OFF"
                print(f"📊 Confidence: {status}")
            
            elif key == ord('d'):
                app.show_detailed_info = not app.show_detailed_info
                status = "ON" if app.show_detailed_info else "OFF"
                print(f"📋 Detailed info: {status}")
            
            elif key == ord('['):
                app.tracking_config.window_scale = max(0.5, app.tracking_config.window_scale - 0.1)
                print(f"🔍 Window scale: {app.tracking_config.window_scale:.1f}x")
            
            elif key == ord(']'):
                app.tracking_config.window_scale = min(2.0, app.tracking_config.window_scale + 0.1)
                print(f"🔍 Window scale: {app.tracking_config.window_scale:.1f}x")
            
            elif key == ord('s'):
                try:
                    filepath = app.save_action_history()
                    print(f"💾 Action history saved")
                except Exception as e:
                    print(f"❌ Failed to save history: {e}")
            
            elif key == ord('p'):
                if app.servo.enabled:
                    app.servo.disable()
                    print("📡 MQTT publishing: OFF")
                else:
                    app.servo.enable()
                    print("📡 MQTT publishing: ON")
            
            elif key == 0xFFBE:  # F1
                current_min = app.tracker.detector.min_size
                new_min = (max(50, current_min[0] - 10), max(50, current_min[1] - 10))
                app.tracker.detector.min_size = new_min
                print(f"🎯 Face sensitivity: LESS sensitive (min_size={new_min})")
            
            elif key == 0xFFBF:  # F2
                current_min = app.tracker.detector.min_size
                new_min = (max(30, current_min[0] - 10), max(30, current_min[1] - 10))
                app.tracker.detector.min_size = new_min
                print(f"🎯 Face sensitivity: MORE sensitive (min_size={new_min})")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        app.shutdown()
        print("Goodbye!")


if __name__ == "__main__":
    main()
