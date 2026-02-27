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
from src.face_detector import RobustFaceDetector, FaceDetectionResult
from src.action_detection import AdvancedActionDetector, ActionClassifier

# Import the working recognition system
from src.recognize import HaarFaceMesh5pt as WorkingFaceDetector


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
            velocity_x, velocity_y = movement_info['velocity']  # Extract x component
            speed = float(movement_info['speed'])  # Convert to float
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
        self.detector = RobustFaceDetector(
            min_size=config.min_face_size,
            confidence_threshold=0.7,
            quality_threshold=0.3,
            enable_mediapipe=True,
            debug=False
        )
        self.embedder = ArcFaceEmbedderONNX(model_path=config.model_path, debug=False)
        
        # Use the working recognition system from recognize.py
        self.working_detector = WorkingFaceDetector(
            min_size=config.min_face_size,
            debug=False
        )
        
        # Load database and matcher
        db = load_db_npz(Path(config.db_path))
        self.matcher = FaceDBMatcher(db=db, dist_thresh=0.4)
        
        # Check if target identity exists
        if self.config.target_identity is not None:
            if self.config.target_identity not in db:
                available_users = list(db.keys())
                raise ValueError(
                    f"Target identity '{self.config.target_identity}' not found in database. "
                    f"Available: {available_users}"
                )
        
        print("Face Tracker initialized")
        if self.config.target_identity:
            print(f"Target identity: {self.config.target_identity}")
        else:
            print("Demo mode - no target identity set")
        
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
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using robust detector"""
        return self.detector.detect_faces(frame, max_faces=10)  # Increased to 10 faces
    
    def recognize_face(self, frame: np.ndarray, face_det: FaceDetectionResult) -> MatchResult:
        """Recognize a specific face with caching optimization"""
        if not face_det.is_valid:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        
        center = ((face_det.x + face_det.x + face_det.width) // 2, 
                  (face_det.y + face_det.y + face_det.height) // 2)
        
        mr = None
        needs_recognition = self.frame_count % self.config.recognition_interval == 0
        
        # Clean old cache entries
        self.identity_cache = {
            pos: (cached_mr, last_fidx)
            for pos, (cached_mr, last_fidx) in self.identity_cache.items()
            if self.frame_count - last_fidx < 5
        }
        
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
    
    def recognize_all_faces(self, frame: np.ndarray, faces: List[FaceDetectionResult]) -> List[Tuple[FaceDetectionResult, MatchResult]]:
        """Recognize all detected faces using optimized recognition system"""
        results = []
        
        # Use the working detector from recognize.py with caching
        working_faces = self.working_detector.detect(frame, max_faces=10)
        
        # Process each face with optimized recognition
        for working_face in working_faces:
            center = ((working_face.x1 + working_face.x2) // 2, (working_face.y1 + working_face.y2) // 2)
            
            # Check cache first for performance
            mr = None
            needs_recognition = self.frame_count % 10 == 0  # Recognize every 10 frames
            
            if not needs_recognition:
                # Check cache for existing recognition
                for pos, (cached_mr, last_fidx) in self.identity_cache.items():
                    dist = np.sqrt((center[0] - pos[0])**2 + (center[1] - pos[1])**2)
                    if dist < 80:  # 80 pixel threshold
                        mr = cached_mr
                        break
            
            # Only do heavy recognition if needed
            if mr is None or needs_recognition:
                try:
                    aligned, _ = align_face_5pt(frame, working_face.kps, out_size=(112, 112))
                    emb = self.embedder.embed(aligned)
                    mr = self.matcher.match(emb)
                    self.identity_cache[center] = (mr, self.frame_count)
                except Exception as e:
                    print(f"Recognition error: {e}")
                    mr = MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
            
            # Convert FaceDet to FaceDetectionResult format
            face_result = FaceDetectionResult(
                x1=working_face.x1,
                y1=working_face.y1,
                x2=working_face.x2,
                y2=working_face.y2,
                score=working_face.score,
                kps=working_face.kps,
                confidence=working_face.score,
                quality_score=0.8,
                is_valid=True
            )
            
            results.append((face_result, mr))
        
        return results
    
    def find_target_face(self, frame: np.ndarray, faces: List[FaceDetectionResult]) -> Optional[Tuple[FaceDetectionResult, MatchResult]]:
        """Find target face among detected faces using optimized recognition system"""
        if self.config.target_identity is None:
            return None
        
        # Use the working detector from recognize.py
        working_faces = self.working_detector.detect(frame, max_faces=10)
        
        for working_face in working_faces:
            center = ((working_face.x1 + working_face.x2) // 2, (working_face.y1 + working_face.y2) // 2)
            
            # Check cache first for performance
            mr = None
            needs_recognition = self.frame_count % 5 == 0  # Recognize target every 5 frames (more frequent)
            
            if not needs_recognition:
                # Check cache for existing recognition
                for pos, (cached_mr, last_fidx) in self.identity_cache.items():
                    dist = np.sqrt((center[0] - pos[0])**2 + (center[1] - pos[1])**2)
                    if dist < 80:  # 80 pixel threshold
                        mr = cached_mr
                        break
            
            # Only do heavy recognition if needed
            if mr is None or needs_recognition:
                try:
                    aligned, _ = align_face_5pt(frame, working_face.kps, out_size=(112, 112))
                    emb = self.embedder.embed(aligned)
                    mr = self.matcher.match(emb)
                    self.identity_cache[center] = (mr, self.frame_count)
                except Exception as e:
                    print(f"Target recognition error: {e}")
                    continue
            
            # Check if this is our target
            if mr.accepted and mr.name == self.config.target_identity:
                # Convert to our format
                face_result = FaceDetectionResult(
                    x1=working_face.x1,
                    y1=working_face.y1,
                    x2=working_face.x2,
                    y2=working_face.y2,
                    score=working_face.score,
                    kps=working_face.kps,
                    confidence=working_face.score,
                    quality_score=0.9,  # High quality for target
                    is_valid=True
                )
                return (face_result, mr)
        
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
    
    def detect_actions(self, face: FaceDetectionResult, frame_time: float) -> List[ActionRecord]:
        """Detect actions for a face (smile and movement only, no blinking)"""
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
        
        # Smile detection (but no blinking)
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
        """Process a single frame and return tracking results (optimized)"""
        start_time = time.time()
        self.frame_count += 1
        frame_time = time.time()
        
        # Use the working detector from recognize.py for face detection (only once)
        working_faces = self.working_detector.detect(frame, max_faces=10)
        
        # Convert working faces to our format for visualization
        faces = []
        for working_face in working_faces:
            face_result = FaceDetectionResult(
                x1=working_face.x1,
                y1=working_face.y1,
                x2=working_face.x2,
                y2=working_face.y2,
                score=working_face.score,
                kps=working_face.kps,
                confidence=working_face.score,
                quality_score=0.8,
                is_valid=True
            )
            faces.append(face_result)
        
        # Find target face first (more frequent recognition)
        target_result = self.find_target_face(frame, faces)
        
        # Only recognize all faces if we need detailed info (less frequent)
        all_recognitions = []
        if self.frame_count % 15 == 0:  # Every 15 frames for full recognition
            all_recognitions = self.recognize_all_faces(frame, faces)
        elif target_result:
            # If we have target, just add that to recognitions
            all_recognitions = [target_result]
        
        actions = []
        if target_result:
            face, mr = target_result
            self.update_tracking_state(face, mr)
            actions = self.detect_actions(face, frame_time)
        else:
            self.check_lock_timeout()
        
        # Performance monitoring
        process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        result = {
            'faces': faces,
            'all_recognitions': all_recognitions,
            'target_face': target_result[0] if target_result else None,
            'match_result': target_result[1] if target_result else None,
            'tracking_state': self.tracking_state,
            'actions': actions,
            'is_locked': self.is_locked,
            'frame_time': frame_time,
            'frame_count': self.frame_count,
            'process_time_ms': process_time  # Add performance metric
        }
        
        # Show performance info every 60 frames
        if self.frame_count % 60 == 0:
            print(f"⚡ Frame processing: {process_time:.1f}ms | Faces: {len(faces)} | Cache: {len(self.identity_cache)}")
        
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
        try:
            # Store frame size for servo calculations
            self.frame_size = (frame.shape[1], frame.shape[0])
            
            # Mirror if enabled
            if self.tracking_config.mirror_mode:
                frame = cv2.flip(frame, 1)
            
            # Process frame through tracker
            result = self.tracker.process_frame(frame)
            
            # Create visualization
            vis_frame = self._create_visualization(frame, result)
            
            # Update servo with correct frame size (with error handling)
            try:
                self.servo.process_tracking_update(result, self.frame_size)
            except Exception as e:
                print(f"Servo update error: {e}")
                # Continue without servo update
            
            return vis_frame
        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback to see where error occurs
            # Return original frame if processing fails
            return frame
    
    def _create_visualization(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Create visualization with overlays"""
        vis = frame.copy()
        faces = result['faces']
        all_recognitions = result.get('all_recognitions', [])  # Get all face recognitions
        target_face = result.get('target_face')
        match_result = result.get('match_result')
        tracking_state = result.get('tracking_state')
        actions = result.get('actions', [])
        is_locked = result.get('is_locked', False)
        
        # Draw ALL detected and recognized faces
        for face_det in faces:
            # Find the recognition result for this face
            face_recognition = None
            for f, mr in all_recognitions:
                if (f.x1 == face_det.x1 and f.y1 == face_det.y1 and 
                    f.x2 == face_det.x2 and f.y2 == face_det.y2):
                    face_recognition = (f, mr)
                    break
            
            # Determine color and status
            if face_recognition:
                f, mr = face_recognition
                if mr.accepted:
                    if mr.name == self.tracking_config.target_identity:
                        color = (0, 255, 0)  # Green for target
                        status = f"TARGET: {mr.name}"
                    else:
                        color = (0, 255, 255)  # Yellow for known but not target
                        status = mr.name or "KNOWN"
                else:
                    color = (255, 0, 0)  # Red for unknown
                    status = "UNKNOWN"
            else:
                color = (255, 165, 0)  # Orange for detected but not recognized
                status = "DETECTED"
            
            # Draw bounding box
            cv2.rectangle(vis, (face_det.x1, face_det.y1), (face_det.x2, face_det.y2), color, 2)
            
            # Draw landmarks
            if face_det.kps is not None:
                for (x, y) in face_det.kps.astype(int):
                    cv2.circle(vis, (x, y), 2, (255, 255, 255), -1)
            
            # Draw label with quality score
            quality_text = f"Q:{face_det.quality_score:.2f}" if hasattr(face_det, 'quality_score') else ""
            label = f"{status} {quality_text}".strip()
            cv2.putText(vis, label, (face_det.x1, max(0, face_det.y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight target face if locked
        if is_locked and target_face and match_result and match_result.accepted:
            if match_result.name == self.tracking_config.target_identity:
                # Draw special indicator for locked target
                center_x = (target_face.x1 + target_face.x2) // 2
                center_y = (target_face.y1 + target_face.y2) // 2
                cv2.circle(vis, (center_x, center_y), 50, (0, 255, 0), 3)
                cv2.putText(vis, "🔒 LOCKED", (target_face.x1, target_face.y2 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.show_confidence:
                    conf_text = f"conf: {match_result.similarity:.3f}"
                    cv2.putText(vis, conf_text, (target_face.x1, target_face.y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
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
                'value': float(action.value) if action.value is not None else None
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
# ============================================================================

def main():
    """Main application entry point"""
    import sys
    
    # Load database to get available users
    db_path = "data/db/face_db.npz"
    available_users = []
    
    try:
        db = load_db_npz(Path(db_path))
        available_users = list(db.keys())
    except Exception as e:
        print(f"Warning: Could not load database: {e}")
        available_users = []
    
    # Handle user selection
    target_identity = None
    
    if not available_users:
        print("No users found in database!")
        print("Please enroll a user first:")
        print("  python -m src.enroll")
        print("\nOr continue with demo mode (no face locking)...")
        choice = input("Continue with demo mode? (y/n): ").lower().strip()
        if choice != 'y':
            sys.exit(1)
        target_identity = None
    else:
        print("\nAvailable users in database:")
        for i, user in enumerate(available_users, 1):
            print(f"  {i}. {user}")
        
        print(f"\nDefault user: {available_users[0]}")
        choice = input(f"Select user (1-{len(available_users)}) or press Enter for default: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(available_users):
            target_identity = available_users[int(choice) - 1]
        else:
            target_identity = available_users[0]
        
        print(f"Selected user: {target_identity}")
    
    # Parse command line arguments
    window_scale = 1.0
    mirror_mode = True
    mqtt_enabled = True
    
    if len(sys.argv) > 1:
        try:
            window_scale = float(sys.argv[1])
        except ValueError:
            pass
    
    if len(sys.argv) > 2:
        mirror_mode = sys.argv[2].lower() != 'false'
    
    if len(sys.argv) > 3:
        mqtt_enabled = sys.argv[3].lower() != 'false'
    
    # Initialize MQTT config
    mqtt_config = None
    if mqtt_enabled:
        try:
            mqtt_config = ServoConfig()
        except Exception as e:
            print(f"Warning: MQTT initialization failed: {e}")
            print("Continuing without servo control...")
            mqtt_config = None
    
    # Initialize application
    app = FaceLockingApp(
        target_identity=target_identity,
        window_scale=window_scale,
        mirror_mode=mirror_mode,
        mqtt_config=mqtt_config
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create resizable window
    window_name = 'Face Locking System'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(1280 * window_scale), int(720 * window_scale))
    
    # Enable window resizing (not fullscreen)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    
    # Main loop
    print("\n=== Face Locking System ===")
    print("Controls:")
    print("  Q - Quit")
    print("  R - Reload database")
    print("  L - Toggle lock on target")
    print("  D - Toggle detailed UI")
    print("  M - Toggle mirror mode")
    print("  C - Toggle confidence display")
    print("  +/- - Adjust smile threshold")
    print("  F1/F2 - Adjust face detection sensitivity")
    print("  [/] - Adjust window scale")
    print("  W/S - Increase/Decrease window size")
    print("  F - Toggle fullscreen")
    print("  A - Save action history")
    print("  P - Toggle MQTT publishing")
    
    if target_identity:
        print(f"\nTarget: {target_identity}")
        print("Press 'L' to lock onto target face")
    else:
        print("\nDemo mode - no target user selected")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = app.process_frame(frame)
            
            # Display
            cv2.imshow(window_name, processed_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):
                app.reload_database()
            elif key == ord('l'):
                app.toggle_lock()
            elif key == ord('d'):
                app.toggle_detailed_info()
            elif key == ord('m'):
                app.toggle_mirror_mode()
            elif key == ord('c'):
                app.toggle_confidence_display()
            elif key == ord('+') or key == ord('='):
                app.adjust_smile_threshold(0.05)
            elif key == ord('-') or key == ord('_'):
                app.adjust_smile_threshold(-0.05)
            elif key == ord('['):
                app.adjust_window_scale(0.8)
            elif key == ord(']'):
                app.adjust_window_scale(1.2)
            elif key == ord('w') or key == ord('W'):
                # Increase window size
                current_size = cv2.getWindowImageRect(window_name)
                new_width = int(current_size[2] * 1.1)
                new_height = int(current_size[3] * 1.1)
                cv2.resizeWindow(window_name, new_width, new_height)
                print(f"Window resized to: {new_width}x{new_height}")
            elif key == ord('s') or key == ord('S'):
                # Decrease window size
                current_size = cv2.getWindowImageRect(window_name)
                new_width = int(current_size[2] * 0.9)
                new_height = int(current_size[3] * 0.9)
                if new_width > 400 and new_height > 300:  # Minimum size
                    cv2.resizeWindow(window_name, new_width, new_height)
                    print(f"Window resized to: {new_width}x{new_height}")
            elif key == ord('f') or key == ord('F'):
                # Toggle fullscreen
                current_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("Fullscreen OFF")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Fullscreen ON")
            elif key == ord('a') or key == ord('A'):
                app.save_action_history()
            elif key == ord('p'):
                app.toggle_mqtt_publishing()
            elif key == ord('1'):
                app.adjust_face_sensitivity(-10)
            elif key == ord('2') and key != ord('l'):  # F2
                app.adjust_face_sensitivity(10)
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
