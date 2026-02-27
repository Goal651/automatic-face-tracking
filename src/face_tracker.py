"""
Base Face Tracking Framework

Provides a reusable framework for face tracking applications with:
- Target face selection and tracking
- Action detection integration
- Extensible callback system for custom behaviors
- Face recognition integration
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from collections import deque
from pathlib import Path

import cv2
import numpy as np

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
class FaceTrackingState:
    """Current state of face tracking"""
    identity: str
    last_position: Tuple[int, int]  # center (x, y)
    last_seen_frame: int
    confidence_history: deque
    position_history: deque
    blink_state: str
    blink_counter: int
    smile_state: bool
    lock_start_time: float
    is_locked: bool = False
    
    def __post_init__(self):
        if not hasattr(self, "confidence_history") or self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)
        if not hasattr(self, "position_history") or self.position_history is None:
            self.position_history = deque(maxlen=5)


@dataclass
class ActionRecord:
    """Single action record with timestamp"""
    timestamp: str
    action_type: str
    description: str
    value: Optional[float] = None


class FaceTracker:
    """
    Base face tracking system with extensible callback architecture
    
    This class provides the core face tracking functionality that can be
    extended for different applications (servo control, UI, recording, etc.)
    """
    
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
        
        # Callback system for extensibility
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
        
        # Try cache first
        mr = None
        needs_recognition = self.frame_count % self.config.recognition_interval == 0
        
        # Clean old cache entries
        self.identity_cache = {
            pos: val
            for pos, val in self.identity_cache.items()
            if self.frame_count - val[1] < 5
        }
        
        if not needs_recognition:
            # Check spatial proximity in cache
            for pos, (cached_mr, last_fidx) in self.identity_cache.items():
                dist = np.sqrt((center[0] - pos[0])**2 + (center[1] - pos[1])**2)
                if dist < self.cache_distance_threshold:
                    mr = cached_mr
                    break
        
        # Perform recognition if needed
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
            # Create new tracking state
            self.tracking_state = FaceTrackingState(
                identity=mr.name,
                last_position=center,
                last_seen_frame=self.frame_count,
                confidence_history=deque(maxlen=10),
                position_history=deque(maxlen=5),
                blink_state="unknown",
                blink_counter=0,
                smile_state=False,
                lock_start_time=current_time
            )
            self.is_locked = True
            self._trigger_callbacks('on_lock_acquired', self.tracking_state)
        else:
            # Update existing state
            self.tracking_state.last_position = center
            self.tracking_state.last_seen_frame = self.frame_count
            self.tracking_state.confidence_history.append(mr.similarity)
            self.tracking_state.position_history.append(center)
        
        # Trigger face detected callback
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
            # Smile started
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
            # Smile ended
            action_type = "smile_end"
            if self.action_classifier.should_record_action(action_type, frame_time):
                description = self.action_classifier.get_action_description(action_type)
                actions.append(ActionRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    action_type=action_type,
                    description=description
                ))
            self.tracking_state.smile_state = False
        
        # Trigger action callbacks
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
        """
        Process a single frame and return tracking results
        
        Returns:
            Dict containing:
            - faces: List of all detected faces
            - target_face: Target face if found
            - tracking_state: Current tracking state
            - actions: Actions detected in this frame
            - is_locked: Whether target is currently locked
        """
        self.frame_count += 1
        frame_time = time.time()
        
        # Detect all faces
        faces = self.detect_faces(frame)
        
        # Find target face
        target_result = None
        if faces:
            target_result = self.find_target_face(frame, faces)
        
        # Update tracking if target found
        actions = []
        if target_result:
            face, mr = target_result
            self.update_tracking_state(face, mr)
            actions = self.detect_actions(face, frame_time)
        else:
            # Check for timeout
            self.check_lock_timeout()
        
        # Trigger frame processed callback
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
    
    def get_action_history(self, limit: Optional[int] = None) -> List[ActionRecord]:
        """Get action history, optionally limited"""
        if limit:
            return self.action_history[-limit:]
        return self.action_history.copy()
    
    def clear_action_history(self):
        """Clear action history"""
        self.action_history.clear()
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracking_state = None
        self.is_locked = False
        self.identity_cache.clear()
        self.action_detector.reset_state()
        self.action_classifier.last_actions.clear()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current tracking status summary"""
        return {
            'target_identity': self.config.target_identity,
            'is_locked': self.is_locked,
            'frame_count': self.frame_count,
            'tracking_state': self.tracking_state,
            'action_history_length': len(self.action_history),
            'cache_size': len(self.identity_cache),
            'action_detector_state': self.action_detector.get_action_summary()
        }
