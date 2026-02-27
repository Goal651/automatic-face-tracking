# src/action_detection.py
"""
Advanced Action Detection Module

Provides more sophisticated algorithms for detecting face actions:
- Movement tracking with smoothing
- Blink detection using eye aspect ratio
- Smile detection using mouth geometry
- Expression analysis
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EyeMetrics:
    """Eye-related measurements for blink detection"""
    left_eye_ratio: float
    right_eye_ratio: float
    avg_eye_ratio: float
    blink_detected: bool


@dataclass
class MouthMetrics:
    """Mouth-related measurements for smile detection"""
    mouth_width: float
    mouth_height: float
    mouth_ratio: float
    smile_detected: bool


class AdvancedActionDetector:
    """Advanced action detection with improved algorithms"""
    
    def __init__(self):
        # Movement detection
        self.movement_threshold = 25  # pixels
        self.movement_smoothing = 3   # frames to smooth over
        
        # Blink detection parameters
        self.eye_aspect_ratio_threshold = 0.25
        self.blink_consecutive_frames = 2
        self.eye_closed_threshold = 0.2
        
        # Smile detection parameters  
        self.smile_ratio_threshold = 1.2  # adjusted for combined score
        self.smile_consecutive_frames = 2  # reduced for more responsive detection
        self.smile_width_threshold = 0.15  # minimum normalized mouth width
        self.smile_elevation_threshold = -2  # mouth corner elevation (negative = up)
        
        # State tracking
        self.blink_frame_counter = 0
        self.smile_frame_counter = 0
        self.last_blink_time = 0
        self.last_smile_time = 0
        
        # History for smoothing
        self.position_history = deque(maxlen=10)
        self.eye_ratio_history = deque(maxlen=5)
        self.mouth_ratio_history = deque(maxlen=5)
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        For 5-point landmarks, we approximate using eye positions
        In a full implementation, you'd use 6 points per eye
        """
        try:
            # For 5-point landmarks, we only have eye centers
            # This is a simplified approximation
            left_eye = eye_landmarks[0]  # left eye center
            right_eye = eye_landmarks[1] # right eye center
            
            # Estimate eye opening based on relative positions
            # In practice, you'd need more detailed eye landmarks
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Simulate EAR calculation (normally uses 6 points per eye)
            # This is a placeholder - real EAR needs vertical/horizontal eye measurements
            estimated_ear = min(1.0, eye_distance / 100.0)  # normalized estimate
            
            return estimated_ear
            
        except Exception:
            return 0.3  # default "open" value
    
    def detect_blink_advanced(self, landmarks: np.ndarray, frame_time: float) -> Tuple[bool, EyeMetrics]:
        """
        Advanced blink detection using eye aspect ratio
        
        Args:
            landmarks: (5, 2) array with facial landmarks
            frame_time: current frame timestamp
            
        Returns:
            (blink_detected, eye_metrics)
        """
        try:
            # Calculate eye aspect ratios
            left_ear = self.calculate_eye_aspect_ratio(landmarks)
            right_ear = self.calculate_eye_aspect_ratio(landmarks)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Add to history for smoothing
            self.eye_ratio_history.append(avg_ear)
            
            # Smooth the EAR values
            if len(self.eye_ratio_history) >= 3:
                smoothed_ear = np.mean(list(self.eye_ratio_history)[-3:])
            else:
                smoothed_ear = avg_ear
            
            # Detect blink
            blink_detected = False
            
            if smoothed_ear < self.eye_aspect_ratio_threshold:
                self.blink_frame_counter += 1
            else:
                if self.blink_frame_counter >= self.blink_consecutive_frames:
                    # Blink completed
                    blink_detected = True
                    self.last_blink_time = frame_time
                self.blink_frame_counter = 0
            
            metrics = EyeMetrics(
                left_eye_ratio=left_ear,
                right_eye_ratio=right_ear,
                avg_eye_ratio=smoothed_ear,
                blink_detected=blink_detected
            )
            
            return blink_detected, metrics
            
        except Exception:
            return False, EyeMetrics(0.3, 0.3, 0.3, False)
    
    def detect_smile_advanced(self, landmarks: np.ndarray, frame_time: float) -> Tuple[bool, MouthMetrics]:
        """
        Advanced smile detection using improved mouth geometry
        
        Args:
            landmarks: (5, 2) array with facial landmarks
            frame_time: current frame timestamp
            
        Returns:
            (smile_detected, mouth_metrics)
        """
        try:
            # Extract mouth and reference points
            left_mouth = landmarks[3]   # left mouth corner
            right_mouth = landmarks[4]  # right mouth corner
            nose = landmarks[2]         # nose tip
            left_eye = landmarks[0]     # left eye center
            right_eye = landmarks[1]    # right eye center
            
            # Calculate mouth dimensions
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            mouth_center = (left_mouth + right_mouth) / 2
            
            # Calculate face width for normalization
            face_width = np.linalg.norm(right_eye - left_eye)
            
            # Improved mouth height estimation using multiple reference points
            # Use distance from mouth center to nose as primary reference
            mouth_to_nose_dist = np.linalg.norm(mouth_center - nose)
            
            # Calculate eye-line to mouth distance for better height estimation
            eye_center = (left_eye + right_eye) / 2
            eye_to_mouth_dist = np.linalg.norm(mouth_center - eye_center)
            
            # Estimate mouth height using facial proportions
            mouth_height = mouth_to_nose_dist * 0.4  # improved approximation
            
            # Normalize mouth width by face width for better consistency
            normalized_mouth_width = mouth_width / max(face_width, 1.0)
            
            # Calculate multiple smile indicators
            mouth_ratio = mouth_width / max(mouth_height, 1.0)
            width_to_face_ratio = normalized_mouth_width
            
            # Mouth corner elevation detection
            # Check if mouth corners are elevated relative to mouth center
            left_elevation = left_mouth[1] - mouth_center[1]  # negative means elevated
            right_elevation = right_mouth[1] - mouth_center[1]
            avg_elevation = (left_elevation + right_elevation) / 2
            
            # Combined smile score using multiple factors
            smile_score = (
                mouth_ratio * 0.4 +                    # mouth width/height ratio
                width_to_face_ratio * 3.0 +            # normalized width
                max(0, -avg_elevation / 5.0) * 0.6     # corner elevation bonus
            )
            
            # Add to history for smoothing
            self.mouth_ratio_history.append(smile_score)
            
            # Smooth the score
            if len(self.mouth_ratio_history) >= 3:
                smoothed_score = np.mean(list(self.mouth_ratio_history)[-3:])
            else:
                smoothed_score = smile_score
            
            # Detect smile with improved threshold
            smile_detected = False
            smile_threshold = 1.2  # adjusted threshold for combined score
            
            if smoothed_score > smile_threshold:
                self.smile_frame_counter += 1
                if self.smile_frame_counter >= self.smile_consecutive_frames:
                    smile_detected = True
                    self.last_smile_time = frame_time
            else:
                self.smile_frame_counter = 0
            
            metrics = MouthMetrics(
                mouth_width=mouth_width,
                mouth_height=mouth_height,
                mouth_ratio=smoothed_score,  # using combined score
                smile_detected=smile_detected
            )
            
            return smile_detected, metrics
            
        except Exception as e:
            return False, MouthMetrics(0, 0, 0, False)
    
    def detect_movement_advanced(self, current_pos: Tuple[int, int], 
                               frame_time: float) -> Optional[Dict[str, Any]]:
        """
        Advanced movement detection with smoothing and velocity
        
        Args:
            current_pos: current face center position (x, y)
            frame_time: current frame timestamp
            
        Returns:
            Movement information dict or None
        """
        self.position_history.append((current_pos, frame_time))
        
        if len(self.position_history) < 4:
            return None
        
        # Get recent positions
        recent_positions = list(self.position_history)[-4:]
        
        # Calculate movement vector
        start_pos, start_time = recent_positions[0]
        end_pos, end_time = recent_positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dt = max(end_time - start_time, 0.001)  # avoid division by zero
        
        # Calculate velocity
        velocity_x = dx / dt
        velocity_y = dy / dt
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Check for significant movement
        if abs(dx) > self.movement_threshold or abs(dy) > self.movement_threshold:
            
            # Determine primary direction
            if abs(dx) > abs(dy):
                # Horizontal movement
                direction = "face_moved_right" if dx > 0 else "face_moved_left"
            else:
                # Vertical movement  
                direction = "face_moved_down" if dy > 0 else "face_moved_up"
            
            return {
                "direction": direction,
                "displacement": (dx, dy),
                "velocity": (velocity_x, velocity_y),
                "speed": speed,
                "distance": np.sqrt(dx**2 + dy**2)
            }
        
        return None
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get summary of recent action detection state"""
        return {
            "blink_frames": self.blink_frame_counter,
            "smile_frames": self.smile_frame_counter,
            "last_blink": self.last_blink_time,
            "last_smile": self.last_smile_time,
            "position_history_length": len(self.position_history),
            "eye_ratio_avg": np.mean(list(self.eye_ratio_history)) if self.eye_ratio_history else 0,
            "mouth_ratio_avg": np.mean(list(self.mouth_ratio_history)) if self.mouth_ratio_history else 0
        }
    
    def reset_state(self):
        """Reset all detection state"""
        self.blink_frame_counter = 0
        self.smile_frame_counter = 0
        self.last_blink_time = 0
        self.last_smile_time = 0
        self.position_history.clear()
        self.eye_ratio_history.clear()
        self.mouth_ratio_history.clear()


class ActionClassifier:
    """Classifies and filters detected actions"""
    
    def __init__(self):
        self.action_cooldown = 1.0  # seconds between same action types
        self.last_actions = {}      # action_type -> timestamp
    
    def should_record_action(self, action_type: str, current_time: float) -> bool:
        """Check if action should be recorded based on cooldown"""
        if action_type not in self.last_actions:
            self.last_actions[action_type] = current_time
            return True
        
        time_since_last = current_time - self.last_actions[action_type]
        if time_since_last >= self.action_cooldown:
            self.last_actions[action_type] = current_time
            return True
        
        return False
    
    def classify_movement(self, movement_info: Dict[str, Any]) -> str:
        """Classify movement type with additional context"""
        direction = movement_info["direction"]
        speed = movement_info["speed"]
        distance = movement_info["distance"]
        
        # Add speed/intensity classification
        if speed > 100:
            intensity = "fast"
        elif speed > 50:
            intensity = "medium"
        else:
            intensity = "slow"
        
        return f"{direction}_{intensity}"
    
    def get_action_description(self, action_type: str, metrics: Any = None) -> str:
        """Generate human-readable action description"""
        descriptions = {
            "face_moved_left": "Face moved to the left",
            "face_moved_right": "Face moved to the right", 
            "face_moved_up": "Face moved upward",
            "face_moved_down": "Face moved downward",
            "eye_blink": "Eye blink detected",
            "smile": "Smile detected",
            "smile_end": "Smile ended"
        }
        
        base_desc = descriptions.get(action_type, f"Action: {action_type}")
        
        # Add metrics if available
        if metrics and hasattr(metrics, '__dict__'):
            if hasattr(metrics, 'speed'):
                base_desc += f" (speed: {metrics.speed:.1f})"
            elif hasattr(metrics, 'avg_eye_ratio'):
                base_desc += f" (EAR: {metrics.avg_eye_ratio:.3f})"
            elif hasattr(metrics, 'mouth_ratio'):
                base_desc += f" (mouth ratio: {metrics.mouth_ratio:.2f})"
        
        return base_desc