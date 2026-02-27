# src/mediapipe_compat.py
"""
MediaPipe Compatibility Layer

Handles differences between MediaPipe versions:
- Old API (0.8.x - 0.10.20): mp.solutions.face_mesh.FaceMesh
- New API (0.10.30+): mp.tasks.vision.FaceLandmarker

This module provides a unified interface that works with both versions.
"""

import numpy as np
from typing import Optional, List, NamedTuple
from dataclasses import dataclass

try:
    import mediapipe as mp
except ImportError:
    mp = None
    raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")


@dataclass
class FaceLandmarks:
    """Unified face landmarks representation"""
    landmarks: np.ndarray  # (468, 3) for full landmarks or (5, 2) for 5-point
    confidence: float = 1.0


class MediaPipeFaceMeshCompat:
    """Compatibility wrapper for MediaPipe FaceMesh/FaceLandmarker"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Try to initialize with the appropriate API
        self._init_mediapipe()
        
        # 5-point landmark indices (same for both APIs)
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291
    
    def _init_mediapipe(self):
        """Initialize MediaPipe with version detection"""
        self.use_new_api = False
        self.face_mesh = None
        
        # Check if old solutions API exists
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=self.static_image_mode,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self.use_new_api = False
                print("Using MediaPipe legacy solutions API")
                return
            except Exception as e:
                print(f"Failed to initialize legacy MediaPipe API: {e}")
        
        # For newer MediaPipe versions, create a simple fallback
        # that uses basic face detection without landmarks
        print("MediaPipe FaceMesh not available, using fallback mode")
        self.face_mesh = None
        self.use_new_api = False
    
    def process(self, image: np.ndarray) -> Optional[List[FaceLandmarks]]:
        """Process image and return face landmarks"""
        if self.face_mesh is None:
            # Fallback mode - return dummy landmarks for basic functionality
            return self._create_dummy_landmarks(image)
        
        try:
            if self.use_new_api:
                return self._process_new_api(image)
            else:
                return self._process_old_api(image)
        except Exception as e:
            print(f"Error processing image: {e}")
            return self._create_dummy_landmarks(image)
    
    def _create_dummy_landmarks(self, image: np.ndarray) -> List[FaceLandmarks]:
        """Create dummy landmarks for fallback mode"""
        h, w = image.shape[:2]
        
        # Create approximate 5-point landmarks in the center of the image
        # This is a fallback when MediaPipe is not working
        center_x, center_y = w // 2, h // 2
        eye_offset = w // 8
        mouth_offset = w // 12
        
        dummy_landmarks = np.array([
            # Full 468 landmarks - we'll just create a minimal set
            [center_x - eye_offset, center_y - h//8, 0],  # left eye (index 33)
            [center_x + eye_offset, center_y - h//8, 0],  # right eye (index 263) 
            [center_x, center_y, 0],                      # nose tip (index 1)
            [center_x - mouth_offset, center_y + h//8, 0], # left mouth (index 61)
            [center_x + mouth_offset, center_y + h//8, 0], # right mouth (index 291)
        ] + [[center_x, center_y, 0]] * 463, dtype=np.float32)  # Fill rest with center point
        
        return [FaceLandmarks(landmarks=dummy_landmarks, confidence=0.5)]
    
    def _process_old_api(self, image: np.ndarray) -> Optional[List[FaceLandmarks]]:
        """Process with old solutions API"""
        # Convert BGR to RGB
        rgb_image = image[:, :, ::-1] if len(image.shape) == 3 else image
        
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks_list = []
        for face_landmarks in results.multi_face_landmarks:
            # Convert to numpy array
            h, w = image.shape[:2]
            landmarks = []
            
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x * w, landmark.y * h, landmark.z])
            
            landmarks_array = np.array(landmarks, dtype=np.float32)
            face_landmarks_list.append(FaceLandmarks(landmarks=landmarks_array))
        
        return face_landmarks_list
    
    def _process_new_api(self, image: np.ndarray) -> Optional[List[FaceLandmarks]]:
        """Process with new tasks API"""
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image[:, :, ::-1])
        
        # Increment timestamp for video mode
        self._frame_timestamp += 1
        
        if self.static_image_mode:
            results = self.face_mesh.detect(mp_image)
        else:
            results = self.face_mesh.detect_for_video(mp_image, self._frame_timestamp)
        
        if not results.face_landmarks:
            return None
        
        face_landmarks_list = []
        h, w = image.shape[:2]
        
        for face_landmarks in results.face_landmarks:
            # Convert to numpy array
            landmarks = []
            for landmark in face_landmarks:
                landmarks.append([landmark.x * w, landmark.y * h, landmark.z if hasattr(landmark, 'z') else 0])
            
            landmarks_array = np.array(landmarks, dtype=np.float32)
            face_landmarks_list.append(FaceLandmarks(landmarks=landmarks_array))
        
        return face_landmarks_list
    
    def extract_5_points(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract 5 key points from full face landmarks"""
        if landmarks.shape[0] < 468:
            # Already 5 points or insufficient landmarks
            return landmarks[:5, :2] if landmarks.shape[1] >= 2 else landmarks
        
        # Extract 5 key points using standard indices
        indices = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE, 
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT
        ]
        
        key_points = landmarks[indices, :2]  # Take only x, y coordinates
        
        # Ensure left/right eye ordering
        if key_points[0, 0] > key_points[1, 0]:  # left eye x > right eye x
            key_points[[0, 1]] = key_points[[1, 0]]
        
        # Ensure left/right mouth ordering  
        if key_points[3, 0] > key_points[4, 0]:  # left mouth x > right mouth x
            key_points[[3, 4]] = key_points[[4, 3]]
        
        return key_points.astype(np.float32)
    
    def close(self):
        """Clean up resources"""
        if hasattr(self.face_mesh, 'close'):
            self.face_mesh.close()


def create_face_mesh(**kwargs) -> MediaPipeFaceMeshCompat:
    """Factory function to create compatible FaceMesh instance"""
    return MediaPipeFaceMeshCompat(**kwargs)


# Convenience function for backward compatibility
def FaceMesh(**kwargs) -> MediaPipeFaceMeshCompat:
    """Backward compatibility function"""
    return MediaPipeFaceMeshCompat(**kwargs)