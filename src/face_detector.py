# src/face_detector.py
"""
Dedicated Face Detection Module

This module provides reliable face detection with validation and filtering:
- Haar + MediaPipe FaceMesh 5-point landmark detection
- Face quality validation and filtering
- Consistent bounding box calculation from landmarks
- Reusable across all face processing modules
- Integrated with FaceLocking repository components

Key improvements over existing system:
1. Better face validation to prevent false positives
2. Consistent coordinate system using 5-point landmarks
3. Quality filtering to reject poor face detections
4. Single source of truth for face detection
5. Uses proven Haar5ptDetector from FaceLocking
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import FaceLocking components
from .haar_5pt import Haar5ptDetector, FaceKpsBox, align_face_5pt, _kps_span_ok


@dataclass
class FaceDetectionResult:
    """Result of face detection with quality metrics
    
    Compatible with FaceKpsBox from FaceLocking but with additional
    quality metrics for the servo tracking system.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 - facial landmarks
    confidence: float
    quality_score: float
    is_valid: bool
    
    @property
    def x(self) -> int:
        return self.x1
    
    @property
    def y(self) -> int:
        return self.y1
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def landmarks(self) -> Optional[np.ndarray]:
        return self.kps


class RobustFaceDetector:
    """Robust face detection using FaceLocking's Haar5ptDetector
    
    This class wraps the proven Haar5ptDetector from the FaceLocking
    repository and adds quality scoring and validation for servo tracking.
    """
    
    def __init__(self, 
                 min_size: Tuple[int, int] = (70, 70),
                 confidence_threshold: float = 0.7,
                 quality_threshold: float = 0.3,
                 enable_mediapipe: bool = True,
                 smooth_alpha: float = 0.80,
                 debug: bool = False):
        self.min_size = min_size
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        self.enable_mediapipe = enable_mediapipe
        self.debug = debug
        
        # Initialize FaceLocking's Haar5ptDetector
        self.haar_5pt_detector = Haar5ptDetector(
            min_size=min_size,
            smooth_alpha=smooth_alpha,
            debug=debug
        )
        
        print(f"Robust Face Detector initialized with FaceLocking components")
        print(f"Min size: {self.min_size}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"MediaPipe: {'Enabled' if self.enable_mediapipe else 'Disabled'}")
    
    def _calculate_quality_score(self, kps: np.ndarray, face_img: np.ndarray) -> float:
        """Calculate face quality score based on landmark stability"""
        try:
            # Factor 1: Landmark geometry validation using FaceLocking's _kps_span_ok
            if not _kps_span_ok(kps, min_eye_dist=12.0):
                return 0.0  # Invalid geometry
            
            # Factor 2: Image clarity (Laplacian variance)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_score = max(0, 1.0 - laplacian_var / 1000)
            
            # Factor 3: Face size relative to detection bounds
            bbox_width = np.max(kps[:, 0]) - np.min(kps[:, 0])
            bbox_height = np.max(kps[:, 1]) - np.min(kps[:, 1])
            area_ratio = (bbox_width * bbox_height) / (face_img.shape[0] * face_img.shape[1])
            size_score = min(1.0, area_ratio * 3)  # Penalize too small faces
            
            # Factor 4: Landmark stability (variance check)
            landmarks_2d = kps[:, :2]
            variance = np.var(landmarks_2d, axis=0)
            stability_score = max(0, 1.0 - np.mean(variance) / 1000)
            
            # Combine scores
            quality_score = (stability_score * 0.3 + clarity_score * 0.3 + size_score * 0.4)
            
            return quality_score
            
        except Exception:
            return 0.5  # Default medium quality
    
    def detect_faces(self, frame: np.ndarray, max_faces: int = 10) -> List[FaceDetectionResult]:
        """Main face detection method using FaceLocking's Haar5ptDetector"""
        results = []
        
        # Use FaceLocking's proven detection with higher max_faces
        face_boxes = self.haar_5pt_detector.detect(frame, max_faces=max_faces)
         
        if self.debug:
            print(f"[face_detector] FaceLocking detected {len(face_boxes)} faces")
        
        # Convert FaceKpsBox to FaceDetectionResult with quality scoring
        for face_box in face_boxes:
            # Extract face ROI for quality assessment
            face_roi = frame[face_box.y1:face_box.y2, face_box.x1:face_box.x2]
            
            if face_roi.size == 0:
                continue
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(face_box.kps, face_roi)
            
            # Apply quality threshold (more lenient for multiple faces)
            is_valid = quality_score >= (self.quality_threshold * 0.8)  # 20% more lenient
            
            if is_valid:
                result = FaceDetectionResult(
                    x1=face_box.x1,
                    y1=face_box.y1,
                    x2=face_box.x2,
                    y2=face_box.y2,
                    score=face_box.score,
                    kps=face_box.kps,
                    confidence=face_box.score,  # FaceLocking uses 1.0 for valid detections
                    quality_score=quality_score,
                    is_valid=is_valid
                )
                results.append(result)
                
                if self.debug:
                    print(f"[face_detector] Valid face: quality={quality_score:.2f}")
            else:
                if self.debug:
                    print(f"[face_detector] Rejected face: quality={quality_score:.2f}")
        
        # Sort by quality score (best first)
        results.sort(key=lambda x: x.quality_score, reverse=True)
        
        return results
    
    def get_largest_face(self, frame: np.ndarray) -> Optional[FaceDetectionResult]:
        """Get the single best face detection"""
        faces = self.detect_faces(frame, max_faces=1)
        return faces[0] if faces else None
    
    def draw_detections(self, frame: np.ndarray, results: List[FaceDetectionResult]) -> np.ndarray:
        """Draw face detection results with quality indicators"""
        vis = frame.copy()
        
        for result in results:
            x, y, w, h = result.x, result.y, result.width, result.height
            
            # Color based on quality
            if result.is_valid:
                color = (0, 255, 0)  # Green for good detections
                label = f"VALID ({result.quality_score:.2f})"
            else:
                color = (0, 0, 255)  # Red for invalid detections
                label = f"INVALID ({result.quality_score:.2f})"
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
            # Draw landmarks if available
            if result.kps is not None:
                for i, (lx, ly) in enumerate(result.kps[:, :2].astype(int)):
                    cv2.circle(vis, (lx, ly), 2, (255, 255, 255), -1)
            
            # Draw quality label
            cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis
    
    def align_face(self, frame: np.ndarray, face_result: FaceDetectionResult, out_size: Tuple[int, int] = (112, 112)) -> Tuple[np.ndarray, np.ndarray]:
        """Align face using FaceLocking's align_face_5pt function"""
        if face_result.kps is None:
            raise ValueError("No landmarks available for face alignment")
        
        return align_face_5pt(frame, face_result.kps, out_size=out_size)


# ============================================================================
# Demo and Testing
# ============================================================================

def main():
    """Test the updated face detector with FaceLocking components"""
    print("Robust Face Detection Test with FaceLocking Integration")
    print("Press 'q' to quit")
    
    detector = RobustFaceDetector(
        min_size=(70, 70),
        confidence_threshold=0.7,
        quality_threshold=0.3,
        enable_mediapipe=True,
        smooth_alpha=0.80,
        debug=True
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nControls:")
    print("q: quit")
    print("Space: align and show first detected face")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        results = detector.detect_faces(frame, max_faces=3)
        
        # Draw results
        vis = detector.draw_detections(frame, results)
        
        # Show statistics
        valid_count = sum(1 for r in results if r.is_valid)
        cv2.putText(vis, f"Valid: {valid_count}/{len(results)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show FaceLocking integration info
        cv2.putText(vis, "FaceLocking Integration Active", (10, vis.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Robust Face Detection - FaceLocking", vis)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and results:
            # Test face alignment
            try:
                aligned, transform = detector.align_face(frame, results[0])
                cv2.imshow("Aligned Face", aligned)
                print(f"Face aligned successfully. Shape: {aligned.shape}")
            except Exception as e:
                print(f"Alignment failed: {e}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
