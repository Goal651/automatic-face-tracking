"""
Face Locking Application

Specialized application using the face tracking framework with:
- MQTT servo control integration
- UI visualization and overlays
- Action history management
- Keyboard controls
- Real-time face locking and tracking
"""

from __future__ import annotations

import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .face_tracker import FaceTracker, TrackingConfig, ActionRecord
from .servo_controller import ServoController, ServoConfig


class FaceLockingApp:
    """
    Main face locking application that combines face tracking
    with servo control and visualization
    """
    
    def __init__(self, 
                 target_identity: str = "Wilson",
                 db_path: str = "data/db/face_db.npz",
                 model_path: str = "models/embedder_arcface.onnx",
                 window_scale: float = 1.0,
                 mirror_mode: bool = True,
                 mqtt_config: Optional[ServoConfig] = None):
        
        # Initialize tracking config
        self.tracking_config = TrackingConfig(
            target_identity=target_identity,
            db_path=db_path,
            model_path=model_path,
            window_scale=window_scale,
            mirror_mode=mirror_mode
        )
        
        # Initialize components
        self.tracker = FaceTracker(self.tracking_config)
        self.servo = ServoController(mqtt_config or ServoConfig())
        
        # UI settings
        self.show_detailed_info = True
        self.show_landmarks = True
        self.show_confidence = True
        self.ui_alpha = 0.8
        self.last_ui_update = time.time()
        self.ui_update_interval = 0.1
        
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
            """Handle face lock acquisition"""
            self.servo.reset_centering()
            print(f"🔒 Locked onto {tracking_state.identity}")
        
        def on_lock_lost(tracking_state):
            """Handle face lock loss"""
            print(f"🔓 Lost track of {tracking_state.identity}")
        
        def on_action_detected(action: ActionRecord):
            """Handle action detection"""
            print(f"🎯 {action.description}")
        
        def on_frame_processed(result: Dict[str, Any]):
            """Handle frame processing - update servo"""
            frame_size = (640, 480)  # Will be updated with actual frame size
            self.servo.process_tracking_update(result, frame_size)
        
        # Register callbacks
        self.tracker.register_callback('on_lock_acquired', on_lock_acquired)
        self.tracker.register_callback('on_lock_lost', on_lock_lost)
        self.tracker.register_callback('on_action_detected', on_action_detected)
        self.tracker.register_callback('on_frame_processed', on_frame_processed)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame and return visualization
        
        Args:
            frame: Input camera frame
            
        Returns:
            Frame with overlays and visualizations
        """
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
            # Determine face status
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
            
            # Draw bounding box
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
            
            # Draw landmarks if enabled
            if self.show_landmarks:
                for x, y in face.kps.astype(int):
                    cv2.circle(vis, (x, y), 2, color, -1)
            
            # Draw status label
            label = f"{i+1}: {status}"
            cv2.putText(vis, label, (face.x1, face.y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence if enabled and available
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
        
        # Tracking info box
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
        
        # Draw semi-transparent background
        overlay = vis.copy()
        box_height = len(info_lines) * 20 + 10
        cv2.rectangle(overlay, (10, 60), (250, 60 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.ui_alpha, vis, 1 - self.ui_alpha, 0, vis)
        
        # Draw text
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (15, 75 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _draw_servo_info(self, vis: np.ndarray):
        """Draw servo status information"""
        h, w = vis.shape[:2]
        
        # Servo info
        current_angle = self.servo.get_current_angle()
        target_angle = self.servo.get_target_angle()
        is_scanning = self.servo.is_scanning()
        is_centered = self.servo.is_centered()
        
        servo_lines = [
            f"Servo: {current_angle:.1f}°",
            f"Target: {target_angle:.1f}°" if target_angle else "Target: N/A",
            f"Mode: {'SCAN' if is_scanning else 'TRACK'}",
            f"Centered: {'YES' if is_centered else 'NO'}"
        ]
        
        # Draw semi-transparent background
        overlay = vis.copy()
        box_height = len(servo_lines) * 20 + 10
        cv2.rectangle(overlay, (w - 200, 60), (w - 10, 60 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.ui_alpha, vis, 1 - self.ui_alpha, 0, vis)
        
        # Draw text
        for i, line in enumerate(servo_lines):
            cv2.putText(vis, line, (w - 195, 75 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _draw_action_indicators(self, vis: np.ndarray, actions):
        """Draw action detection indicators"""
        h, w = vis.shape[:2]
        
        if not actions:
            return
        
        # Show recent actions
        recent_actions = actions[-3:]  # Show last 3 actions
        for i, action in enumerate(recent_actions):
            # Action indicator with color coding
            if "blink" in action.action_type:
                color = (255, 255, 0)  # Yellow for blink
            elif "smile" in action.action_type:
                color = (0, 255, 255)  # Cyan for smile
            elif "moved" in action.action_type:
                color = (255, 165, 0)  # Orange for movement
            else:
                color = (255, 255, 255)  # White for other
            
            # Draw action text
            text = f"🎯 {action.action_type}"
            cv2.putText(vis, text, (10, h - 30 - i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_ui_header(self, vis: np.ndarray, result: Dict[str, Any]):
        """Draw main UI header with status"""
        h, w = vis.shape[:2]
        
        # Status information
        is_locked = result.get('is_locked', False)
        frame_count = self.tracker.frame_count
        action_count = len(self.tracker.action_history)
        
        header_text = (
            f"Face Locking | Target: {self.tracking_config.target_identity} | "
            f"Status: {'LOCKED' if is_locked else 'SEARCHING'} | "
            f"Frame: {frame_count} | Actions: {action_count}"
        )
        
        # Draw header background
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)
        
        # Draw header text
        status_color = (0, 255, 0) if is_locked else (0, 0, 255)
        cv2.putText(vis, header_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    def save_action_history(self, filename: Optional[str] = None):
        """Save action history to timestamped file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"actions_{timestamp}.json"
        
        filepath = self.history_dir / filename
        
        # Convert action records to dict
        history_data = []
        for action in self.tracker.action_history:
            action_dict = {
                'timestamp': action.timestamp,
                'action_type': action.action_type,
                'description': action.description,
                'value': action.value
            }
            history_data.append(action_dict)
        
        # Save to file
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
        """Check if currently locked onto target face"""
        return self.tracker.is_locked
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        tracker_status = self.tracker.get_status_summary()
        servo_status = {
            'current_angle': self.servo.get_current_angle(),
            'target_angle': self.servo.get_target_angle(),
            'is_scanning': self.servo.is_scanning(),
            'is_centered': self.servo.is_centered(),
            'enabled': self.servo.enabled
        }
        
        return {
            'tracker': tracker_status,
            'servo': servo_status,
            'ui': {
                'show_detailed_info': self.show_detailed_info,
                'show_landmarks': self.show_landmarks,
                'show_confidence': self.show_confidence
            }
        }
    
    def shutdown(self):
        """Shutdown the application"""
        print("Shutting down Face Locking Application...")
        
        # Save final action history
        if self.tracker.action_history:
            self.save_action_history("final_actions.json")
        
        # Shutdown components
        self.servo.shutdown()
        
        print("Shutdown complete")
