"""
MQTT Servo Controller Module

Provides servo control functionality with:
- MQTT communication for servo positioning
- P-controller for smooth face tracking
- Auto-scan mode when no face is detected
- Configurable servo parameters
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import paho.mqtt.client as mqtt


@dataclass
class ServoConfig:
    """Configuration for servo control system"""
    broker: str = "localhost"
    port: int = 1883
    topic_movement: str = "vision/team351/movement"
    topic_status: str = "robot/status"
    topic_servo_status: str = "servo/status"
    client_id: str = f"servo_controller_{int(time.time())}"
    publish_interval: float = 0.1  # seconds between publishes
    confidence_threshold: float = 0.7
    
    # Servo angle range (degrees)
    servo_min_angle: int = 0
    servo_max_angle: int = 180
    servo_center_angle: int = 90  # angle sent when no face is detected
    
    # P-controller settings
    servo_Kp: float = 60.0  # degrees per unit of normalized error
    servo_ema_alpha: float = 0.45  # smoothing factor (0=frozen, 1=no smoothing)
    
    # Auto-scan settings
    scan_enabled: bool = True
    scan_speed: float = 30.0  # degrees per second
    scan_min_angle: float = 20.0
    scan_max_angle: float = 160.0
    
    # Auto-stop on lock
    auto_stop_on_lock: bool = True
    centering_tolerance: float = 0.05  # 5% tolerance for centering
    centering_stability_threshold: int = 5  # frames to confirm centered


class ServoController:
    """
    MQTT-based servo controller with face tracking capabilities
    
    This controller handles servo positioning based on face location,
    with smooth P-controller and auto-scan functionality.
    """
    
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
        
        # Callbacks
        self.on_angle_changed: Optional[Callable[[float], None]] = None
        self.on_scan_state_changed: Optional[Callable[[bool], None]] = None
        
        # Initialize MQTT
        self._init_mqtt()
        
        print(f"Servo Controller initialized")
        print(f"MQTT Broker: {config.broker}:{config.port}")
        print(f"Movement Topic: {config.topic_movement}")
        print(f"Angle Range: {config.servo_min_angle}° - {config.servo_max_angle}°")
    
    def _init_mqtt(self):
        """Initialize MQTT connection and subscribe to servo feedback"""
        self.mqtt_client = mqtt.Client(client_id=self.config.client_id)
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"✅ Connected to MQTT broker at {self.config.broker}")
                self._publish_status("ONLINE", "Servo controller started")
                # Subscribe to servo feedback
                client.subscribe(self.config.topic_servo_status)
            else:
                print(f"❌ Failed to connect to MQTT broker: {rc}")
        
        def on_message(client, userdata, msg):
            """Handle incoming MQTT messages (servo feedback)"""
            try:
                payload = msg.payload.decode().strip()
                if msg.topic == self.config.topic_servo_status:
                    # Parse servo feedback (e.g., "ANGLE:90" or "POSITION:90")
                    if payload.startswith("ANGLE:"):
                        angle = float(payload.split(":")[1])
                        self._handle_servo_feedback(angle)
                    elif payload.startswith("POSITION:"):
                        angle = float(payload.split(":")[1])
                        self._handle_servo_feedback(angle)
            except Exception as e:
                print(f"MQTT message error: {e}")
        
        def on_disconnect(client, userdata, rc):
            if rc != 0:
                print(f"⚠️ Unexpected MQTT disconnection: {rc}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        self.mqtt_client.on_disconnect = on_disconnect
        
        # Connect to broker
        try:
            self.mqtt_client.connect(self.config.broker, self.config.port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"❌ Failed to connect to MQTT broker: {e}")
            self.enabled = False
    
    def _handle_servo_feedback(self, angle: float):
        """Handle feedback from servo about current position"""
        self._current_servo_angle = angle
        if self.on_angle_changed:
            self.on_angle_changed(angle)
    
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
            # Only publish if angle changed significantly
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
        if self.on_scan_state_changed:
            self.on_scan_state_changed(False)
    
    def update_face_tracking(self, face_center: Tuple[int, int], frame_size: Tuple[int, int], 
                           is_locked: bool, confidence: float) -> Optional[int]:
        """
        Update servo position based on face tracking
        
        Args:
            face_center: (x, y) center of detected face
            frame_size: (width, height) of frame
            is_locked: whether face is currently locked
            confidence: recognition confidence
            
        Returns:
            Current servo angle or None if not tracking
        """
        if not self.enabled or confidence < self.config.confidence_threshold:
            return None
        
        frame_width, frame_height = frame_size
        face_x, face_y = face_center
        
        # Normalize face position (-1 to +1, where 0 is center)
        normalized_x = (face_x - frame_width / 2) / (frame_width / 2)
        
        # Auto-stop on lock: center immediately on first lock
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
        
        # P-controller for servo positioning
        error = -normalized_x  # Negative because servo direction is opposite
        control_signal = self.config.servo_Kp * error
        
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
        if self.on_scan_state_changed:
            self.on_scan_state_changed(False)
        
        # Publish angle command
        angle_int = int(round(target_angle))
        self._publish_angle(angle_int)
        
        return angle_int
    
    def update_scan_mode(self, current_time: float) -> Optional[int]:
        """
        Update servo position in scan mode (when no face is detected)
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Current servo angle or None if not scanning
        """
        if not self.enabled or not self.config.scan_enabled:
            return None
        
        # Start scanning if not already
        if not self._scanning:
            self._scanning = True
            self._scan_angle = self._current_servo_angle
            self._scan_last_time = current_time
            if self.on_scan_state_changed:
                self.on_scan_state_changed(True)
        
        # Calculate scan movement
        dt = current_time - self._scan_last_time
        if dt < 0.001:  # Avoid division by zero
            return int(round(self._scan_angle))
        
        # Update scan angle
        angle_change = self._scan_speed * dt * self._scan_direction
        self._scan_angle += angle_change
        
        # Check bounds and reverse direction
        if self._scan_angle >= self.config.scan_max_angle:
            self._scan_angle = self.config.scan_max_angle
            self._scan_direction = -1.0
        elif self._scan_angle <= self.config.scan_min_angle:
            self._scan_angle = self.config.scan_min_angle
            self._scan_direction = 1.0
        
        self._scan_last_time = current_time
        self._target_servo_angle = self._scan_angle
        
        # Publish angle command
        angle_int = int(round(self._scan_angle))
        self._publish_angle(angle_int)
        
        return angle_int
    
    def process_tracking_update(self, tracking_result: dict, frame_size: Tuple[int, int]) -> Optional[int]:
        """
        Process tracking update and update servo accordingly
        
        Args:
            tracking_result: Result from face tracker
            frame_size: (width, height) of frame
            
        Returns:
            Current servo angle or None
        """
        current_time = tracking_result.get('frame_time', time.time())
        is_locked = tracking_result.get('is_locked', False)
        
        if is_locked and tracking_result.get('target_face') and tracking_result.get('match_result'):
            # Face is locked - track it
            face = tracking_result['target_face']
            face_center = ((face.x1 + face.x2) // 2, (face.y1 + face.y2) // 2)
            confidence = tracking_result['match_result'].similarity
            
            return self.update_face_tracking(face_center, frame_size, is_locked, confidence)
        else:
            # No face locked - scan mode
            return self.update_scan_mode(current_time)
    
    def get_current_angle(self) -> float:
        """Get current servo angle"""
        return self._current_servo_angle
    
    def get_target_angle(self) -> Optional[float]:
        """Get target servo angle"""
        return self._target_servo_angle
    
    def is_scanning(self) -> bool:
        """Check if currently in scan mode"""
        return self._scanning
    
    def is_centered(self) -> bool:
        """Check if face is centered (for auto-stop)"""
        return self._lock_centering_complete
    
    def reset_centering(self):
        """Reset centering state (for new lock)"""
        self._lock_centering_complete = False
        self._centered_stable_frames = 0
    
    def set_scan_parameters(self, speed: Optional[float] = None, 
                           min_angle: Optional[float] = None,
                           max_angle: Optional[float] = None):
        """Update scan parameters"""
        if speed is not None:
            self.config.scan_speed = speed
        if min_angle is not None:
            self.config.scan_min_angle = min_angle
        if max_angle is not None:
            self.config.scan_max_angle = max_angle
    
    def set_controller_parameters(self, Kp: Optional[float] = None,
                                ema_alpha: Optional[float] = None):
        """Update P-controller parameters"""
        if Kp is not None:
            self.config.servo_Kp = Kp
        if ema_alpha is not None:
            self.config.servo_ema_alpha = max(0.0, min(1.0, ema_alpha))
    
    def shutdown(self):
        """Shutdown servo controller"""
        self._publish_status("OFFLINE", "Servo controller shutting down")
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        print("Servo controller shutdown complete")
