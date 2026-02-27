"""
Serial Communication Module for Face Locking System

This module handles serial communication with Arduino/ESP8266 for servo control:
- JSON message formatting and transmission
- Serial port management
- Error handling and reconnection
- Compatible with the Arduino servo controller code
"""

import json
import time
import serial
import serial.tools.list_ports
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SerialConfig:
    """Serial communication configuration"""
    
    port: str = "/dev/ttyUSB0"  # Default USB port
    baudrate: int = 115200
    timeout: float = 1.0
    publish_interval: float = 0.1  # seconds between messages
    confidence_threshold: float = 0.7
    # Servo angle range (degrees)
    servo_min_angle: int = 0
    servo_max_angle: int = 180
    servo_center_angle: int = 90  # angle sent when no face is detected


class SerialCommunicator:
    """Serial communicator for Arduino servo control"""
    
    def __init__(self, config: SerialConfig):
        self.config = config
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.last_publish_time = 0
        self.last_published_angle: int = -1
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
    def _find_available_ports(self) -> List[str]:
        """Find available serial ports"""
        ports = serial.tools.list_ports.comports()
        available_ports = []
        
        for port in ports:
            # Filter for likely Arduino/ESP devices
            if any(keyword in port.description.lower() 
                   for keyword in ['arduino', 'ch340', 'cp210', 'ftdi', 'usb']):
                available_ports.append(port.device)
        
        # If no specific ports found, return all available ports
        if not available_ports:
            available_ports = [port.device for port in ports]
            
        return available_ports
    
    def connect(self) -> bool:
        """Connect to serial port"""
        if self.is_connected and self.serial_conn:
            return True
            
        # Try the configured port first
        ports_to_try = [self.config.port]
        
        # If configured port fails, try auto-detection
        available_ports = self._find_available_ports()
        for port in available_ports:
            if port not in ports_to_try:
                ports_to_try.append(port)
        
        for port in ports_to_try:
            try:
                print(f"🔌 Attempting to connect to {port}...")
                self.serial_conn = serial.Serial(
                    port=port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout
                )
                
                # Wait for connection to stabilize
                time.sleep(2)
                
                # Test connection by sending a simple command
                test_msg = '{"test":1}'
                self.serial_conn.write(test_msg.encode())
                time.sleep(0.1)
                
                self.is_connected = True
                self.connection_attempts = 0
                print(f"✅ Connected to {port} at {self.config.baudrate} baud")
                return True
                
            except (serial.SerialException, OSError) as e:
                print(f"❌ Failed to connect to {port}: {e}")
                if self.serial_conn:
                    self.serial_conn.close()
                    self.serial_conn = None
                continue
        
        self.connection_attempts += 1
        if self.connection_attempts >= self.max_connection_attempts:
            print(f"❌ Failed to connect after {self.max_connection_attempts} attempts")
        return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.is_connected = False
            print("🔌 Serial connection closed")
    
    def publish_movement(self, status: str, confidence: float, angle: Optional[int] = None, 
                        force: bool = False, frame: int = 0, target: str = "unknown") -> bool:
        """Publish movement command to Arduino"""
        if not self.is_connected or not self.serial_conn:
            return False
        
        # Rate limiting
        current_time = time.time()
        if not force and current_time - self.last_publish_time < self.config.publish_interval:
            return True
        
        # Skip duplicate angles unless forced
        if not force and angle is not None and angle == self.last_published_angle:
            return True
        
        # Prepare message
        servo_angle = angle if angle is not None else self.config.servo_center_angle
        
        message = {
            "status": status,
            "angle": servo_angle,
            "confidence": float(confidence),
            "timestamp": int(time.time()),
            "frame": frame,
            "target": target
        }
        
        try:
            # Send JSON message wrapped in {} as expected by Arduino
            json_str = json.dumps(message)
            wrapped_msg = f"{{{json_str}}}"  # Double wrap for Arduino parsing
            
            self.serial_conn.write(wrapped_msg.encode())
            self.serial_conn.flush()
            
            self.last_publish_time = current_time
            if angle is not None:
                self.last_published_angle = angle
            
            print(f"📤 Serial: {status} | angle={servo_angle}° (conf: {confidence:.2f})")
            return True
            
        except (serial.SerialException, OSError) as e:
            print(f"❌ Serial write error: {e}")
            self.is_connected = False
            return False
    
    def publish_status(self, status: str, message: str) -> bool:
        """Publish system status"""
        if not self.is_connected or not self.serial_conn:
            return False
        
        status_msg = {
            "system_status": status,
            "message": message,
            "timestamp": int(time.time())
        }
        
        try:
            json_str = json.dumps(status_msg)
            wrapped_msg = f"{{{json_str}}}"
            self.serial_conn.write(wrapped_msg.encode())
            self.serial_conn.flush()
            return True
            
        except (serial.SerialException, OSError) as e:
            print(f"❌ Serial status error: {e}")
            self.is_connected = False
            return False
    
    def read_response(self) -> Optional[str]:
        """Read response from Arduino"""
        if not self.is_connected or not self.serial_conn:
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                response = self.serial_conn.readline().decode('utf-8').strip()
                if response:
                    print(f"📥 Arduino: {response}")
                    return response
        except (serial.SerialException, OSError):
            self.is_connected = False
            return None
        
        return None
    
    def reconnect(self) -> bool:
        """Attempt to reconnect"""
        print("🔄 Attempting to reconnect...")
        self.disconnect()
        time.sleep(1)
        return self.connect()


def auto_detect_arduino_port() -> Optional[str]:
    """Auto-detect Arduino/ESP port"""
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        description = port.description.lower()
        if any(keyword in description for keyword in ['arduino', 'ch340', 'cp210', 'ftdi']):
            return port.device
    
    # If no specific keywords found, return first available port
    if ports:
        return ports[0].device
    
    return None


if __name__ == "__main__":
    # Test serial communication
    config = SerialConfig()
    
    # Auto-detect port if not specified
    if not config.port or config.port == "/dev/ttyUSB0":
        detected_port = auto_detect_arduino_port()
        if detected_port:
            config.port = detected_port
            print(f"Auto-detected port: {detected_port}")
    
    communicator = SerialCommunicator(config)
    
    if communicator.connect():
        print("Testing servo control...")
        
        # Test movements
        test_commands = [
            ("CENTERED", 0.8, 90),
            ("MOVE_LEFT", 0.9, 45),
            ("MOVE_RIGHT", 0.9, 135),
            ("NO_FACE", 0.0, 90)
        ]
        
        for status, confidence, angle in test_commands:
            success = communicator.publish_movement(status, confidence, angle, force=True)
            if success:
                time.sleep(2)
                response = communicator.read_response()
        
        communicator.disconnect()
    else:
        print("Failed to connect to Arduino")
