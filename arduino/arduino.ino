#include <ArduinoJson.h>
#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>
#include <math.h>

// WiFi credentials
const char *ssid = "__h_a_c_k_e_r";
const char *password = "wigo@hacker";

// MQTT settings
const char *mqtt_server = "192.168.8.101";
const int mqtt_port = 1883;
const char *mqtt_topic_sub = "vision/dragonfly/movement";
const char *mqtt_topic_pub = "servo/status";

// Pin definitions
const int servoPin = D1;  // GPIO14 as per exam spec

// Servo limits
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

// Default positions
const int DEFAULT_ANGLE = 90;
const int CENTER_ANGLE = 90;

// Movement parameters
const float STEP_SIZE = 2.0;        // Degrees per update (smaller = smoother)
const float MAX_SPEED = 60.0;       // Max degrees per second
const float ACCELERATION = 120.0;   // Degrees per second²
const float DEADBAND = 1.0;         // Stop if within this many degrees of target
const unsigned long UPDATE_INTERVAL = 10;  // milliseconds between updates

Servo myServo;
WiFiClient espClient;
PubSubClient client(espClient);

// Motion control variables
float currentAngle = DEFAULT_ANGLE;
float targetAngle = DEFAULT_ANGLE;
float currentVelocity = 0.0;
unsigned long lastUpdateTime = 0;
unsigned long lastMsgTime = 0;
String lastStatus = "CENTERED";

// Scan mode variables
bool isScanning = false;
float scanStartTime = 0;
const float SCAN_PERIOD = 4.0;  // Seconds for full sweep
const float SCAN_AMPLITUDE = 60.0;  // Sweep ±60° from center

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void updateServoPosition() {
  unsigned long now = millis();
  float deltaTime = (now - lastUpdateTime) / 1000.0;
  lastUpdateTime = now;
  
  // Clamp deltaTime to avoid huge jumps after lag
  if (deltaTime > 0.1) deltaTime = 0.1;
  if (deltaTime <= 0) return;

  if (isScanning) {
    // Smooth sine wave scanning when no face detected
    float scanTime = (now / 1000.0) - scanStartTime;
    currentAngle = CENTER_ANGLE + SCAN_AMPLITUDE * sin(scanTime * (2.0 * PI / SCAN_PERIOD));
    
  } else {
    // Smooth motion towards target using acceleration/deceleration
    float error = targetAngle - currentAngle;
    
    // Check if we're within deadband
    if (abs(error) < DEADBAND) {
      currentVelocity = 0;
      currentAngle = targetAngle;
    } else {
      // Calculate desired velocity (proportional to error, with max speed)
      float desiredVelocity = constrain(error * 3.0, -MAX_SPEED, MAX_SPEED);
      
      // Smooth acceleration/deceleration
      float maxDeltaV = ACCELERATION * deltaTime;
      if (desiredVelocity > currentVelocity) {
        currentVelocity = min(currentVelocity + maxDeltaV, desiredVelocity);
      } else {
        currentVelocity = max(currentVelocity - maxDeltaV, desiredVelocity);
      }
      
      // Update position
      currentAngle += currentVelocity * deltaTime;
    }
  }
  
  // Constrain to valid range
  currentAngle = constrain(currentAngle, MIN_ANGLE, MAX_ANGLE);
  
  // Write to servo
  myServo.write((int)currentAngle);
}

void setTarget(int newTarget, const char *reason) {
  newTarget = constrain(newTarget, MIN_ANGLE, MAX_ANGLE);
  
  if (abs(newTarget - targetAngle) > DEADBAND) {
    Serial.print("🎯 New target: ");
    Serial.print(newTarget);
    Serial.print("° (from ");
    Serial.print(currentAngle, 1);
    Serial.print("°) - Reason: ");
    Serial.println(reason);
    
    targetAngle = newTarget;
    isScanning = false;
  }
}

void startScanning() {
  if (!isScanning) {
    Serial.println("🔍 Starting scan mode - searching for face...");
    isScanning = true;
    scanStartTime = millis() / 1000.0;
  }
}

void stopScanning() {
  if (isScanning) {
    Serial.println("✅ Face found - stopping scan");
    isScanning = false;
    targetAngle = currentAngle;  // Hold current position
  }
}

void callback(char *topic, byte *payload, unsigned int length) {
  Serial.println();
  Serial.println("========== MQTT Message Received ==========");
  Serial.print("Topic: ");
  Serial.println(topic);

  // Convert payload to string
  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.print("Payload: ");
  Serial.println(message);

  // Parse JSON from vision node
  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, message);

  if (error) {
    Serial.print("❌ JSON parsing failed: ");
    Serial.println(error.c_str());
    return;
  }

  // Extract fields
  const char *status = doc["status"] | "UNKNOWN";
  float confidence = doc["confidence"] | 0.0;
  const char *target = doc["target"] | "unknown";
  bool locked = doc["locked"] | false;
  
  Serial.println("--- Parsed Data ---");
  Serial.print("Status: ");
  Serial.println(status);
  Serial.print("Confidence: ");
  Serial.println(confidence, 2);
  Serial.print("Target: ");
  Serial.println(target);
  Serial.print("Locked: ");
  Serial.println(locked ? "YES" : "NO");
  Serial.println("-------------------");

  // Handle different statuses with smooth transitions
  if (strcmp(status, "MOVE_LEFT") == 0) {
    stopScanning();
    setTarget(currentAngle - 10, "MOVE_LEFT");
    Serial.println("⬅️ Tracking: Subject moved LEFT");
  } 
  else if (strcmp(status, "MOVE_RIGHT") == 0) {
    stopScanning();
    setTarget(currentAngle + 10, "MOVE_RIGHT");
    Serial.println("➡️ Tracking: Subject moved RIGHT");
  } 
  else if (strcmp(status, "CENTERED") == 0) {
    stopScanning();
    // Subject is centered - hold position
    targetAngle = currentAngle;
    currentVelocity = 0;
    Serial.println("✅ Subject CENTERED - holding position");
  } 
  else if (strcmp(status, "NO_FACE") == 0) {
    startScanning();
    Serial.println("👀 NO_FACE - searching...");
  } 
  else {
    Serial.print("❓ Unknown status: ");
    Serial.println(status);
  }

  lastStatus = String(status);
  Serial.println("===========================================");
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection to ");
    Serial.print(mqtt_server);
    Serial.print("...");

    String clientId = "ESP8266-SmoothTracker-";
    clientId += String(random(0xffff), HEX);

    if (client.connect(clientId.c_str())) {
      Serial.println("connected!");

      if (client.subscribe(mqtt_topic_sub)) {
        Serial.print("✅ Subscribed to: ");
        Serial.println(mqtt_topic_sub);
      }

      client.publish(mqtt_topic_pub, "ONLINE");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds...");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize servo with smooth motion parameters
  myServo.attach(servoPin, 500, 2400);  // Extended pulse range for full 0-180°
  myServo.write(DEFAULT_ANGLE);
  currentAngle = DEFAULT_ANGLE;
  targetAngle = DEFAULT_ANGLE;

  Serial.println();
  Serial.println("===========================================");
  Serial.println("  SMOOTH AI CAMERA TRACKER - ESP8266");
  Serial.println("===========================================");
  Serial.print("Servo Pin: D5 (GPIO14)");
  Serial.print("MQTT Topic: ");
  Serial.println(mqtt_topic_sub);
  Serial.print("Broker: ");
  Serial.println(mqtt_server);
  Serial.println();
  Serial.println("Motion Profile:");
  Serial.println("  - Acceleration/Deceleration smoothing");
  Serial.println("  - Sine wave scan when no face detected");
  Serial.println("  - 10ms update rate for fluid motion");
  Serial.println("===========================================");
  
  setup_wifi();

  // Setup MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  client.setBufferSize(2048);

  lastUpdateTime = millis();
  delay(1000);
  
  Serial.println("✅ Setup complete! Tracking ready...");
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Smooth servo update loop - runs at ~100Hz
  unsigned long now = millis();
  if (now - lastUpdateTime >= UPDATE_INTERVAL) {
    updateServoPosition();
  }

  // Heartbeat status update every 30 seconds
  if (now - lastMsgTime > 30000) {
    lastMsgTime = now;
    
    String statusMsg = String((int)currentAngle); 
    client.publish(mqtt_topic_pub, statusMsg.c_str());
    
    Serial.print("💓 Heartbeat - Angle: ");
    Serial.print(currentAngle, 1);
    Serial.print("° | Mode: ");
    Serial.println(isScanning ? "SCANNING" : "TRACKING");
  }
}