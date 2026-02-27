#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Servo.h>

// WiFi credentials
const char* ssid = "wigothehacker";
const char* password = "wigothehacker";

// MQTT settings
const char* mqtt_server = "10.42.0.1";
const int mqtt_port = 1883;
const char* mqtt_topic_sub = "vision/team351/movement";  
const char* mqtt_topic_angle = "servo/angle";           
const char* mqtt_topic_pub = "servo/status";             

// Pin definitions
const int servoPin = D7;

// Servo limits
const int minAngle = 0;
const int maxAngle = 180;

// Default positions
const int DEFAULT_ANGLE = 90;

Servo myServo;
WiFiClient espClient;
PubSubClient client(espClient);

int currentAngle = DEFAULT_ANGLE;
unsigned long lastMsgTime = 0;

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

void moveServo(int angle, const char* reason) {
  if (angle >= minAngle && angle <= maxAngle) {
    Serial.print("Moving servo to: ");
    Serial.print(angle);
    Serial.print("° - Reason: ");
    Serial.println(reason);
    
    myServo.write(angle);
    currentAngle = angle;
    
    // Publish movement confirmation
    String statusMsg = "Moved to " + String(angle) + "° due to: " + String(reason);
    client.publish(mqtt_topic_pub, statusMsg.c_str());
  } else {
    Serial.print("Invalid angle: ");
    Serial.println(angle);
  }
}

void handleMovementMessage(const char* status, int angle, float confidence, 
                          long timestamp, int frame, const char* target) {
  Serial.print("Status: ");
  Serial.print(status);
  Serial.print(" | Angle: ");
  Serial.print(angle);
  Serial.print("° | Confidence: ");
  Serial.print(confidence, 2);
  Serial.print(" | Target: ");
  Serial.print(target);
  Serial.print(" | Frame: ");
  Serial.println(frame);
  
  // Move servo to the exact angle from the message
  char reason[100];
  sprintf(reason, "%s (conf: %.2f, target: %s)", status, confidence, target);
  moveServo(angle, reason);
}

void handleAngleCommand(int angle) {
  char reason[50];
  sprintf(reason, "Direct angle command: %d°", angle);
  moveServo(angle, reason);
}

void callback(char* topic, byte* payload, unsigned int length) {
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
  
  // Check which topic the message came from
  if (strcmp(topic, mqtt_topic_angle) == 0) {
    // Direct angle command - payload should be a number
    int angle = message.toInt();
    Serial.print("Parsed angle: ");
    Serial.println(angle);
    
    if (angle >= minAngle && angle <= maxAngle) {
      handleAngleCommand(angle);
    } else {
      Serial.println("❌ Angle out of range (0-180)");
    }
  }
  else {
    // Main movement topic - parse JSON with all fields
    DynamicJsonDocument doc(512);  // Increased buffer for all fields
    DeserializationError error = deserializeJson(doc, message);
    
    if (error) {
      Serial.print("❌ JSON parsing failed: ");
      Serial.println(error.c_str());
      
      // Try to parse as simple angle if JSON fails
      int angle = message.toInt();
      if (angle >= minAngle && angle <= maxAngle) {
        Serial.println("Parsed as direct angle instead");
        handleAngleCommand(angle);
      }
      return;
    }
    
    // Extract all fields from the message
    const char* status = doc["status"] | "UNKNOWN";
    int angle = doc["angle"] | -1;
    float confidence = doc["confidence"] | 0.0;
    long timestamp = doc["timestamp"] | 0;
    int frame = doc["frame"] | 0;
    const char* target = doc["target"] | "unknown";
    
    // If angle is not in the message, map from status (fallback)
    if (angle == -1) {
      if (strcmp(status, "MOVE_LEFT") == 0) {
        angle = 0;
      } else if (strcmp(status, "MOVE_RIGHT") == 0) {
        angle = 180;
      } else if (strcmp(status, "CENTERED") == 0) {
        angle = 90;
      } else if (strcmp(status, "NO_FACE") == 0) {
        angle = 90;
      } else {
        angle = DEFAULT_ANGLE;
      }
      Serial.println("Using mapped angle from status (no angle field)");
    }
    
    handleMovementMessage(status, angle, confidence, timestamp, frame, target);
  }
  
  Serial.println("===========================================");
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    // Create a unique client ID
    String clientId = "ESP8266Servo-";
    clientId += String(random(0xffff), HEX);
    
    // Attempt to connect
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      
      // Subscribe to both topics
      if (client.subscribe(mqtt_topic_sub)) {
        Serial.print("✅ Subscribed to: ");
        Serial.println(mqtt_topic_sub);
      }
      
      if (client.subscribe(mqtt_topic_angle)) {
        Serial.print("✅ Subscribed to: ");
        Serial.println(mqtt_topic_angle);
      }
      
      // Publish online status
      String onlineMsg = "Servo controller online - Following face with exact angles";
      client.publish(mqtt_topic_pub, onlineMsg.c_str());
      
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(9600);
  
  // Initialize servo
  myServo.attach(servoPin, 500, 2400);
  myServo.write(DEFAULT_ANGLE);
  currentAngle = DEFAULT_ANGLE;
  
  Serial.println();
  Serial.println("===========================================");
  Serial.println("SERVO MQTT CONTROLLER - EXACT ANGLE MODE");
  Serial.println("===========================================");
  Serial.println("Receiving exact angles from face tracker:");
  Serial.println("  Format: {'status':'MOVE_RIGHT','angle':101,");
  Serial.println("           'confidence':0.86,'target':'liana',");
  Serial.println("           'timestamp':1772097324,'frame':969}");
  Serial.println("  Servo moves to EXACT angle specified");
  Serial.println();
  Serial.println("Also supports:");
  Serial.println("  - Direct angles to 'servo/angle' topic");
  Serial.println("  - Legacy status-only messages (mapped 0/90/180)");
  Serial.println("===========================================");
  
  // Setup WiFi
  setup_wifi();

  // Test servo movement
  moveServo(90, "Initialization");
  
  // Setup MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  client.setBufferSize(1024);  // Increased buffer for larger messages
  
  delay(1000);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  
  // Status update every 30 seconds
  unsigned long now = millis();
  if (now - lastMsgTime > 30000) {
    lastMsgTime = now;
    
    String statusMsg = "Current angle: " + String(currentAngle) + "° - Ready";
    client.publish(mqtt_topic_pub, statusMsg.c_str());
    Serial.print("📊 Status: ");
    Serial.println(statusMsg);
  }
}