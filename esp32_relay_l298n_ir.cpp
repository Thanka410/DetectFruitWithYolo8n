#include <Arduino.h>
#include <ESP32Servo.h>

const int RELAY_PIN = 23;

const int MOTOR_ENA_PIN = 25;
const int MOTOR_IN1_PIN = 26;
const int MOTOR_IN2_PIN = 27;

const int IR_PIN = 32;
const int SERVO_PIN = 18;

const bool RELAY_ACTIVE_LOW = false;
const bool IR_ACTIVE_LOW = true;

const int SERVO_HOME_ANGLE = 180;
const int SERVO_APPLE_ANGLE = 55;
const int SERVO_ORANGE_ANGLE = 135;

const unsigned long WAIT_AFTER_IR_MS = 1500;
const unsigned long SERVO_MOVE_MS = 500;
const unsigned long SERVO_HOLD_MS = 2000;

const int FRUIT_QUEUE_SIZE = 12;
const unsigned long IR_DEBOUNCE_MS = 300;
const unsigned long IR_STABLE_MS = 120;

Servo sorterServo;

String fruitQueue[FRUIT_QUEUE_SIZE];
int queueHead = 0;
int queueTail = 0;
int queueCount = 0;

int motorSpeed = 180;
bool motorRunning = false;
unsigned long lastIrEventMs = 0;
bool irRawActive = false;
bool irStableActive = false;
bool lastIrStableActive = false;
unsigned long irRawChangedAtMs = 0;

enum SortState {
  SORT_IDLE,
  SORT_WAIT_AFTER_IR,
  SORT_MOVING_TO_GATE,
  SORT_HOLDING,
  SORT_RETURNING
};

SortState sortState = SORT_IDLE;
String activeFruit = "";
int activeGateAngle = SERVO_HOME_ANGLE;
unsigned long stateStartedAtMs = 0;

void setRelay(bool on) {
  if (RELAY_ACTIVE_LOW) {
    digitalWrite(RELAY_PIN, on ? LOW : HIGH);
  } else {
    digitalWrite(RELAY_PIN, on ? HIGH : LOW);
  }
}

void applyMotor() {
  if (!motorRunning || motorSpeed <= 0) {
    analogWrite(MOTOR_ENA_PIN, 0);
    digitalWrite(MOTOR_IN1_PIN, LOW);
    digitalWrite(MOTOR_IN2_PIN, LOW);
    return;
  }

  digitalWrite(MOTOR_IN1_PIN, HIGH);
  digitalWrite(MOTOR_IN2_PIN, LOW);
  analogWrite(MOTOR_ENA_PIN, motorSpeed);
}

bool isValidFruit(const String &fruit) {
  return fruit == "apple" || fruit == "banana" || fruit == "orange";
}

bool enqueueFruit(const String &fruit) {
  if (queueCount >= FRUIT_QUEUE_SIZE) {
    return false;
  }

  fruitQueue[queueTail] = fruit;
  queueTail = (queueTail + 1) % FRUIT_QUEUE_SIZE;
  queueCount++;
  return true;
}

bool peekFruit(String &fruit) {
  if (queueCount <= 0) {
    return false;
  }

  fruit = fruitQueue[queueHead];
  return true;
}

bool removeFirstFruit(String &fruit) {
  if (queueCount <= 0) {
    return false;
  }

  fruit = fruitQueue[queueHead];
  queueHead = (queueHead + 1) % FRUIT_QUEUE_SIZE;
  queueCount--;
  return true;
}

void clearFruitQueue() {
  queueHead = 0;
  queueTail = 0;
  queueCount = 0;
}

void printQueueStatus() {
  Serial.print("QUEUE:");
  Serial.println(queueCount);
}

void handleIrTrigger() {
  if (sortState != SORT_IDLE) {
    Serial.println("IR:IGNORED:SERVO_BUSY");
    return;
  }

  String fruit;
  if (!peekFruit(fruit)) {
    Serial.println("IR:QUEUE_EMPTY");
    return;
  }

  Serial.print("IR:TRIGGER:");
  Serial.println(fruit);

  if (fruit == "apple") {
    activeGateAngle = SERVO_APPLE_ANGLE;
  } else if (fruit == "orange") {
    activeGateAngle = SERVO_ORANGE_ANGLE;
  } else {
    String removed;
    removeFirstFruit(removed);
    Serial.println("SORT:NO_GATE:banana");
    Serial.println("SORT:DONE:banana");
    printQueueStatus();
    return;
  }

  activeFruit = fruit;
  sortState = SORT_WAIT_AFTER_IR;
  stateStartedAtMs = millis();
  Serial.print("SORT:WAIT:");
  Serial.println(activeFruit);
}

void updateSortState() {
  unsigned long now = millis();

  if (sortState == SORT_IDLE) {
    return;
  }

  if (sortState == SORT_WAIT_AFTER_IR && now - stateStartedAtMs >= WAIT_AFTER_IR_MS) {
    sorterServo.write(activeGateAngle);
    sortState = SORT_MOVING_TO_GATE;
    stateStartedAtMs = now;
    Serial.print("SORT:GATE:");
    Serial.print(activeFruit);
    Serial.print(":");
    Serial.println(activeGateAngle);
    return;
  }

  if (sortState == SORT_MOVING_TO_GATE && now - stateStartedAtMs >= SERVO_MOVE_MS) {
    sortState = SORT_HOLDING;
    stateStartedAtMs = now;
    Serial.print("SORT:HOLD:");
    Serial.println(activeFruit);
    return;
  }

  if (sortState == SORT_HOLDING && now - stateStartedAtMs >= SERVO_HOLD_MS) {
    sorterServo.write(SERVO_HOME_ANGLE);
    sortState = SORT_RETURNING;
    stateStartedAtMs = now;
    Serial.print("SORT:RETURN:");
    Serial.println(activeFruit);
    return;
  }

  if (sortState == SORT_RETURNING && now - stateStartedAtMs >= SERVO_MOVE_MS) {
    String removed;
    removeFirstFruit(removed);
    Serial.print("SORT:DONE:");
    Serial.println(activeFruit);
    printQueueStatus();

    activeFruit = "";
    activeGateAngle = SERVO_HOME_ANGLE;
    sortState = SORT_IDLE;
  }
}

void handleCommand(String cmd) {
  cmd.trim();
  String original = cmd;
  cmd.toLowerCase();

  Serial.print("RECEIVED:");
  Serial.println(original);

  if (cmd == "ping") {
    Serial.println("OK:PONG");
    return;
  }

  if (cmd == "relay:on") {
    setRelay(true);
    motorRunning = true;
    applyMotor();
    Serial.println("OK:RELAY:ON");
    return;
  }

  if (cmd == "relay:off") {
    setRelay(false);
    motorRunning = false;
    applyMotor();
    Serial.println("OK:RELAY:OFF");
    return;
  }

  if (cmd == "motor:stop") {
    motorRunning = false;
    applyMotor();
    Serial.println("OK:MOTOR:STOP");
    return;
  }

  if (cmd == "motor:start") {
    motorRunning = true;
    applyMotor();
    Serial.println("OK:MOTOR:START");
    return;
  }

  if (cmd.startsWith("motor:speed:")) {
    int speed = cmd.substring(12).toInt();
    motorSpeed = constrain(speed, 0, 255);
    applyMotor();
    Serial.print("OK:MOTOR:SPEED:");
    Serial.println(motorSpeed);
    return;
  }

  if (cmd.startsWith("fruit:")) {
    String fruit = cmd.substring(6);
    fruit.trim();

    if (!isValidFruit(fruit)) {
      Serial.println("ERR:UNKNOWN_FRUIT");
      return;
    }

    if (!enqueueFruit(fruit)) {
      Serial.println("ERR:QUEUE_FULL");
      return;
    }

    Serial.print("OK:FRUIT:QUEUED:");
    Serial.println(fruit);
    printQueueStatus();

    // Queue moi khong duoc phep kich servo neu IR dang bi giu active.
    // Bat buoc cam bien phai nha ra roi kich hoat lai.
    lastIrStableActive = irStableActive;
    return;
  }

  if (cmd == "queue:clear") {
    clearFruitQueue();
    sorterServo.write(SERVO_HOME_ANGLE);
    activeFruit = "";
    activeGateAngle = SERVO_HOME_ANGLE;
    sortState = SORT_IDLE;
    Serial.println("OK:QUEUE:CLEAR");
    printQueueStatus();
    return;
  }

  if (cmd == "ir:test") {
    Serial.println("IR:TEST");
    handleIrTrigger();
    return;
  }

  if (cmd == "servo:home") {
    sorterServo.write(SERVO_HOME_ANGLE);
    Serial.println("OK:SERVO:HOME");
    return;
  }

  Serial.println("ERR:UNKNOWN_COMMAND");
}

bool readIrActive() {
  int value = digitalRead(IR_PIN);
  return IR_ACTIVE_LOW ? value == LOW : value == HIGH;
}

void initIrState() {
  bool active = readIrActive();
  irRawActive = active;
  irStableActive = active;
  lastIrStableActive = active;
  irRawChangedAtMs = millis();

  Serial.print("IR INIT:");
  Serial.println(active ? "ACTIVE" : "INACTIVE");
}

void checkIrSensor() {
  bool rawActive = readIrActive();
  unsigned long now = millis();

  if (rawActive != irRawActive) {
    irRawActive = rawActive;
    irRawChangedAtMs = now;
  }

  if (now - irRawChangedAtMs < IR_STABLE_MS) {
    return;
  }

  irStableActive = irRawActive;

  if (irStableActive && !lastIrStableActive && now - lastIrEventMs >= IR_DEBOUNCE_MS) {
    lastIrEventMs = now;
    handleIrTrigger();
  }

  lastIrStableActive = irStableActive;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(MOTOR_ENA_PIN, OUTPUT);
  pinMode(MOTOR_IN1_PIN, OUTPUT);
  pinMode(MOTOR_IN2_PIN, OUTPUT);
  pinMode(IR_PIN, INPUT_PULLUP);

  sorterServo.setPeriodHertz(50);
  sorterServo.attach(SERVO_PIN, 500, 2400);
  sorterServo.write(SERVO_HOME_ANGLE);
  initIrState();

  setRelay(false);
  applyMotor();

  Serial.println("ESP32 READY");
  Serial.println("SERVO READY");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }

  checkIrSensor();
  updateSortState();
}
