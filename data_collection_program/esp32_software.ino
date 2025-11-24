#include <Wire.h>

// PIR CLICK 보드의 I2C 주소
const int I2C_ADDRESS = 0x4D;

void setup() {
  Serial.begin(115200);
  Wire.begin();
}

void loop() {
  Wire.requestFrom(I2C_ADDRESS, 2);
  if (Wire.available() == 2) {
    unsigned int sensorValue = (Wire.read() << 8) | Wire.read();
    Serial.println(sensorValue);
  }
  
  // 50Hz 샘플링
  delay(20); 
}