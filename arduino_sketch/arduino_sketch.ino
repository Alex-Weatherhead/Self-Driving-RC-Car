#include <Wire.h>
#include <Servo.h> 
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 

Adafruit_DCMotor *motor1 = AFMS.getMotor(1);
Adafruit_DCMotor *motor2 = AFMS.getMotor(2);

Servo servo;

bool flag = true;

void setup() {

    AFMS.begin();
    servo.attach(10);
    Serial.begin(9600);   

    motor1->run(FORWARD);
    motor2->run(FORWARD);
     
}

void loop() {

    if (flag){
      
        if (Serial.available() >= 6){
            Serial.println();
            unsigned char angle_command[6];
            for (int i = 0; i < 6; i ++) {
                angle_command[i] = Serial.read();
                Serial.println(angle_command[i]);
            }
            float angle_ = atof(angle_command);
            Serial.println(angle_);
            flag = false;
            servo.write(angle_);
        }    
    
    }
    else{

        if (Serial.available() >= 3){
            Serial.println();
            unsigned char speed_command[3];
            for (int i = 0; i < 3; i ++) {
                speed_command[i] = Serial.read();
                Serial.println(speed_command[i]);
            }
            int speed_ = atoi(speed_command);
            Serial.println(speed_);
            flag = true;
            motor1->setSpeed(speed_);
            motor2->setSpeed(speed_);
        }    
            
    }
    
}
