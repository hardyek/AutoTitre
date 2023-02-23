from pyfirmata import Arduino,SERVO
from time import sleep

class ServoClass():
    def __init__(self,COM,PIN):
        self.board = Arduino(COM)
        self.pin = PIN
        self.board.digital[self.pin].mode=SERVO
        self.angle = -1
        print("✅ Servo initialised.")

    def Rotate(self,angle):
        self.board.digital[self.pin].write(angle)
    
    def Reset(self):
        self.Rotate(0)
        sleep(1)
        self.angle = 0
        print(f"Servo Reset ({self.angle}°)")

    def Open(self):
        self.Rotate(90)
        sleep(0.4)
        self.angle = 90
        print(f"Servo Opened ({self.angle}°)")

    def Close(self):
        self.Rotate(0)
        sleep(0.4)
        self.angle = 0
        print(f"Servo Closed ({self.angle}°)")

    def Move(self,angle):
        self.Rotate(angle)
        sleep(0.4)
        self.angle = angle
        print(f"Servo Moved ({self.angle}°)")