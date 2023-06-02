from pyfirmata import Arduino,SERVO
from time import sleep

class ServoClass():
    def __init__(self,COM,PIN):
        sleep(1)
        self.board = Arduino(COM)
        self.pin = PIN
        self.board.digital[self.pin].mode=SERVO
        self.angle = -1
        print("Servo initialised.")
    
    def Reset(self):
        self.board.digital[self.pin].write(0)
        sleep(1)
        self.angle = 0
        print(f"🔧 Servo Reset ({self.angle}°)")

    def Close(self):
        self.board.digital[self.pin].write(0)
        sleep(0.4)
        self.angle = 0
        print(f"🔧 Servo Closed ({self.angle}°)")

    def Open(self):
        self.board.digital[self.pin].write(95)
        sleep(0.4)
        self.angle = 95
        print(f"🔧 Servo Open ({self.angle}°)")

    def Move(self,angle):
        self.board.digital[self.pin].write(angle)
        sleep(0.4)
        self.angle = angle
        print(f"🔧 Servo Moved ({self.angle}°)")