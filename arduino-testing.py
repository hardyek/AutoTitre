from pyfirmata import Arduino,SERVO
from time import sleep

board = Arduino('COM3')

pin = 9 
board.digital[pin].mode=SERVO

print("Board initialised")

def rotateServo(pin,angle):
    board.digital[pin].write(angle)

def initServo(pin):
    rotateServo(pin,0)
    sleep(1)
    print("Reset (0°)")

def openServo(pin):
    rotateServo(pin,90)
    sleep(0.4)
    print("Opened (90°)")

def closeServo(pin):
    rotateServo(pin,0)
    sleep(1)
    print("Closed (0°)")

def moveServo(pin,angle):
    rotateServo(pin,angle)
    sleep(0.4)
    print(f"Moved ({angle}°)")

