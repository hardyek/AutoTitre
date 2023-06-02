import os
import cv2
import numpy as np
import time
import datetime
import threading

# region External Classes / Functions
from servo_class import ServoClass

lower_alkali = np.array([20, 100, 100])
upper_alkali = np.array([30, 255, 255])

lower_acid = np.array([0, 70, 180])
upper_acid = np.array([15, 255, 255])

y1 = 180
y2 = 550
x1 = 150
x2 = 550


def detect_color(frame, lower_colour, upper_colour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    return mask

def seconds_to_mmss(seconds):  # sourcery skip: remove-unnecessary-cast
    minutes = seconds // 60
    seconds %= 60
    return f"{int(minutes):02d}:{int(seconds):02d}"

print("Imports + Functions")

Servo = ServoClass("COM3",9)
Servo.Reset()

print("Servo")


Webcam = cv2.VideoCapture(0) # OBS Command : VideoCapture('udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000',cv2.CAP_FFMPEG)
font = cv2.FONT_HERSHEY_SIMPLEX

print("Webcam")
# endregion
# region Session Setup
os.system('cls')
print(f"Setup Complete {datetime.datetime.now().strftime('%H:%M:%S')}")
session_start_time = round(time.time(),2)
# endregion

def display():
    while Webcam.isOpened():
        ret, frame = Webcam.read()

        if not ret:
            break

        cropped_frame = frame[180:550, 150:550]

        color_mask_acid = detect_color(cropped_frame, lower_acid, upper_acid)
        color_mask_alkali = detect_color(cropped_frame, lower_alkali, upper_alkali)

        cv2.imshow('Color Mask Alkali', color_mask_alkali)
        cv2.imshow('Color Mask Acid', color_mask_acid)
        cv2.imshow('Frame', cropped_frame)

        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            Webcam.release()
            break

def process():
    #Trial
    _ = input("Press any key to start the trial...")

    os.system('cls')
    trial_start_time = time.time()

    print("Beginning trial...")

    Trial = True
    Servo.Open()
    while Trial:
        ret, frame = Webcam.read()
        if not ret:
            break

        # Crop the image to focus on the conical flask region
        cropped_frame = frame[x1:x2,y1:y2]  # Replace with coordin  ates of the flask region

        color_mask_acid = detect_color(cropped_frame, lower_acid, upper_acid)
        color_mask_alkali = detect_color(cropped_frame, lower_alkali, upper_alkali)

        # Calculate the percentage of the color in the cropped frame
        color_percentage_acid = (np.sum(color_mask_acid) / (color_mask_acid.shape[0] * color_mask_acid.shape[1] * 255)) * 100
        color_percentage_alkali = (np.sum(color_mask_alkali) / (color_mask_alkali.shape[0] * color_mask_alkali.shape[1] * 255)) * 100

        if color_percentage_acid > color_percentage_alkali:
            Trial = False
            Servo.Close()
            trial_end_time = time.time()

    trial_open_time = trial_end_time - trial_start_time
    pre_drop_time = trial_open_time - 2

    print("Trial is over.")
    print(f"Burette was open for: {trial_open_time}")

    _ = input("Press any key to continue...")

    os.system('cls')

    #Main Titrations
    global titration_count
    titration_count = 0
    global titration_lengths
    titration_lengths = []
    print("Ready to titrate.")

    MainLoop = True
    while MainLoop:
        print(f"Pre-drop time: {pre_drop_time}")
        titration_start_time = time.time()
        elapsed_time = 0
        Servo.Open()
        while elapsed_time <= pre_drop_time:
            elapsed_time = time.time() - titration_start_time

        print(f"Time elapsed {elapsed_time}.")
        print("Switching to drops...")
        Servo.Move(60)
        Dropping = True

        while Dropping:

            ret, frame = Webcam.read()
            if not ret:
                break

            # Crop the image to focus on the conical flask region
            cropped_frame = frame[x1:x2,y1:y2]  # Replace with coordin  ates of the flask region

            color_mask_acid = detect_color(cropped_frame, lower_acid, upper_acid)
            color_mask_alkali = detect_color(cropped_frame, lower_alkali, upper_alkali)

            # Calculate the percentage of the color in the cropped frame
            color_percentage_acid = (np.sum(color_mask_acid) / (color_mask_acid.shape[0] * color_mask_acid.shape[1] * 255)) * 100
            color_percentage_alkali = (np.sum(color_mask_alkali) / (color_mask_alkali.shape[0] * color_mask_alkali.shape[1] * 255)) * 100

            if color_percentage_acid > color_percentage_alkali:
                Servo.Close()
                Dropping = False

        total_elapsed_time = time.time() - titration_start_time
        os.system('cls')
        print("Titration over.")
        print(f"Total elapsed time: {total_elapsed_time}")

        titration_count += 1
        titration_lengths.append(round(total_elapsed_time,2))

        action = input("Start another titration or exit. [y/q]")

        if action == 'q':
            MainLoop = False

t1 = threading.Thread(target=process)
t2 = threading.Thread(target=display)

t1.start()
t2.start()

t1.join()
t2.join()

# region Ending Session
os.system('cls')
session_end_time = round(time.time(),2)

print("Session Over")
print(f"Session end time : {datetime.datetime.now().strftime('%H:%M:%S')}")
print(f"Session lasted : {seconds_to_mmss(session_end_time-session_start_time)}")
print(f"Titrations in this session : {titration_count}")
# endregion

# region Writing Session Statistics to logs.txt file
with open("logs.txt", "w") as file:
    for element in titration_lengths:
        file.write(f"{str(element)} \n")
# endregion

_ = input("Waiting for close...")
exit()

# OBS virtual camera ffmpeg script (ffmpeg install required)
# ffmpeg\bin\ffmpeg.exe -y -f dshow -thread_queue_size 4096 -hwaccel cuda -hwaccel_output_format cuda -i video="OBS Virtual Camera" -f rawvideo -c:v mjpeg -qscale:v 0 -r 10 "udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000"