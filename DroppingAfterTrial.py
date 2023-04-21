# region Packages
import os
import cv2
import numpy as np
import tensorflow as tf
import time
import datetime
import collections
import threading
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow logging
# endregion

# region External Classes / Functions
from servo_class import ServoClass


# Functions for object detection
@tf.function
def predict(image):
    return model(image)

@tf.function
def pre_process(image):
    return tf.expand_dims(tf.convert_to_tensor(image), 0)

def post_process(output):
    num_detections = int(output.pop('num_detections'))
    output = {key: value[0, :num_detections].numpy() for key, value in output.items()}
    output['num_detections'] = num_detections

    boxes = output['detection_boxes']
    scores = output['detection_scores']
    classes = output['detection_classes']

    return boxes,scores,classes

def detect(image):
    image_tensor = pre_process(image)
    raw_detections = predict(image_tensor)
    boxes, scores, classes = post_process(raw_detections)
    return boxes, scores, classes

# Misc functions
def readable(class_num):
    if class_num == 3:
        return "Acid"
    if class_num == 4:
        return "Alkali"
    
def mode(arr):
    counter = collections.Counter(arr)
    return counter.most_common(1)[0][0]

def seconds_to_mmss(seconds):  # sourcery skip: remove-unnecessary-cast
    minutes = seconds // 60
    seconds %= 60
    return f"{int(minutes):02d}:{int(seconds):02d}"

def return_highest(class0,confidence):
    return 2 if class0 == 2 and confidence > .375 else 1

# endregion

# region Model Setup
model_dir = os.path.join('Indicator-Network','workspace','models','iter2','export','saved_model')
model = tf.saved_model.load(model_dir)

print("Model loaded.")
# endregion

# region Servo Setup
Servo = ServoClass("COM3",9)
Servo.Reset()
# endregion

# region Webcam Setup
WIDTH = 1280
HEIGHT = 720
Webcam = cv2.VideoCapture(0) # OBS Command : VideoCapture('udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000',cv2.CAP_FFMPEG)
Webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
Webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
font = cv2.FONT_HERSHEY_SIMPLEX
prev_frame_time = 0
print(f"Webcam started. ({WIDTH}x{HEIGHT})")
label_map_dir = os.path.join('Indicator-Network','workspace','annotations','annotation_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(label_map_dir)
# endregion

# region Variable Setup
INITIAL_STATE_STEPS = 100

print("Setup complete.")
time.sleep(3)
# endregion

# region Session Setup
os.system('cls')
print(f"Setup Complete {datetime.datetime.now().strftime('%H:%M:%S')}")
session_start_time = round(time.time(),2)
# endregion

def display():
    while Webcam.isOpened():
        _, frame = Webcam.read()

        frame_np = np.array(frame)
        boxes, scores, classes = detect(frame_np)

        classes = classes.astype(np.int64)

        frame_np_with_detections = frame_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            frame_np_with_detections,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.35,
            agnostic_mode=False)
        
        cv2.imshow('Webcam Feed', cv2.resize(frame_np_with_detections,(WIDTH,HEIGHT)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            Webcam.release()
            break

def process():
    #Pre
    Pre = True
    while Pre:
        _ = input("Press any key to start detecting initial state...")
        Pre = False

    #Initial 
    Initial = True
    step = 0
    initial_classes = []
    os.system('cls')
    print("Detecting initial state...")
    while Initial:
        while step < INITIAL_STATE_STEPS:
            _, frame = Webcam.read()

            frame_np = np.array(frame)
            _, scores, classes = detect(frame_np)

            classes = classes.astype(np.int64)

            classes[0] = return_highest(classes[0],scores[0])

            initial_classes.append(classes[0])
            step += 1
    
        initial_state = mode(initial_classes)
        readable_initial_state = readable(initial_state)
        print(f"🧪 After {step} frames, initial state measured as {readable_initial_state}")

        action = input("Is this correct? [y/n]")

        if action == 'y':
            Initial = False

        if action == 'n':
            os.system('cls')
            print("Detecting initial state...")
            step = 0

    #Trial
    os.system('cls')
    trial_start_time = time.time()
    print("Beginning trial...")
    Trial = True
    Servo.Open()
    while Trial:
        _, frame = Webcam.read()

        frame_np = np.array(frame)
        _, scores, classes = detect(frame_np)

        classes = classes.astype(np.int64)

        classes[0] = return_highest(classes[0],scores[0])

        if classes[0] != initial_state:
            trial_end_time = time.time()
            Servo.Close()
            Trial = False

    trial_open_time = trial_end_time - trial_start_time
    pre_drop_time = trial_open_time - 1
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
            _, frame = Webcam.read()

            frame_np = np.array(frame)
            _, scores, classes = detect(frame_np)

            classes = classes.astype(np.int64)

            classes[0] = return_highest(classes[0],scores[0])

            elapsed_time = time.time() - titration_start_time

        print(f"Time elapsed {elapsed_time}.")
        print("Switching to drops...")
        Servo.Move(50)
        while classes[0] == initial_state:
            _, frame = Webcam.read()

            frame_np = np.array(frame)
            _, scores, classes = detect(frame_np)

            classes = classes.astype(np.int64)

            classes[0] = return_highest(classes[0],scores[0])

        Servo.Close()
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

print("⬛ Session Over")
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