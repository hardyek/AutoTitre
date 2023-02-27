# region Packages
import os
import cv2
import numpy as np
import tensorflow as tf
import time
import datetime
import collections
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

# endregion

# region Model Setup
model_dir = os.path.join('Indicator-Network','workspace','models','Indicator_Network','export','saved_model')
model = tf.saved_model.load(model_dir)
input_tensor = model.signatures['serving_default'].inputs
output_tensor = model.signatures['serving_default'].outputs

print("Model loaded.")
# endregion

# region Servo Setup
Servo = ServoClass("COM3",9)
Servo.Reset()
# endregion

# region Webcam Setup
WIDTH = 1400
HEIGHT = 1080
Webcam = cv2.VideoCapture('udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000',cv2.CAP_FFMPEG) # OBS Command : VideoCapture('udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000',cv2.CAP_FFMPEG)
Webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
Webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
font = cv2.FONT_HERSHEY_SIMPLEX
prev_frame_time = 0
cv2.namedWindow("Webcam")
print(f"Webcam started. ({WIDTH}x{HEIGHT})")
# endregion

# region Variable Setup
INITIAL_STATE_STEPS = 100
VAL_STATE_STEPS = 10

label_map_dir = os.path.join('Indicator-Network','workspace','annotations','annotation_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(label_map_dir)

current_stage = "Pre"
end = False
first_step = True

titration_count = 0
titration_lengths = []

print("Setup complete...  clearing terminal")
# endregion

# region Session Setup
os.system('cls')
print(f"Setup Complete {datetime.datetime.now().strftime('%H:%M:%S')}")
session_start_time = round(time.time(),2)
# endregion

# region Main Loop
while not end:

    # region "Pre" - waits for user to start the titration
    while current_stage == "Pre":
        if first_step:
            print("Entering Stage 0 - Waiting for Start")
            print("Press N to begin.")
            first_step = False

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
            min_score_thresh=.8,
            agnostic_mode=False)


        new_frame_time = round(time.time(),2)
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame_np_with_detections, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Webcam', cv2.resize(frame_np_with_detections,(WIDTH,HEIGHT)))

        if cv2.waitKey(1) & 0xFF == ord('n'):
            current_stage = "Titration"
            first_step = True
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            current_stage = "Close"
            break
    # endregion

    # region "Titration" - carries out the titration
    while current_stage == "Titration":
        if first_step:
            start_time = round(time.time(),2)
            print("Entering Stage 1 - Detecting initial state")
            init_step = 0
            frame_classes = []
            first_step = False

        _, frame = Webcam.read()

        frame_np = np.array(frame)
        boxes, scores, classes = detect(frame_np)

        classes = classes.astype(np.int64)

        #Detect the class initially
        if init_step < INITIAL_STATE_STEPS:
            frame_classes.append(classes[0])
            init_step += 1

        if init_step == INITIAL_STATE_STEPS:
            initial_state = mode(frame_classes)
            readable_initial_state = readable(initial_state)
            print(f"🧪 After {init_step} frames, initial state measured as {readable_initial_state}")
            init_step = 101
            first_step = True

        #Detect frame by frame wether their is change from the initial state.
        if init_step > INITIAL_STATE_STEPS:
            if first_step:
                print("Entering Stage 2 - Titration Stage")
                first_step = False
                Servo.Open()

            if classes[0] != initial_state:
                Servo.Close()
                print("🧪 Change detected.")
                val_state = classes[0]
                current_stage = "Val"
                first_step = True


        frame_np_with_detections = frame_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                frame_np_with_detections,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

        cv2.imshow('Webcam', cv2.resize(frame_np_with_detections,(WIDTH,HEIGHT)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            current_stage = "Close"
            break

        if cv2.waitKey(1) & 0xFF == ord('p'):
            current_stage = "Post"
            val_state = 'True'
            first_step = True
            break
    # endregion

    # region "Val" - validates whether the change detected was a true change
    while current_stage == "Val":
        if first_step:
            val_step = 0
            print("🧪 Validating change.")
            first_step = False
            val_states = []

        _, frame = Webcam.read()

        frame_np = np.array(frame)
        boxes, scores, classes = detect(frame_np)

        classes = classes.astype(np.int64)

        if val_step < VAL_STATE_STEPS:
            val_states.append(classes[0])
            val_step += 1

        if val_step == VAL_STATE_STEPS:
            val_mode = mode(val_states)
            if val_mode == val_state:
                current_stage = "Post"
                first_step = True
                print("🧪 Change Validated - Ending Titration")

            else:
                current_stage = "Titration"
                Servo.Open()
                os.system('cls')
                print("Re-entering Titration Stage")

        frame_np_with_detections = frame_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                frame_np_with_detections,
                boxes,
                classes,
                scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

        cv2.imshow('Webcam', cv2.resize(frame_np_with_detections,(WIDTH,HEIGHT)))
    # endregion

    # region "Post" - displays statistics about the titration and gives the user to close the program or start another titration
    while current_stage == "Post":

        end_time = round(time.time(),2)
        titration_count += 1
        titration_length = round(end_time - start_time,2)
        titration_lengths.append(titration_length)
        os.system('cls')
        print("Titration Over")
        print(f"Change Detected {readable(initial_state)} -> {readable(val_state)}")
        print(f"Titration took : {titration_length}s")

        _, frame = Webcam.read()
        cv2.imshow('Webcam',cv2.resize(frame,(WIDTH,HEIGHT)))


        action = input("Close (c) or Restart (r)...")
        if action in ["Close", "close", "c"]:
            current_stage = "Close"

        if action in ['Restart','restart','r']:
            current_stage = "Pre"
            Servo.Reset()
            os.system('cls')
            first_step = True
    # endregion

    if current_stage == "Close":
        break

# endregion

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
Webcam.release()
cv2.destroyAllWindows() 
exit()

#OBS virtual camera ffmpeg script
#ffmpeg\bin\ffmpeg.exe -y -f dshow -thread_queue_size 4096 -hwaccel cuda -hwaccel_output_format cuda -i video="OBS Virtual Camera" -f rawvideo -c:v mjpeg -qscale:v 0 -r 10 "udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000"