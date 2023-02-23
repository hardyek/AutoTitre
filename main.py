import os
import cv2
import numpy as np
import tensorflow as tf
import collections
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from servo_class import ServoClass

#Indicator Model Setup
model_dir = os.path.join('Indicator-Network','workspace','models','Indicator_Network','export','saved_model')
model = tf.saved_model.load(model_dir)
input_tensor = model.signatures['serving_default'].inputs
output_tensor = model.signatures['serving_default'].outputs

#Functions using the model
@tf.function
def predict(image):
    output = model(image)
    return output

@tf.function
def pre_process(image):
    image_tensor = tf.expand_dims(tf.convert_to_tensor(image), 0)
    return image_tensor

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

print("✅ Model loaded.")

#Servo Setup
#Servo = ServoClass("COM3",9)

#Webcam Setup
Webcam = cv2.VideoCapture('udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000',cv2.CAP_FFMPEG)
WIDTH = int(Webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(Webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Webcam")
print(f"✅ Webcam started. ({WIDTH}x{HEIGHT})")

#Constant and Function Setup
INITIAL_STATE_STEPS = 100
VAL_STATE_STEPS = 10

def mode(arr):
    counter = collections.Counter(arr)
    mode = counter.most_common(1)[0][0]
    return mode

def readable(class_num):
    if class_num == 3:
        return "Acid"
    if class_num == 4:
        return "Alkali"

label_map_dir = os.path.join('Indicator-Network','workspace','annotations','annotation_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(label_map_dir)

current_stage = "Pre"
end = False
first_step = True

print("🟦 Setup Complete")

#Infinite Loop
while not end:

    #Stage that waits untill the user presses 'n' / is ready to start.
    while current_stage == "Pre":
        if first_step:
            print("🟪 Entering Stage 0 - Waiting for Start")
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
    
        cv2.imshow('Webcam', cv2.resize(frame_np_with_detections,(WIDTH,HEIGHT)))
        
        if cv2.waitKey(1) & 0xFF == ord('n'):
            current_stage = "Titration"
            first_step = True
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            current_stage = "Close"
            break
    
    #Titration stage responsible for carrying out the titration
    while current_stage == "Titration":
        if first_step:
            print("🟨 Entering Stage 1 - Detecting initial state")
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
                print("🟧 Entering Stage 2 - Titration Stage")
                first_step = False
                #Servo.Open()

            if classes[0] != initial_state:
                #Servo.Close()
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

    #Validates for a duration of frames wether a change in state is valid.
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

            elif val_mode != val_state:
                current_stage = "Titration"
                print("🟧 Re-entering Titration Stage")

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

    #Stage where you can Close or Restart the program
    while current_stage == "Post":
        if first_step:
            print("🟩 Entering Stage 3 - Titration over")
            print("Press Q to quit, R to restart")

        first_step = False
        _, frame = Webcam.read()

        cv2.imshow('Webcam',frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            current_stage = "Close"

        if cv2.waitKey(1) & 0xFF == ord('r'):
            current_stage = "Pre"
            print("🟦 Restarting")
            first_step = True
    
    if current_stage == "Close":
        print("⬛ Closing")
        Webcam.release()
        cv2.destroyAllWindows()
        exit()

#ffmpeg\bin\ffmpeg.exe -y -f dshow -thread_queue_size 4096 -hwaccel cuda -hwaccel_output_format cuda -i video="OBS Virtual Camera" -f rawvideo -c:v mjpeg -qscale:v 0 -r 10 "udp://@239.1.1.7:5107?overrun_nonfatal=1&fifo_size=50000000"

