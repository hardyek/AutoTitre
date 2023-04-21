import cv2
import uuid

# Create a video capture object to access the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Unable to access webcam")
    exit()

# Specify the file path to save the images
file_path = "C://Users//hardy//titration-ML//Indicator-Network//workspace//images//collected_images_2//_alkali//"

# Initialize a counter to keep track of the number of images captured
counter = 0

# Capture images from the webcam until the user interrupts the program
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error capturing frame")
        break

    # Display the captured frame in a window
    cv2.imshow("Frame", frame)

    # Wait for a key press for 1 millisecond
    key = cv2.waitKey(1)

    # If the user presses the 's' key, save the current frame to a file
    if key == ord('s'):
        # Construct the file name for the image
        file_name = f"image_{uuid.uuid1()}.jpg"

        # Save the image to the specified file location
        cv2.imwrite(file_path + file_name, frame)

        # Increment the counter
        counter += 1

        # Print a message to the console indicating that the image was saved
        print(f"Image saved as {file_name}")

    elif key == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
