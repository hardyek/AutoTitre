import cv2
import numpy as np

def detect_color(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def main():
    cap = cv2.VideoCapture(0)  # Change the number to the ID of your camera

    # Define color range for methyl orange
    lower_alkali = np.array([20, 100, 100])  # Lower bound of yellow in HSV format
    upper_alkali = np.array([30, 255, 255])  # Upper bound of yellow in HSV format

    lower_acid = np.array([0, 70, 180])
    upper_acid = np.array([15, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the image to focus on the conical flask region
        cropped_frame = frame[180:550, 150:550]  # Replace with coordin  ates of the flask region

        color_mask_acid = detect_color(cropped_frame, lower_acid, upper_acid)
        color_mask_alkali = detect_color(cropped_frame, lower_alkali, upper_alkali)

        # Calculate the percentage of the color in the cropped frame
        color_percentage_acid = (np.sum(color_mask_acid) / (color_mask_acid.shape[0] * color_mask_acid.shape[1] * 255)) * 100
        color_percentage_alkali = (np.sum(color_mask_alkali) / (color_mask_alkali.shape[0] * color_mask_alkali.shape[1] * 255)) * 100

        cv2.imshow('Color Mask Alkali', color_mask_alkali)
        cv2.imshow('Color Mask Acid', color_mask_acid)
        cv2.imshow('Frame', cropped_frame)

        if color_percentage_acid > color_percentage_alkali:
            print("Acid")
        else:
            print("Alkali")

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
