import cv2
import numpy as np
import glob
import time

# Callback function for trackbar changes
def update_hsv(value):
    pass

# Function to explore HSV values interactively
def explore_hsv():
    def on_change(value):
        pass

    cv2.namedWindow('HSV Explorer')
    cv2.createTrackbar('Hue', 'HSV Explorer', 0, 255, on_change)
    cv2.createTrackbar('Saturation', 'HSV Explorer', 0, 255, on_change)
    cv2.createTrackbar('Value', 'HSV Explorer', 0, 255, on_change)

    while True:
        # Use the trackbar values for exploration
        h = cv2.getTrackbarPos('Hue', 'HSV Explorer')
        s = cv2.getTrackbarPos('Saturation', 'HSV Explorer')
        v = cv2.getTrackbarPos('Value', 'HSV Explorer')

        lower_green = np.array([h, s, v])
        upper_green = np.array([h + 40, 255, 255])

        # Print the current HSV range for reference
        print("Lower HSV:", lower_green)
        print("Upper HSV:", upper_green)

        # Simulate a green screen frame for testing
        frame = np.full((100, 100, 3), lower_green, dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        # Show the simulated green screen frame
        cv2.imshow('HSV Explorer', frame)

        key = cv2.waitKey(10)
        if key == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

def load_background_images(folder_path):
    # Load background images from the specified folder path
    background_images = []
    for i in range(len(glob.glob(folder_path + '/*.jpg'))):
        background_images.append(cv2.imread(background_folder_path + '/{}.jpg'.format(i)))
    # for filename in glob.glob(folder_path + '/*.jpg'):
    #     img = cv2.imread(filename)
    #     if img is not None:
    #         background_images.append(img)
    return background_images

def get_time_until_next_change(last_change_time, change_interval):
    time_elapsed = time.time() - last_change_time
    time_remaining = max(change_interval - time_elapsed, 0)
    return int(time_remaining)


# Function to process the frame and apply green screen effect
def process_frame(frame, background, x_position, y_position):

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # # Define the range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([100, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # The converted mask
    inverted_mask = cv2.bitwise_not(mask)

    # Debugging output to visualize the mask
    # cv2.imshow('Mask', mask)

    # Create a transparent overlay by combining the frame and the background with alpha
    overlay = frame.copy()
    # print (overlay.shape[0], overlay.shape[1])

    # Calculate the position to center the video on the screen
    y_offset = int((background.shape[0] - overlay.shape[0] - y_position)) #// 1.033
    x_offset = int((background.shape[1] - overlay.shape[1] - x_position)) #// 2.44

    overlay[mask != 0] = background[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]][mask != 0] # Set the green color in the frame to be transparent

    # Place the overlay on the background at the calculated position
    background[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

    return background




if __name__ == "__main__":
    # Specify the folder containing background images
    background_folder_path = "background_images"

    # # Create a named window with full-screen property
    # cv2.namedWindow('Video with Green Screen', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Video with Green Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Load background images
    background_images = load_background_images(background_folder_path)

    if len(background_images) == 0:
        print(f"Error: No background images found in the folder {background_folder_path}.")
        exit()

    # Open video capture
    cap = cv2.VideoCapture(0)

    """
    Initial Setup
    """

    # Index for cycling through background images
    background_index = 0  # 0 for the normal telescope and 4 for the ELT

    # The Setup below is for Sutti's Lenovo laptop
    # Set the set of desired scale factors for resizing the video frames based on the background image
    #               [Telescope, Newall, Angel,  Cathedral, ISS,    JWST,   ELT]
    scale_factors = np.array([0.889,     0.247,  0.0791, 0.075,     0.666,  0.137,  0.0296])*0.7

    # Set the offsets along the y-axis
    #               [Telescope, Newall, Angel,  Cathedral, ISS,    JWST,   ELT]
    x_offsets =     [392,       433,    610,    532,       519,    562,    712]
    y_offsets =     [14,        100,    102,    128,       220,    310,    49]


    # Time interval for changing background images (in seconds)
    change_interval = 15

    # Time variable for tracking the last background change
    last_change_time = time.time()

    while True:
        # Capture video frame
        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Check if it's time to change the background image
        if time.time() - last_change_time >= change_interval:
            # Increment background index and reset if it exceeds the number of images
            background_index = (background_index + 1) % len(background_images)
            # Update the time of the last background change
            last_change_time = time.time()

        # Resize the video frame while keeping the background fixed
        resized_frame = cv2.resize(frame, None, fx=scale_factors[background_index], fy=scale_factors[background_index])

        # Process the resized frame with the transparent overlay and get the mask
        result = process_frame(resized_frame, background_images[background_index].copy(), x_offsets[background_index], y_offsets[background_index])


        # Get the time remaining until the next background change
        time_remaining = get_time_until_next_change(last_change_time, change_interval)

        # Add countdown text to the frame
        countdown_text = f"Next Image in {time_remaining} s"
        cv2.putText(result, countdown_text, (1020, 740), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 38, 100), 2)

        # Display the result in fullscreen
        cv2.imshow('Video with Green Screen', result)



        # Check for manual background change
        key = cv2.waitKey(1) & 0xFF
        if ord('0') <= key <= ord('9'):
            index = key - ord('0')
            if index < len(background_images):
                background_index = index
            else:
                background_index = index % len(background_images)

        # Check for 'q' key
        if key == ord('q'):
            break


    # Release video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()




    # GreatNorth Museum Setup Specification
    # #               [Telescope, Newall, Angel,  Cathedral, ISS,    JWST,   ELT]
    # scale_factors = np.array([0.889,     0.247,  0.0791, 0.075,     0.666,  0.137,  0.0296])*0.5

    # # Set the offsets along the y-axis
    # #               [Telescope, Newall, Angel,  Cathedral, ISS,    JWST,   ELT]
    # x_offsets =     [279,       561,    854,    890,       501,    834,    989]
    # y_offsets =     [25,        128,    145,    112,       290,    386,    61]



    # # Read the initial background image
    # initial_background_path = "angel.jpg"
    # initial_background = cv2.imread(initial_background_path)
    # if initial_background is None:
    #     print(f"Error: Unable to read the initial background image at path {initial_background_path}.")
    #     exit()
    #
    # # Open video capture
    # cap = cv2.VideoCapture(0)
    #
    # # Set the desired scale factor for resizing the video frames
    # scale_factor = 0.2  # Adjust as needed
    # # # Create a named window with full-screen property
    # # cv2.namedWindow('Video with Green Screen', cv2.WND_PROP_FULLSCREEN)
    # # cv2.setWindowProperty('Video with Green Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #
    #
    # while True:
    #     # Capture video frame
    #     ret, frame = cap.read()
    #
    #     # Flip the frame horizontally
    #     frame = cv2.flip(frame, 1)
    #
    #     # Resize the video frame while keeping the background fixed
    #     resized_frame = cv2.resize(frame, None, fx = scale_factor, fy = scale_factor)
    #
    #     # Get the dimensions of the resized frame
    #     h, w, _ = resized_frame.shape
    #
    #     # Resize the video frame while keeping the background fixed
    #     resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    #
    #     # Process the frame with the green screen effect
    #     processed_frame = process_frame(resized_frame, initial_background.copy())
    #
    #     # Display the result
    #     cv2.imshow('Video with Green Screen', processed_frame)
    #
    #
    #     # Check for key events
    #     key = cv2.waitKey(1) & 0xFF
    #
    #     # Change the background image on 'b' key press
    #     if key == ord('1'):
    #         new_background_path = "angel.jpg"
    #         new_background = cv2.imread(new_background_path)
    #         scale_factor = 0.2
    #         if new_background is not None:
    #             initial_background = new_background
    #         else:
    #             print(f"Error: Unable to read the new background image at path {new_background_path}.")
    #
    #     if key == ord('2'):
    #         new_background_path = "vlt.jpg"
    #         new_background = cv2.imread(new_background_path)
    #         scale_factor = 0.07
    #         if new_background is not None:
    #             initial_background = new_background
    #         else:
    #             print(f"Error: Unable to read the new background image at path {new_background_path}.")
    #
    #     if key == ord('3'):
    #         new_background_path = "durham.jpg"
    #         new_background = cv2.imread(new_background_path)
    #         scale_factor = 0.2
    #         if new_background is not None:
    #             initial_background = new_background
    #         else:
    #             print(f"Error: Unable to read the new background image at path {new_background_path}.")
    #
    #     if key == ord('4'):
    #         new_background_path = "telescope.jpg"
    #         new_background = cv2.imread(new_background_path)
    #         scale_factor = 0.6
    #         if new_background is not None:
    #             initial_background = new_background
    #         else:
    #             print(f"Error: Unable to read the new background image at path {new_background_path}.")
    #
    #     # Break the loop if 'q' key is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # # Release video capture object and close all windows
    # cap.release()
    # cv2.destroyAllWindows()
