import cv2

# Define the ROI for tracking (in this case, the first frame of the video)
x, y, w, h = 300, 200, 100, 100

# Set up the initial location of the window
track_window = (x, y, w, h)

# Set up the termination criteria for the mean shift algorithm
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)

# Read the first frame from the camera
ret, frame = cap.read()

# Set up the histogram of the ROI
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set up the output window
cv2.namedWindow('tracking')

# Loop through the frames of the video
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if ret:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the back projection of the histogram on the current frame
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Apply mean shift to the back projection to get the new location of the window
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw the tracking window on the frame
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        cv2.imshow('tracking', img2)

        # Check for key presses
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            # If the user presses the ESC key, exit the loop
            break
    else:
        break

# Release the VideoCapture object and close the output window
cap.release()
cv2.destroyAllWindows()