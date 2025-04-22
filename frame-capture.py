import cv2

# Load the video
cap = cv2.VideoCapture('./resources/car-traffic.mp4')


seconds = 13
cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000) 

# Read first frame
ret1, frame1 = cap.read()

# Go forward one second
cap.set(cv2.CAP_PROP_POS_MSEC, (seconds+1) * 1000) 

# Read second frame
ret2, frame2 = cap.read()

# Make sure both were read successfully
if not (ret1 and ret2):
    print("Could not read frames")
else:
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Show for confirmation
    while True:
        cv2.imshow("Frame 1", gray1)
        cv2.imshow("Frame 2", gray2)

        # Wait for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imwrite('./resources/frame1.png', gray1)
    cv2.imwrite('./resources/frame2.png', gray2)

cap.release()
cv2.destroyAllWindows()