import cv2
import sys
import traceback
import time

# Force Python to flush output immediately
sys.stdout.reconfigure(line_buffering=True)

print("Testing webcam access... Press 'q' to quit.")

try:
    print("Attempting to open webcam with index 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam with index 0. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open webcam with index 1 either. Trying index 2...")
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                print("Error: Could not open webcam with index 2 either. Exiting...")
                sys.exit(1)
    print("Webcam opened successfully.")

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        print("Captured frame successfully.")
        cv2.imshow('Webcam Test', frame)
        print("Displayed frame.")

        loop_time = time.time() - start_time
        print(f"Loop time: {loop_time:.3f} seconds")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test stopped.")
except Exception as e:
    print("Error during webcam test:", str(e))
    traceback.print_exc()
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(1)