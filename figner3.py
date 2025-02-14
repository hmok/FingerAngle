import tkinter as tk
from tkinter import filedialog, scrolledtext
import os
import csv
import cv2
import mediapipe as mp
import math
import numpy as np

# poseiq uses MediaPipe internally:
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# List of official hand landmark names:
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP"
]

def calculate_angle(a, b, c):
    """
    Calculate angle ABC (in degrees) given three points:
    A, B, C in normalized (x,y) coordinates [0..1].
    
    Equation:
      angle = arccos( (AB · BC) / (|AB| * |BC|) )
    where:
      - AB and BC are the vectors from point B to A and B to C respectively.
      - The dot product and magnitudes are computed accordingly.
    """
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot_product = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    if mag_ab == 0 or mag_cb == 0:
        return 0.0
    angle_radians = math.acos(dot_product / (mag_ab * mag_cb))
    return math.degrees(angle_radians)

def process_video(video_path):
    """
    Process the video with poseiq.com (using MediaPipe) to:
      - Detect hand landmarks for each frame.
      - Compute the index finger joint angle (MCP-PIP-DIP).
      - Annotate the video with landmarks and angle text.
      - Save the per-frame landmark positions and joint angle to a CSV.
      - Save the annotated video.
    """
    # Get script folder (so we save outputs in the same directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths for output CSV and processed video
    csv_path = os.path.join(script_dir, "landmarks.csv")
    processed_video_path = os.path.join(script_dir, "processed_output.mp4")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video:", video_path)
        return

    # Prepare video writer (same size/fps as input)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    # Prepare CSV file with header columns for landmarks and index finger angle.
    header = ["frame"]
    for name in HAND_LANDMARKS:
        header += [f"{name}_x", f"{name}_y", f"{name}_z"]
    header.append("index_finger_angle")
    
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        # Use poseiq (MediaPipe) Hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            frame_index = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Default angle value
                index_angle = 0.0
                # Prepare row data: frame index, then 21 landmarks (x,y,z) and index angle.
                row_data = [frame_index] + [0]*(21*3) + [index_angle]

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        lm = hand_landmarks.landmark

                        # Store each landmark's (x,y,z) using the official landmark names.
                        for i, landmark_name in enumerate(HAND_LANDMARKS):
                            x, y, z = lm[i].x, lm[i].y, lm[i].z
                            base_idx = 1 + i*3
                            row_data[base_idx]   = x
                            row_data[base_idx+1] = y
                            row_data[base_idx+2] = z

                        # Compute index finger angle (using landmarks 5,6,7: MCP, PIP, DIP)
                        MCP = (lm[5].x, lm[5].y)
                        PIP = (lm[6].x, lm[6].y)
                        DIP = (lm[7].x, lm[7].y)
                        index_angle = calculate_angle(MCP, PIP, DIP)
                        row_data[-1] = index_angle

                        # Annotate frame with the computed angle near the PIP joint.
                        h, w, _ = frame.shape
                        px, py = int(PIP[0]*w), int(PIP[1]*h)
                        cv2.putText(frame, f"{index_angle:.1f} deg", 
                                    (px+10, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (255,0,0), 2)

                writer.writerow(row_data)
                out.write(frame)
                cv2.imshow("poseiq Hand Detection", frame)
                frame_index += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! CSV saved to: {csv_path}")
    print(f"Annotated video saved to: {processed_video_path}")

def show_help():
    """
    Open a new window with help information.
    """
    help_win = tk.Toplevel()
    help_win.title("Help - poseiq Finger Angle App")
    help_win.geometry("500x300")
    
    help_text = (
        "poseiq Finger Angle App\n"
        "-------------------------\n\n"
        "This application uses poseiq (built on top of MediaPipe) to process a video and "
        "detect hand landmarks in each frame. It computes the index finger joint angle using "
        "landmarks for the MCP, PIP, and DIP joints.\n\n"
        "Joint Angle Calculation:\n"
        "  angle = arccos( (AB · BC) / (|AB| * |BC|) )\n"
        "where AB and BC are vectors from the middle joint (PIP) to the adjacent joints.\n\n"
        "Outputs:\n"
        "  - A CSV file (landmarks.csv) with each frame's landmark positions (x, y, z) and the "
        "    index finger angle.\n"
        "  - An annotated video (processed_output.mp4) showing the detected landmarks and angle.\n\n"
        "Next Steps:\n"
        "  - Integrate real-time plotting of the angle data.\n"
        "  - Use additional gestures or multiple hands for further analysis.\n"
        "  - Extend the app to support real-time video capture.\n\n"
        "Press 'q' in the video window to exit early."
    )
    
    st = scrolledtext.ScrolledText(help_win, wrap=tk.WORD, width=60, height=15)
    st.insert(tk.END, help_text)
    st.configure(state='disabled')
    st.pack(padx=10, pady=10)

def select_video():
    """
    Opens a file dialog for selecting a video, then processes it.
    """
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm")]
    )
    if video_path:
        process_video(video_path)

# --- Tkinter GUI setup ---
root = tk.Tk()
root.title("poseiq Finger Angle App")
root.geometry("300x150")

select_btn = tk.Button(root, text="Select Video", command=select_video, width=20)
select_btn.pack(pady=10)

help_btn = tk.Button(root, text="Help", command=show_help, width=20)
help_btn.pack(pady=10)

root.mainloop()
