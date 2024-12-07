import cv2
import mediapipe as mp
import numpy as np
# import socket
import json
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Helper function to calculate angle between three points in 3D
# def calculate_angle(a, b, c):
#     ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
#     bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip for numerical stability
#     return angle
#     # return np.degrees(angle)

# Helper function to decompose angle into components
def decompose_angle(a, b, c, plane='xz'):
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    
    # Project vectors onto the specified plane
    if plane == 'xz':  # Abduction/Adduction
        ba_proj = np.array([ba[0], 0, ba[2]])
        bc_proj = np.array([bc[0], 0, bc[2]])
    elif plane == 'xy':  # Secondary Plane
        ba_proj = np.array([ba[0], ba[1], 0])
        bc_proj = np.array([bc[0], bc[1], 0])
    elif plane == 'yz':  # Flexion/Extension
        ba_proj = np.array([0, ba[1], ba[2]])
        bc_proj = np.array([0, bc[1], bc[2]])
    else:
        raise ValueError("Unsupported plane. Choose 'xz', 'xy', or 'yz'.")
    
    # Calculate angle in the projected plane
    cosine_angle = np.dot(ba_proj, bc_proj) / (np.linalg.norm(ba_proj) * np.linalg.norm(bc_proj))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip for numerical stability
    return -angle - math.pi

# # Function to send joint angles via socket
# def send_joint_angles(sock, joint_angles):
#     try:
#         data = json.dumps(joint_angles)  # Convert list to JSON string
#         sock.sendall(data.encode('utf-8'))  # Send data
#     except Exception as e:
#         print(f"[ERROR] Failed to send data: {e}")

# Set up webcam
cap = cv2.VideoCapture(0)

# # Configure socket connection
# host = '127.0.0.1'
# port = 65432
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect((host, port))
# print("[INFO] Connected to the server")

# Configure MediaPipe Hands to detect only one hand
with mp_hands.Hands(min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7, 
                    max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert the frame color
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        result = hands.process(image_rgb)

        joint_angles = []

        # Check if hand landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of landmarks
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Calculate abduction angles for the index finger
                index_abduction_1 = decompose_angle(landmarks[0], landmarks[5], landmarks[6], plane='xz')
                index_flexion_1 = decompose_angle(landmarks[0], landmarks[5], landmarks[6], plane='yz')
                index_flexion_2 = decompose_angle(landmarks[5], landmarks[6], landmarks[7], plane='xz')
                index_flexion_3 = decompose_angle(landmarks[6], landmarks[7], landmarks[8], plane='xz')

                # Append to joint_angles
                joint_angles = [index_abduction_1+2*np.pi,index_flexion_1+2*np.pi,index_flexion_2+2*np.pi,index_flexion_3+2*np.pi]

        

                print(f"Joint Angles: {joint_angles}")

        # # Send joint angles to the server
        # if joint_angles:
        #     send_joint_angles(sock, joint_angles)

        # Show the frame
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
# sock.close()
cv2.destroyAllWindows()
