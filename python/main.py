import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time

import cv2
import mediapipe as mp
import numpy as np
import math

#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more.

#LEAP hand conventions:
#180 is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed.
#The MCP is centered at 180 and can move positive or negative to that.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

"""
########################################################
class LeapNode:
    def __init__(self):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.  
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        expected_port = 'COM6'
        self.dxl_client = DynamixelClient(motors, expected_port, 4000000)
        self.dxl_client.connect()

        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()
    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()
    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()
    
class Tracker:

    def __init__(self):
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

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




#init the node
def main(**kwargs):
    leap_hand = LeapNode()
    tracker = Tracker()
    cap = cv2.VideoCapture(0)

    with tracker.mp_hands.Hands(min_detection_confidence=0.7, 
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
                    tracker.mp_drawing.draw_landmarks(frame, hand_landmarks, tracker.mp_hands.HAND_CONNECTIONS)

                    # Get coordinates of landmarks
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                    # Calculate abduction angles for the index finger
                    index_abduction_1 = tracker.decompose_angle(landmarks[0], landmarks[5], landmarks[6], plane='xz')
                    index_flexion_1 = tracker.decompose_angle(landmarks[0], landmarks[5], landmarks[6], plane='yz')
                    index_flexion_2 = tracker.decompose_angle(landmarks[5], landmarks[6], landmarks[7], plane='xz')
                    index_flexion_3 = tracker.decompose_angle(landmarks[6], landmarks[7], landmarks[8], plane='xz')

                    # Append to joint_angles
                    joint_angles = [index_abduction_1+2*np.pi,index_flexion_1+2*np.pi,index_flexion_2+2*np.pi,index_flexion_3+2*np.pi]

                    print(f"Joint Angles: {joint_angles}")





                    pose = np.array([joint_angles[0],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    leap_hand.set_allegro(pose)



        # while True:


            


        #     pose = np.array([-0.785398,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        #     leap_hand.set_allegro(pose)

        #     #Set to an open pose and read the joint angles 33hz
        #     # leap_hand.set_allegro(np.zeros(16))
        #     # print("Position: " + str(leap_hand.read_pos()))
        #     time.sleep(0.03)

if __name__ == "__main__":
    main()
