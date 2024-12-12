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

        expected_port = 'COM7' # NEED TO CHANGE TO SCAN FOR OPEN SERIAL PORTS
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

    # #Receive LEAP pose and directly control the robot
    # def set_leap(self, pose):
    #     self.prev_pos = self.curr_pos
    #     self.curr_pos = np.array(pose)
    #     self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    
    
    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    # def set_ones(self, pose):
    #     pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
    #     self.prev_pos = self.curr_pos
    #     self.curr_pos = np.array(pose)
    #     self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    # #read position of the robot
    # def read_pos(self):
    #     return self.dxl_client.read_pos()
    # #read velocity
    # def read_vel(self):
    #     return self.dxl_client.read_vel()
    # #read current
    # def read_cur(self):
    #     return self.dxl_client.read_cur()
    # #These combined commands are faster FYI and return a list of data
    # def pos_vel(self):
    #     return self.dxl_client.read_pos_vel()
    # #These combined commands are faster FYI and return a list of data
    # def pos_vel_eff_srv(self):
    #     return self.dxl_client.read_pos_vel_cur()
    
class Tracker:

    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def decompose_angle(self, a, b, c, plane='xz'):
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





def angle_between(v1, v2):
    """Calculates the angle between two 3D vectors in radians."""

    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (mag_v1 * mag_v2)

    # Handle potential floating-point errors
    cos_theta = np.clip(cos_theta, -1, 1) 

    angle_rad = np.arccos(cos_theta)
    return angle_rad


def normal_from_points(p1, p2, p3):
    """Calculate the normal vector to the plane defined by three points."""

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)  # Normalize the vector



def project_vector_onto_plane(vector, plane_normal):
    """
    Projects a vector onto a plane defined by its normal vector.

    Args:
        vector (np.array): The vector to be projected.
        plane_normal (np.array): The normal vector of the plane.

    Returns:
        np.array: The projected vector.
    """

    # Normalize the plane normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Calculate the projection of the vector onto the normal vector
    projection_onto_normal = np.dot(vector, plane_normal) * plane_normal

    # Subtract the projection from the original vector to get the projection onto the plane
    projected_vector = vector - projection_onto_normal

    return projected_vector




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

                    # palm frame
                    # Define all landmarks as numpy arrays
                    wrist = np.array(landmarks[0])            # Wrist
                    thumb_cmc = np.array(landmarks[1])        # Thumb carpometacarpal joint
                    thumb_mcp = np.array(landmarks[2])        # Thumb metacarpophalangeal joint
                    thumb_ip = np.array(landmarks[3])         # Thumb interphalangeal joint
                    thumb_tip = np.array(landmarks[4])        # Thumb tip
                    index_mcp = np.array(landmarks[5])        # Index finger metacarpophalangeal joint
                    index_pip = np.array(landmarks[6])        # Index finger proximal interphalangeal joint
                    index_dip = np.array(landmarks[7])        # Index finger distal interphalangeal joint
                    index_tip = np.array(landmarks[8])        # Index finger tip
                    middle_mcp = np.array(landmarks[9])       # Middle finger metacarpophalangeal joint
                    middle_pip = np.array(landmarks[10])      # Middle finger proximal interphalangeal joint
                    middle_dip = np.array(landmarks[11])      # Middle finger distal interphalangeal joint
                    middle_tip = np.array(landmarks[12])      # Middle finger tip
                    ring_mcp = np.array(landmarks[13])        # Ring finger metacarpophalangeal joint
                    ring_pip = np.array(landmarks[14])        # Ring finger proximal interphalangeal joint
                    ring_dip = np.array(landmarks[15])        # Ring finger distal interphalangeal joint
                    ring_tip = np.array(landmarks[16])        # Ring finger tip
                    pinky_mcp = np.array(landmarks[17])       # Pinky finger metacarpophalangeal joint
                    pinky_pip = np.array(landmarks[18])       # Pinky finger proximal interphalangeal joint
                    pinky_dip = np.array(landmarks[19])       # Pinky finger distal interphalangeal joint
                    pinky_tip = np.array(landmarks[20])       # Pinky finger tip

                    
                    # vertical wrt up and down your hand. palm norm is out of the palm
                    vertical_vec = ring_mcp - wrist
                    horizontal_vec = index_mcp - pinky_mcp
                    palm_norm = normal_from_points(wrist, index_mcp, pinky_mcp)

                    # vector between mcp and pip. project it onto the palm plane for abduction
                    index_lower_vector = index_mcp - index_pip
                    index_middle_vector = index_pip - index_dip
                    index_upper_vector = index_dip - index_tip
                    vec_for_flexion_1 = project_vector_onto_plane(index_lower_vector, horizontal_vec)

                    # this gets the flexion angles
                    flexion_1_index = -angle_between(vec_for_flexion_1, vertical_vec) + np.pi
                    flexion_2_index = angle_between(index_lower_vector, index_middle_vector)
                    flexion_3_index = angle_between(index_middle_vector, index_upper_vector)

                    # Middle finger vectors
                    middle_lower_vector = middle_mcp - middle_pip
                    middle_middle_vector = middle_pip - middle_dip
                    middle_upper_vector = middle_dip - middle_tip
                    vec_for_flexion_1_middle = project_vector_onto_plane(middle_lower_vector, horizontal_vec)

                    # Middle finger flexion angles
                    flexion_1_middle = -angle_between(vec_for_flexion_1_middle, vertical_vec) + np.pi
                    flexion_2_middle = angle_between(middle_lower_vector, middle_middle_vector)
                    flexion_3_middle = angle_between(middle_middle_vector, middle_upper_vector)

                    # Ring finger vectors
                    ring_lower_vector = ring_mcp - ring_pip
                    ring_middle_vector = ring_pip - ring_dip
                    ring_upper_vector = ring_dip - ring_tip
                    vec_for_flexion_1_ring = project_vector_onto_plane(ring_lower_vector, horizontal_vec)

                    # Ring finger flexion angles
                    flexion_1_ring = -angle_between(vec_for_flexion_1_ring, vertical_vec) + np.pi
                    flexion_2_ring = angle_between(ring_lower_vector, ring_middle_vector)
                    flexion_3_ring = angle_between(ring_middle_vector, ring_upper_vector)

                    # Thumb vectors
                    thumb_lower_vector = thumb_mcp - thumb_cmc  # From carpometacarpal to metacarpophalangeal joint
                    thumb_middle_vector = thumb_mcp - thumb_ip  # From MCP to interphalangeal joint
                    thumb_upper_vector = thumb_ip - thumb_tip  # From IP to tip

                    # Projection for first flexion
                    vec_for_flexion_1_thumb = project_vector_onto_plane(thumb_lower_vector, palm_norm)

                    # Flexion Angles
                    flexion_1_thumb = angle_between(vec_for_flexion_1_thumb, vertical_vec)
                    flexion_2_thumb = angle_between(thumb_lower_vector, thumb_middle_vector)
                    flexion_3_thumb = angle_between(thumb_middle_vector, thumb_upper_vector)

                    # Abduction (projecting lower vector onto palm plane)
                    vec_for_abduction_thumb = project_vector_onto_plane(thumb_lower_vector, palm_norm)
                    abduction_thumb = angle_between(vec_for_abduction_thumb, horizontal_vec)

                    # # Calculate abduction angles for the index finger
                    index_abduction = tracker.decompose_angle(landmarks[0], landmarks[5], landmarks[6], plane='xy')
                    middle_abduction = tracker.decompose_angle(landmarks[0], landmarks[9], landmarks[10], plane='xy')
                    ring_abduction = tracker.decompose_angle(landmarks[0], landmarks[13], landmarks[14], plane='xy')

                    # print(flexion_1_thumb*1.5-1)


                    pose = np.array([index_abduction+2*np.pi - .4,flexion_1_index,flexion_2_index,flexion_3_index,
                                     middle_abduction+2*np.pi - .2,flexion_1_middle,flexion_2_middle,flexion_3_middle,
                                     ring_abduction+2*np.pi - .2,flexion_1_ring,flexion_2_ring,flexion_3_ring,
                                     -flexion_1_thumb+.7,flexion_1_thumb*1.5-1.3,-flexion_2_thumb*.8+3.6,flexion_3_thumb])#abduction_thumb, flexion_2_thumb,flexion_3_thumb,])

                    leap_hand.set_allegro(pose)
                    # try:
                    #     leap_hand.set_allegro(pose)
                    # except Exception as e:
                    #     print(f"YOU MIGHT WANNA DEAL WITH {e}")

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            time.sleep(.06)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
