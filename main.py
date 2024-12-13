import numpy as np
import csv
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time

import cv2
import mediapipe as mp
import numpy as np
import math
import serial.tools.list_ports
from enum import Enum, auto

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LeapHand:
    def __init__(self):
        # Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        port_found = False
        while not port_found:

            try:
                port_found = self.find_ports(motors) # find port and connect
            except Exception as e:
                pass
            print("port not found... waiting")
            time.sleep(2)

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

        self.current_row = 0

    def find_ports(self, motors):
        """Find open serial ports and test. returns if a port was found"""

        ports = serial.tools.list_ports.comports()

        # scan all ports
        for port in [port.device for port in ports]: #this is only scaning the names
            try:
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                return True
                
            except Exception as e:
                pass

        return False

    def set_pose(self, pose):
        """set hand pose"""
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def get_pose(self):
        """get the current pose"""
        return self.dxl_client.read_pos()

    def csv_player(self, csv_path):
        """play from a CSV file"""
        try:
            with open(csv_path, "r") as csv_file:
                reader = list(csv.reader(csv_file))
                headers = reader[0]  # Skip the header row
                data = reader[1:]

                # playback
                row = data[self.current_row % len(data)]
                pose = [float(value) for value in row]
                self.set_pose(pose)
                self.current_row += 1

        except FileNotFoundError:
            print(f"Error: File {csv_path} not found.")

class Tracker:

    def __init__(self):
        """Init mediapipe"""
        self.recognition_result_list = []
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options,
                                          running_mode=vision.RunningMode.LIVE_STREAM,
                                          num_hands=1,
                                          min_hand_detection_confidence=.5,
                                          min_hand_presence_confidence=.5,
                                          min_tracking_confidence=.5,
                                          result_callback=self.save_result)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def save_result(self, result: vision.GestureRecognizerResult,
                  unused_output_image: mp.Image, timestamp_ms: int):

        self.recognition_result_list.append(result)

    def decompose_angle(self, a, b, c, plane='xz'):
        """decompose Angle"""
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

    def angle_between(self, v1, v2):
        """Calculates the angle between two 3D vectors in radians."""

        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)

        cos_theta = dot_product / (mag_v1 * mag_v2)

        # Handle potential floating-point errors
        cos_theta = np.clip(cos_theta, -1, 1) 

        angle_rad = np.arccos(cos_theta)
        return angle_rad

    def normal_from_points(self, p1, p2, p3):
        """Calculate the normal vector to the plane defined by three points."""

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)  # Normalize the vector

    def project_vector_onto_plane(self, vector, plane_normal):
        """Projects a vector onto a plane defind by its normal vector"""

        # Normalize the plane normal vector
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Calculate the projection of the vector onto the normal vector
        projection_onto_normal = np.dot(vector, plane_normal) * plane_normal

        # Subtract the projection from the original vector to get the projection onto the plane
        projected_vector = vector - projection_onto_normal

        return projected_vector

    def get_hand_pose(self, landmarks):
        """Get the hand pose"""

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
        palm_norm = self.normal_from_points(wrist, index_mcp, pinky_mcp)

        # vector between mcp and pip. project it onto the palm plane for abduction
        index_lower_vector = index_mcp - index_pip
        index_middle_vector = index_pip - index_dip
        index_upper_vector = index_dip - index_tip
        vec_for_flexion_1 = self.project_vector_onto_plane(index_lower_vector, horizontal_vec)

        # this gets the flexion angles
        flexion_1_index = -self.angle_between(vec_for_flexion_1, vertical_vec) + np.pi
        flexion_2_index = self.angle_between(index_lower_vector, index_middle_vector)
        flexion_3_index = self.angle_between(index_middle_vector, index_upper_vector)

        # Middle finger vectors
        middle_lower_vector = middle_mcp - middle_pip
        middle_middle_vector = middle_pip - middle_dip
        middle_upper_vector = middle_dip - middle_tip
        vec_for_flexion_1_middle = self.project_vector_onto_plane(middle_lower_vector, horizontal_vec)

        # Middle finger flexion angles
        flexion_1_middle = -self.angle_between(vec_for_flexion_1_middle, vertical_vec) + np.pi
        flexion_2_middle = self.angle_between(middle_lower_vector, middle_middle_vector)
        flexion_3_middle = self.angle_between(middle_middle_vector, middle_upper_vector)

        # Ring finger vectors
        ring_lower_vector = ring_mcp - ring_pip
        ring_middle_vector = ring_pip - ring_dip
        ring_upper_vector = ring_dip - ring_tip
        vec_for_flexion_1_ring = self.project_vector_onto_plane(ring_lower_vector, horizontal_vec)

        # Ring finger flexion angles
        flexion_1_ring = -self.angle_between(vec_for_flexion_1_ring, vertical_vec) + np.pi
        flexion_2_ring = self.angle_between(ring_lower_vector, ring_middle_vector)
        flexion_3_ring = self.angle_between(ring_middle_vector, ring_upper_vector)

        # Thumb vectors
        thumb_lower_vector = thumb_mcp - thumb_cmc  # From carpometacarpal to metacarpophalangeal joint
        thumb_middle_vector = thumb_mcp - thumb_ip  # From MCP to interphalangeal joint
        thumb_upper_vector = thumb_ip - thumb_tip  # From IP to tip

        # Projection for first flexion
        vec_for_flexion_1_thumb = self.project_vector_onto_plane(thumb_lower_vector, palm_norm)

        # Flexion Angles
        flexion_1_thumb = self.angle_between(vec_for_flexion_1_thumb, vertical_vec)
        flexion_2_thumb = self.angle_between(thumb_lower_vector, thumb_middle_vector)
        flexion_3_thumb = self.angle_between(thumb_middle_vector, thumb_upper_vector)

        # Abduction (projecting lower vector onto palm plane)
        vec_for_abduction_thumb = self.project_vector_onto_plane(thumb_lower_vector, palm_norm)
        abduction_thumb = self.angle_between(vec_for_abduction_thumb, horizontal_vec)

        # Calculate abduction angles 
        index_abduction = self.decompose_angle(landmarks[0], landmarks[5], landmarks[6], plane='xy')
        middle_abduction = self.decompose_angle(landmarks[0], landmarks[9], landmarks[10], plane='xy')
        ring_abduction = self.decompose_angle(landmarks[0], landmarks[13], landmarks[14], plane='xy')

        # full hand pose
        pose = np.array([index_abduction+2*np.pi - .4,flexion_1_index,flexion_2_index,flexion_3_index,
                            middle_abduction+2*np.pi - .2,flexion_1_middle,flexion_2_middle,flexion_3_middle,
                            ring_abduction+2*np.pi - .2,flexion_1_ring,flexion_2_ring,flexion_3_ring,
                            -flexion_1_thumb+.7,flexion_1_thumb*1.5-1.3,-flexion_2_thumb*.8+3.6,flexion_3_thumb])
        
        return pose

class Game:

    class GameState(Enum):
        """State of the Game"""
        ERROR = auto()
        MENU = auto()
        TELEOPERATION = auto()
        WALKING = auto()
        DANCE = auto()
        
    class Gestures(Enum):
        """duh"""
        NO_GESTURE = auto()
        OPEN_PALM = auto()
        VICTORY = auto()
        CLOSED_FIST = auto()
        THUMB_UP = auto()
        THUMB_DOWN = auto()
        I_LOVE_YOU = auto()
        POINTING_UP = auto()

    def __init__(self):
        """init the game"""
        self.state = self.GameState.TELEOPERATION
        self.recent_gestures = []
        self.landmarks = None
        self.hold_time = 1
        self.menu_select = 0

    def start(self):
        """start the game. not the same as init"""
        self.leap_hand = LeapHand()
        self.tracker = Tracker()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read_pose(self):
        """read the poses to the consol"""
        while(True):
            pose = self.leap_hand.get_pose()
            print(pose)
            time.sleep(.06)

    def loop(self):
        """loop"""

        with self.tracker.mp_hands.Hands(min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7, 
                    max_num_hands=1) as hands:

            # only while the ecapture window is open
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip and convert the frame color
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = hands.process(image_rgb)

                # track gesture
                self.track_gesture(mp_image)

                # check for state changes
                self.check_state_change()
                # print(self.state)

                # run the current state
                self.run_state(frame, result)

                # end of loop
                self.tracker.recognition_result_list.clear() # clear the big list
                #cv2.imshow("Hand Tracking", frame) # display (can be removed)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                # time.sleep(.06)
                print(time.time())

        self.close()
                
    def track_gesture(self, mp_image):
        """tracks the gesutres"""

        self.tracker.recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        current_gesture = self.Gestures.NO_GESTURE

        if self.tracker.recognition_result_list:
            top_gesture = self.tracker.recognition_result_list[0].gestures
            # print(tracker.recognition_result_list[0].gestures)
            if "Open_Palm" in str(top_gesture):
                current_gesture = self.Gestures.OPEN_PALM
            elif "Victory" in str(top_gesture):
                current_gesture = self.Gestures.VICTORY
            elif "Closed_Fist" in str(top_gesture):
                current_gesture = self.Gestures.CLOSED_FIST
            elif "Thumb_Up" in str(top_gesture):
                current_gesture = self.Gestures.THUMB_UP
            elif "Thumb_Down" in str(top_gesture):
                current_gesture = self.Gestures.THUMB_DOWN
            elif "ILoveYou" in str(top_gesture):
                current_gesture = self.Gestures.I_LOVE_YOU
            elif "Pointing_Up" in str(top_gesture):
                current_gesture = self.Gestures.POINTING_UP

        # append a tuple of the current gesture and the time
        self.recent_gestures.append((current_gesture, time.time()))

    def run_state(self, frame, result):
        """rn the current game state"""

        # i am old and dont know how the new match thing workds
        if self.state == self.GameState.ERROR:
            # restart the game. something borked
            pass
        elif self.state == self.GameState.MENU:
            self.menu()
        elif self.state == self.GameState.TELEOPERATION:
            self.teleoperate(frame, result)
        elif self.state == self.GameState.DANCE:
            self.dance()
        elif self.state == self.GameState.WALKING:
            self.walk()        

    def check_state_change(self):
        """based on the current state, check if there can be a state change"""
        
        current_gesture = self.get_current_gesture(self.hold_time)
        # print(current_gesture)

        selection_made = False

        # switch to menu mode!
        if (self.state == self.GameState.TELEOPERATION or self.state == self.GameState.DANCE or self.state == self.GameState.WALKING) and current_gesture == self.Gestures.I_LOVE_YOU:
            self.state = self.GameState.MENU
            selection_made = True
        
        elif self.state == self.GameState.MENU:

            # up in menu if at 0 or 1
            if current_gesture == self.Gestures.THUMB_UP and self.menu_select <= 1:
                self.menu_select += 1
                selection_made = True
            # down in menu if at 2 or 1
            elif current_gesture == self.Gestures.THUMB_DOWN and self.menu_select >= 1:
                self.menu_select = self.menu_select - 1
                selection_made = True

            # a selection was made. change state based on menu_select now
            if current_gesture == self.Gestures.CLOSED_FIST:
                
                # yes these are magic numbers. I dont care. there are three numbers 0,1,2. 
                if self.menu_select == 0:
                    self.state = self.GameState.TELEOPERATION
                elif self.menu_select == 1:
                    self.state = self.GameState.WALKING
                elif self.menu_select == 2:
                    self.state = self.GameState.DANCE
        
        # if a selection was made then clear the memory so that it doesnt spam a menu selection
        if selection_made:
            self.recent_gestures = []

    def get_current_gesture(self, hold_time):
        """returns the current gesture if its been active for the hold time"""

        # checks for how old the list is
        if (time.time() - self.recent_gestures[0][1]) < hold_time:
            return self.Gestures.NO_GESTURE

        active_gesture = self.recent_gestures[-1][0]
        active_gesture_time = self.recent_gestures[-1][1]

        # gesture is valid if its close enough in time
        nearness_threshold = 1
        if (time.time() - active_gesture_time < nearness_threshold):
            
            # go through the recent gestures and check if they were held long enough
            for gesture in reversed(self.recent_gestures):
                g = gesture[0]
                t = gesture[1]

                # gesture switched :(
                if g != active_gesture:
                    return self.Gestures.NO_GESTURE
                
                # gesture stayed the same and held for the hold time
                if (time.time() - t) > hold_time:
                    return active_gesture
                
        return self.Gestures.NO_GESTURE   

    def menu(self):
        """Menu mode"""
        
        if self.menu_select == 0:
            pose = [0,0,0,0,0,1.5,1,1,0,1.5,1,1,0,0,0,0,]

        elif self.menu_select == 1:
            pose = [0,1.5,1,1,0,0,0,0,0,1.5,1,1,0,0,0,0,]

        elif self.menu_select == 2:
            pose = [0,1.5,1,1,0,1.5,1,1,0,0,0,0,0,0,0,0,]

        self.leap_hand.set_pose(pose)

    def teleoperate(self, frame, result):
        """teleoperate the hand"""
        
        # Check if hand landmarks are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.tracker.mp_drawing.draw_landmarks(frame, hand_landmarks,self. tracker.mp_hands.HAND_CONNECTIONS)

                # Get coordinates of landmarks, get pose, move hand
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                pose = self.tracker.get_hand_pose(landmarks)
                self.leap_hand.set_pose(pose)

    def dance(self):
        """DANCE! mode"""
        random_pose = np.random.uniform(0, np.pi-.2, 16).tolist()
        self.leap_hand.set_pose(random_pose)

    def walk(self):
        """When you walkin"""
        self.leap_hand.csv_player("joint_angles.csv")

    def close(self):
        """close the stuffs"""
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()

    game = Game()
    game.start()
    # game.read_pose()
    game.loop()

    
