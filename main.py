import cv2
import time
import random
import numpy as np
import mediapipe as mp

def game_logic(sys_choice, player_choice):
    """
    0 -> Rock,  1-> Paper,  2 -> Scissors

    Return:
            0 -> player lose, 1 -> player won, -1 -> clash
    """
    # If choices are the same, it's a clash
    if sys_choice == player_choice:
        return -1
    
    # Rock beats Scissors
    if sys_choice == 0 and player_choice == 2:
        return 0
    if sys_choice == 2 and player_choice == 0:
        return 1
    
    # Paper beats Rock
    if sys_choice == 1 and player_choice == 0:
        return 0
    if sys_choice == 0 and player_choice == 1:
        return 1
    
    # Scissors beats Paper
    if sys_choice == 2 and player_choice == 1:
        return 0
    if sys_choice == 1 and player_choice == 2:
        return 1
    


def start_sign(hand_landmarks):
    """
    🤘--> sign 
    """
    fore_tip = hand_landmarks.landmark[8].y
    fore_pip = hand_landmarks.landmark[8-3].y
    pinkey_tip = hand_landmarks.landmark[20].y
    pinkey_pip = hand_landmarks.landmark[20-3].y
    middle_pip = hand_landmarks.landmark[12-3].y
    thumb_tip = hand_landmarks.landmark[4].y
    # start_flag = True
    return fore_pip > fore_tip and pinkey_pip > pinkey_tip and thumb_tip > middle_pip

def stone_status(hand_landmarks):
    """
    Check if hand sign is for stone
    """
    thumb_tip = hand_landmarks.landmark[4].y
    index_finger_tip = hand_landmarks.landmark[8].y
    return thumb_tip <= index_finger_tip

def paper_status(hand_landmarks):
    """
    Check if hand sign is for paper
    """
    finger_pos = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
    paper_flag = True

    for _, pos in finger_pos.items():
        finger_tip = hand_landmarks.landmark[pos].y
        finger_pip = hand_landmarks.landmark[pos-3].y      
        if finger_tip > finger_pip:
            paper_flag = False
    return paper_flag

def scisor_status(hand_landmarks):
    """
    Check if hand sign is for Scisor
    """
    index_middle_pos = {"index": 8, "middle": 12}
    ring_pinky_pos =  {"ring": 16, "pinky": 20}
    paper_flag = True    
    for _, pos in index_middle_pos.items():
        finger_tip = hand_landmarks.landmark[pos].y
        finger_pip = hand_landmarks.landmark[pos-3].y      
        if finger_tip > finger_pip:
            paper_flag = False
    
    for _, pos in ring_pinky_pos.items():
        finger_tip = hand_landmarks.landmark[pos].y
        finger_pip = hand_landmarks.landmark[pos-3].y      
        if finger_tip < finger_pip:
            paper_flag = False
    return paper_flag

def main():
    started_flag = False
    move_list = ["Stone", "Paper", "Scissors", "Not Allowed", ""]
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # Open the default camera (usually the first camera connected)
    cap = cv2.VideoCapture(0)
    # system response window
    font = cv2.FONT_HERSHEY_COMPLEX
    sys_pad = np.zeros((471, 636, 3)) 
    cv2.putText(sys_pad, "Welcome!!", (150, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #Draw the hand coordinates
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # print(mp_hands.HAND_CONNECTIONS)
                    # print(hand_landmarks)
                    if start_sign(hand_landmarks=hand_landmarks) or started_flag:
                        
                        sys_pad = np.zeros((471, 636, 3)) # reset pad
                        cv2.putText(sys_pad, "Starting Countdown", (150, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        
                        # time.sleep(3) if not started_flag else None

                        sys_pad = np.zeros((471, 636, 3)) # reset pad
                        cv2.putText(sys_pad, "Show your move", (150, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        cv2.putText(image, "Your move", (350, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        move = -1
                        if stone_status(hand_landmarks=hand_landmarks):
                            move = 0
                        elif paper_status(hand_landmarks=hand_landmarks):
                            move = 1
                        elif scisor_status(hand_landmarks=hand_landmarks):
                            move = 2
                        else:
                            move = 3
                        cv2.putText(image, move_list[move], (150, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        if move == 3:
                            cv2.putText(sys_pad, "Retry....", (250, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        started_flag = True
                        
                        
                        # sys_prediction = random.choice([0, 1, 2])
                        
                        # result = game_logic(sys_choice=sys_prediction, player_choice=move)

                        # sys_pad = np.zeros((471, 636, 3)) # reset pad
                        # cv2.putText(sys_pad, f"Computer's Move: {move_list[sys_prediction]}", (150, 50), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        # if result == 1:
                        #     cv2.putText(sys_pad, "Congratulations!!🥳' you won", (150, 150), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        # if result == 0:
                        #     cv2.putText(sys_pad, "Looser wooo, looser 🤯", (150, 150), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        # if result == -1:    
                        #     cv2.putText(sys_pad, "Draw!! kya kismat h sala", (150, 150), font, 1, (0, 255, 0), 2, cv2.LINE_4)
                        # started_flag = False




            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands',image)# cv2.flip(image, 1))
            cv2.imshow("Khiladi System", sys_pad)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    # # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
