import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pyrebase
import firebase_admin
from firebase_admin import credentials
import time
import datetime

# 현재 스크립트가 실행되는 디렉토리 경로 가져오기
current_dir = os.path.dirname(__file__)

# 서비스 계정 키 파일의 이름 가져오기
service_account_key_filename = "dlfd-cf878-firebase-adminsdk-8c0ve-615d886ff0.json"

# 서비스 계정 키 파일의 전체 경로 설정
service_account_key_path = os.path.join(current_dir, service_account_key_filename)

# Firebase Admin SDK를 사용하여 서비스 계정 자격 증명 초기화
cred = credentials.Certificate(service_account_key_path)
firebase_admin.initialize_app(cred)

# Firebase 설정 파일 읽어오기
with open("auth.json") as f:
    config = json.load(f)

# Pyrebase를 사용하여 Firebase 앱 초기화
firebase = pyrebase.initialize_app(config)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_euclidean_distance(point1_x, point1_y, point2_x, point2_y):
    """
    두 점 사이의 유클리드 거리를 계산
    """
    return np.sqrt((point1_x - point2_x)**2 + (point1_y - point2_y)**2)

def calculate_angle(a, b, c):
    a = np.array(a)  # First 
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return int(angle)

cap = cv2.VideoCapture(0)

# Curl counter variables
fall = 0
falling_start_time = None
falling_duration = 15

stage = None
caputerd_stage = None

# 새로 추가된 거리 임계값 설정
distance_threshold = 150  # 유클리드 거리 임계값
vertical_threshold = 100  # 수직 거리 임계값

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지의 높이와 너비를 얻습니다.
        image_hight, image_width, _ = frame.shape

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
            
            # DOT(점찍기)
            # dot - NOSE
            dot_NOSE_X = int(landmarks[mp_pose.PoseLandmark.NOSE].x * image_width)
            dot_NOSE_Y = int(landmarks[mp_pose.PoseLandmark.NOSE].y * image_hight)
            
            # dot - LEFT_SHOULDER
            dot_LEFT_SHOULDER_X = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width)
            dot_LEFT_SHOULDER_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight)
            
            # dot - RIGHT_SHOULDER
            dot_RIGHT_SHOULDER_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width)
            dot_RIGHT_SHOULDER_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_hight)
            
            # dot - LEFT_ELBOW
            dot_LEFT_ELBOW_X = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width)
            dot_LEFT_ELBOW_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_hight)
            
            # dot - RIGHT_ELBOW
            dot_RIGHT_ELBOW_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width)
            dot_RIGHT_ELBOW_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_hight)
            
            # dot - LEFT_WRIST
            dot_LEFT_WRIST_X = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width)
            dot_LEFT_WRIST_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image_hight)
            
            # dot - RIGHT_WRIST
            dot_RIGHT_WRIST_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)
            dot_RIGHT_WRIST_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_hight)
            
            # dot - LEFT_HIP
            dot_LEFT_HIP_X = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)
            dot_LEFT_HIP_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_hight)
            
            # dot - RIGHT_HIP
            dot_RIGHT_HIP_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)
            dot_RIGHT_HIP_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_hight)
            
            # dot - LEFT_KNEE
            dot_LEFT_KNEE_X = int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)
            dot_LEFT_KNEE_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image_hight)
            
            # dot - RIGHT_KNEE
            dot_RIGHT_KNEE_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width)
            dot_RIGHT_KNEE_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_hight)
            
            # dot - LEFT_ANKLE
            dot_LEFT_ANKLE_X = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width)
            dot_LEFT_ANKLE_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_hight)
            
            # dot - RIGHT_ANKLE
            dot_RIGHT_ANKLE_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width)
            dot_RIGHT_ANKLE_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_hight)
            
            # dot - LEFT_HEEL
            dot_LEFT_HEEL_X = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x * image_width)
            dot_LEFT_HEEL_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * image_hight)
            
            # dot - RIGHT_HEEL
            dot_RIGHT_HEEL_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x * image_width)
            dot_RIGHT_HEEL_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * image_hight)
            
            # dot - LEFT_FOOT_INDEX
            dot_LEFT_FOOT_INDEX_X = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image_width)
            dot_LEFT_FOOT_INDEX_Y = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image_hight)
            
            # dot - RIGHT_FOOT_INDEX
            dot_RIGHT_FOOT_INDEX_X = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image_width)
            dot_RIGHT_FOOT_INDEX_Y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image_hight)
            
            # 중간점 계산
            # dot - LEFT_ARM_WRIST_ELBOW
            dot_LEFT_ARM_A_X = int((dot_LEFT_WRIST_X + dot_LEFT_ELBOW_X) / 2)
            dot_LEFT_ARM_A_Y = int((dot_LEFT_WRIST_Y + dot_LEFT_ELBOW_Y) / 2)
            LEFT_ARM_WRIST_ELBOW = [dot_LEFT_ARM_A_X, dot_LEFT_ARM_A_Y]
            
            # dot - RIGHT_ARM_WRIST_ELBOW
            dot_RIGHT_ARM_A_X = int((dot_RIGHT_WRIST_X + dot_RIGHT_ELBOW_X) / 2)
            dot_RIGHT_ARM_A_Y = int((dot_RIGHT_WRIST_Y + dot_RIGHT_ELBOW_Y) / 2)
            RIGHT_ARM_WRIST_ELBOW = [dot_RIGHT_ARM_A_X, dot_RIGHT_ARM_A_Y]
            
            # dot - LEFT_ARM_SHOULDER_ELBOW
            dot_LEFT_ARM_SHOULDER_ELBOW_X = int((dot_LEFT_SHOULDER_X + dot_LEFT_ELBOW_X) / 2)
            dot_LEFT_ARM_SHOULDER_ELBOW_Y = int((dot_LEFT_SHOULDER_Y + dot_LEFT_ELBOW_Y) / 2)
            LEFT_ARM_SHOULDER_ELBOW = [dot_LEFT_ARM_SHOULDER_ELBOW_X, dot_LEFT_ARM_SHOULDER_ELBOW_Y]
            
            # dot - RIGHT_ARM_SHOULDER_ELBOW
            dot_RIGHT_ARM_SHOULDER_ELBOW_X = int((dot_RIGHT_SHOULDER_X + dot_RIGHT_ELBOW_X) / 2)
            dot_RIGHT_ARM_SHOULDER_ELBOW_Y = int((dot_RIGHT_SHOULDER_Y + dot_RIGHT_ELBOW_Y) / 2)
            RIGHT_ARM_SHOULDER_ELBOW = [dot_RIGHT_ARM_SHOULDER_ELBOW_X, dot_RIGHT_ARM_SHOULDER_ELBOW_Y]
            
            # dot - BODY_SHOULDER_HIP
            dot_BODY_SHOULDER_HIP_X = int((dot_RIGHT_SHOULDER_X + dot_RIGHT_HIP_X + dot_LEFT_SHOULDER_X + dot_LEFT_HIP_X) / 4)
            dot_BODY_SHOULDER_HIP_Y = int((dot_RIGHT_SHOULDER_Y + dot_RIGHT_HIP_Y + dot_LEFT_SHOULDER_Y + dot_LEFT_HIP_Y) / 4)
            BODY_SHOULDER_HIP = [dot_BODY_SHOULDER_HIP_X, dot_BODY_SHOULDER_HIP_Y]
            
            # dot - LEFT_LEG_HIP_KNEE
            dot_LEFT_LEG_HIP_KNEE_X = int((dot_LEFT_HIP_X + dot_LEFT_KNEE_X) / 2)
            dot_LEFT_LEG_HIP_KNEE_Y = int((dot_LEFT_HIP_Y + dot_LEFT_KNEE_Y) / 2)
            LEFT_LEG_HIP_KNEE = [dot_LEFT_LEG_HIP_KNEE_X, dot_LEFT_LEG_HIP_KNEE_Y]
            
            # dot - RIGHT_LEG_HIP_KNEE
            dot_RIGHT_LEG_HIP_KNEE_X = int((dot_RIGHT_HIP_X + dot_RIGHT_KNEE_X) / 2)
            dot_RIGHT_LEG_HIP_KNEE_Y = int((dot_RIGHT_HIP_Y + dot_RIGHT_KNEE_Y) / 2)
            RIGHT_LEG_HIP_KNEE = [dot_RIGHT_LEG_HIP_KNEE_X, dot_RIGHT_LEG_HIP_KNEE_Y]
            
            # dot - LEFT_LEG_KNEE_ANKLE
            dot_LEFT_LEG_KNEE_ANKLE_X = int((dot_LEFT_ANKLE_X + dot_LEFT_KNEE_X) / 2)
            dot_LEFT_LEG_KNEE_ANKLE_Y = int((dot_LEFT_ANKLE_Y + dot_LEFT_KNEE_Y) / 2)
            LEFT_LEG_KNEE_ANKLE = [dot_LEFT_LEG_KNEE_ANKLE_X, dot_LEFT_LEG_KNEE_ANKLE_Y]
            
            # dot - RIGHT_LEG_KNEE_ANKLE
            dot_RIGHT_LEG_KNEE_ANKLE_X = int((dot_RIGHT_ANKLE_X + dot_RIGHT_KNEE_X) / 2)
            dot_RIGHT_LEG_KNEE_ANKLE_Y = int((dot_RIGHT_ANKLE_Y + dot_RIGHT_KNEE_Y) / 2)
            RIGHT_LEG_KNEE_ANKLE = [dot_RIGHT_LEG_KNEE_ANKLE_X, dot_RIGHT_LEG_KNEE_ANKLE_Y]
            
            # dot - LEFT_FOOT_INDEX_HEEL
            dot_LEFT_FOOT_INDEX_HEEL_X = int((dot_LEFT_FOOT_INDEX_X + dot_LEFT_HEEL_X) / 2)
            dot_LEFT_FOOT_INDEX_HEEL_Y = int((dot_LEFT_FOOT_INDEX_Y + dot_LEFT_HEEL_Y) / 2)
            LEFT_FOOT_INDEX_HEEL = [dot_LEFT_FOOT_INDEX_HEEL_X, dot_LEFT_FOOT_INDEX_HEEL_Y]
            # dot - RIGHT_FOOT_INDEX_HEEL
            dot_RIGHT_FOOT_INDEX_HEEL_X = int((dot_RIGHT_FOOT_INDEX_X + dot_RIGHT_HEEL_X) / 2)
            dot_RIGHT_FOOT_INDEX_HEEL_Y = int((dot_RIGHT_FOOT_INDEX_Y + dot_RIGHT_HEEL_Y) / 2)
            RIGHT_FOOT_INDEX_HEEL = [dot_RIGHT_FOOT_INDEX_HEEL_X, dot_RIGHT_FOOT_INDEX_HEEL_Y]
            
            # dot - UPPER_BODY
            dot_UPPER_BODY_X = int((dot_NOSE_X + dot_LEFT_ARM_A_X + dot_RIGHT_ARM_A_X + 
            dot_LEFT_ARM_SHOULDER_ELBOW_X + dot_RIGHT_ARM_SHOULDER_ELBOW_X + 
            dot_BODY_SHOULDER_HIP_X) / 6)
            dot_UPPER_BODY_Y = int((dot_NOSE_Y + dot_LEFT_ARM_A_Y + dot_RIGHT_ARM_A_Y + 
            dot_LEFT_ARM_SHOULDER_ELBOW_Y + dot_RIGHT_ARM_SHOULDER_ELBOW_Y + 
            dot_BODY_SHOULDER_HIP_Y) / 6)
            UPPER_BODY = [dot_UPPER_BODY_X, dot_UPPER_BODY_Y]
            
            # dot - LOWER_BODY
            dot_LOWER_BODY_X = int((dot_LEFT_LEG_HIP_KNEE_X + dot_RIGHT_LEG_HIP_KNEE_X + 
            dot_LEFT_LEG_KNEE_ANKLE_X + dot_RIGHT_LEG_KNEE_ANKLE_X + 
            dot_LEFT_FOOT_INDEX_HEEL_X + dot_RIGHT_FOOT_INDEX_HEEL_X) / 6)
            dot_LOWER_BODY_Y = int((dot_LEFT_LEG_HIP_KNEE_Y + dot_RIGHT_LEG_HIP_KNEE_Y + 
            dot_LEFT_LEG_KNEE_ANKLE_Y + dot_RIGHT_LEG_KNEE_ANKLE_Y + 
            dot_LEFT_FOOT_INDEX_HEEL_Y + dot_RIGHT_FOOT_INDEX_HEEL_Y) / 6)
            LOWER_BODY = [dot_LOWER_BODY_X, dot_LOWER_BODY_Y]
            
            # dot - BODY
            dot_BODY_X = int((dot_UPPER_BODY_X + dot_LOWER_BODY_X) / 2)
            dot_BODY_Y = int((dot_UPPER_BODY_Y + dot_LOWER_BODY_Y) / 2)
            BODY = [dot_BODY_X, dot_BODY_Y]
            
            # Get coordinates for angle calculation
            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            angle_shoulder_l = calculate_angle(elbow_l, shoulder_l, hip_l)
            angle_shoulder_r = calculate_angle(elbow_r, shoulder_r, hip_r)
            angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)
            angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)
            angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)
            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
            
            # Calculate foot center
            Point_of_action_LEFT_X = int((dot_LEFT_FOOT_INDEX_X + dot_LEFT_HEEL_X) / 2)
            Point_of_action_LEFT_Y = int((dot_LEFT_FOOT_INDEX_Y + dot_LEFT_HEEL_Y) / 2)
            Point_of_action_RIGHT_X = int((dot_RIGHT_FOOT_INDEX_X + dot_RIGHT_HEEL_X) / 2)
            Point_of_action_RIGHT_Y = int((dot_RIGHT_FOOT_INDEX_Y + dot_RIGHT_HEEL_Y) / 2)
            
            Point_of_action_X = int((Point_of_action_LEFT_X + Point_of_action_RIGHT_X) / 2)
            Point_of_action_Y = int((Point_of_action_LEFT_Y + Point_of_action_RIGHT_Y) / 2)
            Point_of_action = [Point_of_action_X, Point_of_action_Y]
            
            # Visualize angles and points (이전과 동일한 시각화 코드...)
            # Calculate angle - elbow_l
            angle_elbow_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            
            # Calculate angle - elbow_r
            angle_elbow_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
            # Calculate angle - shoulder_l
            angle_shoulder_l = calculate_angle(elbow_l, shoulder_l, hip_l)
            
            # Calculate angle - shoulder_r
            angle_shoulder_r = calculate_angle(elbow_r, shoulder_r, hip_r)
            
            # Calculate angle - hip_l
            angle_hip_l = calculate_angle(shoulder_l, hip_l, knee_l)
            
            # Calculate angle - hip_r
            angle_hip_r = calculate_angle(shoulder_r, hip_r, knee_r)
            
            # Calculate angle - knee_l
            angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)
            
            # Calculate angle - knee_r
            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
            

            #발 사이값
            Point_of_action_LEFT_X = int( 
                ((dot_LEFT_FOOT_INDEX_X +  dot_LEFT_HEEL_X)/2) )
            
            Point_of_action_LEFT_Y = int( 
                ((dot_LEFT_FOOT_INDEX_Y+   dot_LEFT_HEEL_Y)/2) )
            
            
            Point_of_action_RIGHT_X = int( 
                ((dot_RIGHT_FOOT_INDEX_X +  dot_RIGHT_HEEL_X)/2) )
            
            Point_of_action_RIGHT_Y = int( 
                ((dot_RIGHT_FOOT_INDEX_Y+   dot_RIGHT_HEEL_Y)/2) )           
        
            #발 사이값 평균
        
            Point_of_action_X = int ( (Point_of_action_LEFT_X +  Point_of_action_RIGHT_X)/2 )
            
            Point_of_action_Y = int ( (Point_of_action_LEFT_Y +  Point_of_action_RIGHT_Y)/2 )
            
            
            #발 사이값 좌표
            Point_of_action = [Point_of_action_X , Point_of_action_Y]
            
            
            # Visualize angle - 발 사이값 좌표
            
            
            cv2.putText(image, str(Point_of_action), 
                        (Point_of_action_X,Point_of_action_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA)
            
            cv2.circle(image,  (Point_of_action_X , Point_of_action_Y), 5, (0,0,255), -1)
            
        
        
        
        
            
            
            
            # Visualize angle - elbow_l
            cv2.putText(image, str(angle_elbow_l), 
                        tuple(np.multiply(elbow_l, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - elbow_r
            cv2.putText(image, str(angle_elbow_r), 
                        tuple(np.multiply(elbow_r, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - shoulder_l
            cv2.putText(image, str(angle_shoulder_l), 
                        tuple(np.multiply(shoulder_l, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - shoulder_r
            cv2.putText(image, str(angle_shoulder_r), 
                        tuple(np.multiply(shoulder_r, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - hip_l
            cv2.putText(image, str(angle_hip_l), 
                        tuple(np.multiply(hip_l, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - hip_r
            cv2.putText(image, str(angle_hip_r), 
                        tuple(np.multiply(hip_r, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - knee_l
            cv2.putText(image, str(angle_knee_l), 
                        tuple(np.multiply(knee_l, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            # Visualize angle - knee_r
            cv2.putText(image, str(angle_knee_r), 
                        tuple(np.multiply(knee_r, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
            
            
            
            # Visualize dot - dot_NOSE

                        
            cv2.putText(image, str(dot_NOSE), 
                        (dot_NOSE_X,dot_NOSE_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
            
            cv2.circle(image,  (dot_NOSE_X,dot_NOSE_Y), 5, (204,252,0), -1)
            
            
            
            
            
            
            
            # Visualize dot - LEFT_ARM_WRIST_ELBO

                        
            cv2.putText(image, str(LEFT_ARM_WRIST_ELBOW), 
                        (dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
            
            cv2.circle(image,  (dot_LEFT_ARM_A_X,dot_LEFT_ARM_A_Y), 5, (204,252,0), -1)
            
            # Visualize dot - RIGHT_ARM_WRIST_ELBO

                        
            cv2.putText(image, str(RIGHT_ARM_WRIST_ELBOW), 
                        (dot_RIGHT_ARM_A_X,dot_RIGHT_ARM_A_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_RIGHT_ARM_A_X,dot_RIGHT_ARM_A_Y), 5, (204,252,0), -1)

        
    
            # Visualize dot - LEFT_ARM_SHOULDER_ELBOW

                        
            cv2.putText(image, str(LEFT_ARM_SHOULDER_ELBOW), 
                        (dot_LEFT_ARM_SHOULDER_ELBOW_X,dot_LEFT_ARM_SHOULDER_ELBOW_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                )
            
            cv2.circle(image,  (dot_LEFT_ARM_SHOULDER_ELBOW_X,dot_LEFT_ARM_SHOULDER_ELBOW_Y), 5, (204,252,0), -1)
            
            
            # Visualize dot - RIGHT_ARM_SHOULDER_ELBOW

                        
            cv2.putText(image, str(RIGHT_ARM_SHOULDER_ELBOW), 
                        (dot_RIGHT_ARM_SHOULDER_ELBOW_X,dot_RIGHT_ARM_SHOULDER_ELBOW_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                )
                
            cv2.circle(image,  (dot_RIGHT_ARM_SHOULDER_ELBOW_X,dot_RIGHT_ARM_SHOULDER_ELBOW_Y), 5, (204,252,0), -1)


            # Visualize dot - BODY_SHOULDER_HIP

                        
            cv2.putText(image, str(BODY_SHOULDER_HIP), 
                        (dot_BODY_SHOULDER_HIP_X,dot_BODY_SHOULDER_HIP_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                )
                
            cv2.circle(image,  (dot_BODY_SHOULDER_HIP_X,dot_BODY_SHOULDER_HIP_Y), 5, (204,252,0), -1)
            
            
            # Visualize dot - LEFT_LEG_HIP_KNEE

                        
            cv2.putText(image, str(LEFT_LEG_HIP_KNEE), 
                        (dot_LEFT_LEG_HIP_KNEE_X    ,    dot_LEFT_LEG_HIP_KNEE_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                                )
                    
            cv2.circle(image,  (dot_LEFT_LEG_HIP_KNEE_X    ,    dot_LEFT_LEG_HIP_KNEE_Y), 5, (204,252,0), -1)

            # Visualize dot - RIGHT_LEG_HIP_KNEE

                        
            cv2.putText(image, str(RIGHT_LEG_HIP_KNEE), 
                        (dot_RIGHT_LEG_HIP_KNEE_X    ,    dot_RIGHT_LEG_HIP_KNEE_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
                
            cv2.circle(image,  (dot_RIGHT_LEG_HIP_KNEE_X    ,    dot_RIGHT_LEG_HIP_KNEE_Y), 5, (204,252,0), -1)
            
            # Visualize dot - LEFT_LEG_KNEE_ANKLE

                        
            cv2.putText(image, str(LEFT_LEG_KNEE_ANKLE), 
                        (dot_LEFT_LEG_KNEE_ANKLE_X    ,    dot_LEFT_LEG_KNEE_ANKLE_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
                
            cv2.circle(image,  (dot_LEFT_LEG_KNEE_ANKLE_X    ,    dot_LEFT_LEG_KNEE_ANKLE_Y), 5, (204,252,0), -1)
            

            # Visualize dot - RIGHT_LEG_KNEE_ANKLE

                        
            cv2.putText(image, str(RIGHT_LEG_KNEE_ANKLE), 
                        (dot_RIGHT_LEG_KNEE_ANKLE_X    ,    dot_RIGHT_LEG_KNEE_ANKLE_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_RIGHT_LEG_KNEE_ANKLE_X    ,    dot_RIGHT_LEG_KNEE_ANKLE_Y), 5, (204,252,0), -1)
            
            
            # Visualize dot -   LEFT_FOOT_INDEX_HEEL

                        
            cv2.putText(image, str(LEFT_FOOT_INDEX_HEEL), 
                        (dot_LEFT_FOOT_INDEX_HEEL_X    ,    dot_LEFT_FOOT_INDEX_HEEL_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_LEFT_FOOT_INDEX_HEEL_X    ,    dot_LEFT_FOOT_INDEX_HEEL_Y), 5, (204,252,0), -1)
            
            
            # Visualize dot -   RIGHT_FOOT_INDEX_HEEL

                        
            cv2.putText(image, str(RIGHT_FOOT_INDEX_HEEL), 
                        (dot_RIGHT_FOOT_INDEX_HEEL_X    ,    dot_RIGHT_FOOT_INDEX_HEEL_Y) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (204,252,0), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_RIGHT_FOOT_INDEX_HEEL_X    ,    dot_RIGHT_FOOT_INDEX_HEEL_Y), 5, (204,252,0), -1)
            
            
            # Visualize dot -   UPPER_BODY

                        
            cv2.putText(image, str(UPPER_BODY), 
                        ( dot_UPPER_BODY_X    ,    dot_UPPER_BODY_Y ) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (277,220,0), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_UPPER_BODY_X    ,    dot_UPPER_BODY_Y), 9, (277,220,0), -1)
            
            
            # Visualize dot -   LOWER_BODY

                        
            cv2.putText(image, str(LOWER_BODY), 
                        ( dot_LOWER_BODY_X    ,    dot_LOWER_BODY_Y ) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (277,220,0), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_LOWER_BODY_X    ,    dot_LOWER_BODY_Y), 9, (277,220,0), -1)

            # Visualize dot -   BODY

                        
            cv2.putText(image, str(BODY), 
                        ( dot_BODY_X    ,    dot_BODY_Y ) , 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA
                            )
                    
            cv2.circle(image,  (dot_BODY_X    ,    dot_BODY_Y), 12, (0,0,255), -1)





            
            # 수정된 낙상 감지 로직
            body_to_feet_distance = calculate_euclidean_distance(
                Point_of_action_X, Point_of_action_Y,
                dot_BODY_X, dot_BODY_Y
            )
            
            # 수직 거리 계산
            vertical_distance = abs(Point_of_action_Y - dot_BODY_Y)
            
            # 낙상 상태 판단 - 거리와 수직 높이를 모두 고려
            falling = (body_to_feet_distance > distance_threshold and 
                    vertical_distance < vertical_threshold)
            standing = not falling
            
            # 거리 시각화
            cv2.line(image, 
                    (Point_of_action_X, Point_of_action_Y),
                    (dot_BODY_X, dot_BODY_Y),
                    (0, 255, 0), 2)
            
            # 거리 정보 표시
            cv2.putText(image, f'Euclidean Distance: {body_to_feet_distance:.1f}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.putText(image, f'Vertical Distance: {vertical_distance:.1f}', 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 낙상 감지 및 처리
            if falling:
                if stage != 'falling':
                    falling_start_time = time.time()
                    stage = 'falling'
                    print("Falling detected!")
                
                if ((time.time() - falling_start_time) >= falling_duration and 
                    stage == 'falling' and 
                    caputerd_stage != 'captured'):
                    
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    current_time = datetime.datetime.now().strftime("%H-%M-%S")
                    image_filename = f"fall_image_{current_time}.jpg"
                    image_path = os.path.join('/home/kim/opencv/python_dlfd/cctv_project', 
                                            image_filename)
                    
                    cv2.imwrite(image_path, frame)
                    print(f'Fall detected and image saved as {image_filename}.')
                    
                    # Firebase Storage 업로드
                    storage_path = f"{current_date}/{image_filename}"
                    storage.child(storage_path).put(image_path)
                    image_url = storage.child(storage_path).get_url(None)
                    
                    # 데이터베이스에 저장
                    db = firebase.database()
                    db.child("fall_images").push({
                        "url": image_url,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "euclidean_distance": float(body_to_feet_distance),
                        "vertical_distance": float(vertical_distance)
                    })
                    
                    falling_start_time = None
                    caputerd_stage = 'captured'
                    
            else:
                if stage != 'standing':
                    falling_start_time = None
                    stage = 'standing'
                    caputerd_stage = None

            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        except Exception as e:
            print(f"Error in fall detection: {e}")
            pass
            
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()