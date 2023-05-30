import os
import sys
from collections import deque

import cv2
import numpy as np
import torch

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized
cnt = 0

#op
import mediapipe as mp
import time
sys.path.append('./')

#op
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
pTime=0
cTime = time.time()
fps = 1/(cTime-pTime)
pTime = cTime


#angle
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def demo(configs):
    global left_right
    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)
    result_filename = os.path.join(configs.save_demo_dir, 'results.txt')
    frame_rate = video_loader.video_fps
    if configs.save_demo_output:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        configs.frame2_dir = os.path.join(configs.save_demo_dir, 'frame2')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)

    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    # model
    model = create_model(configs)
    model.cuda()

    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)

    model.eval()
    middle_idx = int(configs.num_frames_sequence / 2)
    queue_frames = deque(maxlen=middle_idx + 1)
    frame_idx = 0
    w_original, h_original = 1920, 1080
    w_resize, h_resize = 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize
    with torch.no_grad():
        for count, resized_imgs in video_loader:
            # take the middle one
            img = cv2.resize(resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)].transpose(1, 2, 0), (w_original, h_original))
            
            
            #openpose 추가
            results = pose.process(img)
            print(results.pose_landmarks)
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h,w,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
            
            #angle 추가-----------------------------------------------------------------------------------------------------
            landmarks = results.pose_landmarks.landmark
            # Get coordinates 왼팔
            shoulder_l = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]
                            
            # Calculate angle
            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
         
            #----------------------------------------------------------------------------------------------------------------
             # Get coordinates 오른팔
            shoulder_r = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            
            # Get coordinates 왼다리
            hip_l = [landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x,landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y]
            knee_l = [landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y]
            ankle_l = [landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].y]
            # Calculate angle
            Rangle_l = calculate_angle(hip_l, knee_l, ankle_l)

            # Get coordinates 오른다리
            hip_r = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
            knee_r = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_r = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angle
            Rangle_r = calculate_angle(hip_r, knee_r, ankle_r)

            # Get coordinates 왼쪽 손목
            elbow_l = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]
            pinky_l = [landmarks[mpPose.PoseLandmark.LEFT_PINKY.value].x,landmarks[mpPose.PoseLandmark.LEFT_PINKY.value].y]
            index_l = [landmarks[mpPose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mpPose.PoseLandmark.LEFT_INDEX.value].y]
            mid_l=[]
            mid_l.append((pinky_l[0]+index_l[0])/2)
            mid_l.append((pinky_l[1]+index_l[1])/2)
            # Calculate angle 

            Foangle_l = calculate_angle(elbow_l, wrist_l, mid_l)
            
            # Get coordinates 오른쪽 손목
            elbow_r = [landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
            pinky_r = [landmarks[mpPose.PoseLandmark.RIGHT_PINKY.value].x,landmarks[mpPose.PoseLandmark.RIGHT_PINKY.value].y]
            index_r = [landmarks[mpPose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mpPose.PoseLandmark.RIGHT_INDEX.value].y]
            mid_r=[]
            mid_r.append((pinky_r[0]+index_r[0])/2)
            mid_r.append((pinky_r[1]+index_r[1])/2)
            # Calculate angle 
            Foangle_r = calculate_angle(elbow_r, wrist_r, mid_r)
            #------------------------------------------------------------------------------------------------------------------
            # Expand the first dim
            resized_imgs = torch.from_numpy(resized_imgs).to(configs.device, non_blocking=True).float().unsqueeze(0)
            t1 = time_synchronized()
            pred_ball_global, pred_ball_local, pred_events, pred_seg = model.run_demo(resized_imgs)
            t2 = time_synchronized()
            prediction_global, prediction_local, prediction_seg, prediction_events = post_processing(
                pred_ball_global, pred_ball_local, pred_events, pred_seg, configs.input_size[0],
                configs.thresh_ball_pos_mask, configs.seg_thresh, configs.event_thresh)
            prediction_ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2)
            ]

            #

            # Get infor of the (middle_idx + 1)th frame
            if len(queue_frames) == middle_idx + 1:
                frame_pred_infor = queue_frames.popleft()
                seg_img = frame_pred_infor['seg'].astype(np.uint8)
                ball_pos = frame_pred_infor['ball']
                seg_img = cv2.resize(seg_img, (w_original, h_original))


                ploted_img = plot_detection(img, ball_pos,  seg_img, prediction_events,angle_l,elbow_l, Rangle_l, Foangle_l, angle_r, elbow_r,Rangle_r, Foangle_r)
                # print("ploted_img의 angle_r", angle_r)
                ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
                
                selected_img = select_detection(img, ball_pos, seg_img, prediction_events,angle_l,elbow_l, Rangle_l, Foangle_l, angle_r, elbow_r,Rangle_r, Foangle_r)
                # print("selected_img의 angle_r", angle_r)
                selected_img = cv2.cvtColor(selected_img, cv2.COLOR_RGB2BGR)
                

                if configs.show_image:
                    cv2.imshow('ploted_img', ploted_img)
                    cv2.waitKey(10)

                if configs.save_demo_output:
                    cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img) 
                    cv2.imwrite(os.path.join(configs.frame2_dir, 'jinsue+{:06d}.jpg'.format(frame_idx)), selected_img) 

            frame_pred_infor = {
                'seg': prediction_seg,
                'ball': prediction_ball_final
            }
            queue_frames.append(frame_pred_infor)

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx, t2 - t1))

    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(
            os.path.join(configs.frame_dir), output_video_path)
        os.system(cmd_str)

def plot_detection(img, ball_pos, seg_img, events,angle_l,elbow_l, Rangle_l, Foangle_l, angle_r, elbow_r,Rangle_r, Foangle_r):
    global left_right
    """Show the predicted information in the image"""
    img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
    img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
    if ball_pos[0]>=-100:
        img = cv2.putText(img, str(ball_pos),(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 1, cv2.LINE_AA)
    event_name = 'Bounce: {:.2f}, Net: {:.2f}'.format(events[0], events[1])
    img = cv2.putText(img, event_name, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    event_angle_l='Left elbow angle: {:.3f}'.format(angle_l)
    event_angle_r='Right elbow angle: {:.3f}'.format(angle_r)
    event_Rangle_l='Left knee angle: {:.3f}'.format(Rangle_l)
    event_Rangle_r='Right knee angle: {:.3f}'.format(Rangle_r)
    event_Foangle_l='Left wrist angle: {:.3f}'.format(Foangle_l)
    event_Foangle_r='Right wrist angle: {:.3f}'.format(Foangle_r)
    if left_right == "left":
        img = cv2.putText(img, "Left-handed",(100,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, event_angle_l,(700,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, event_Foangle_l,(700,70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
        if int(Foangle_l) <=170:
            img = cv2.putText(img, "You're using Forehand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
        else:
            img = cv2.putText(img, "You're using Backhand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
    elif left_right == "right":
        img = cv2.putText(img, "Right-handed",(100,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, event_angle_r,(700,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, event_Foangle_r,(700,70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
        if int(Foangle_r) <=170:
            img = cv2.putText(img, "You're using Forehand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
        else:
            img = cv2.putText(img, "You're using Backhand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX,1, (225, 255, 0), 1, cv2.LINE_AA)
    else:
        print("wrong choice: choose left or right.")
    img = cv2.putText(img, event_Rangle_l,(700,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
    img = cv2.putText(img, event_Rangle_r,(700,110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
    if int(Rangle_l) & int(Rangle_r) >=160:
        img = cv2.putText(img, "Coach: Posture is too high",(1300,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 255), 1, cv2.LINE_AA)
    return img
        
def select_detection(img, ball_pos, seg_img, events,angle_l,elbow_l, Rangle_l, Foangle_l, angle_r, elbow_r,Rangle_r, Foangle_r):
    result_img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
    global cnt
    global left_right
    if events[0] >= 0.6:
        """Show the predicted information in the image"""
        
        img = cv2.addWeighted(img, 1., seg_img * 255, 0.3, 0)
        img = cv2.circle(img, tuple(ball_pos), 5, (255, 0, 255), -1)
        if ball_pos[0]>=-100:
            img = cv2.putText(img, str(ball_pos),(100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 1, cv2.LINE_AA)
        event_name = 'Bounce: {:.2f}, Net: {:.2f}'.format(events[0], events[1])
        event_angle_l='Left elbow angle: {:.3f}'.format(angle_l)
        event_angle_r='Right elbow angle: {:.3f}'.format(angle_r)
        event_Rangle_l='Left knee angle: {:.3f}'.format(Rangle_l)
        event_Rangle_r='Right knee angle: {:.3f}'.format(Rangle_r)
        event_Foangle_l='Left wrist angle: {:.3f}'.format(Foangle_l)
        event_Foangle_r='Right wrist angle: {:.3f}'.format(Foangle_r)
        img = cv2.putText(img, event_name, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if left_right == "left":
            img = cv2.putText(img, "Left-handed",(100,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, event_angle_l,(700,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, event_Foangle_l,(700,70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
            if int(Foangle_l) <=170:
                img = cv2.putText(img, "You're using Forehand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, "You're using Backhand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
            
        elif left_right == "right":
            img = cv2.putText(img, "Right-handed",(100,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, event_angle_r,(700,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
            img = cv2.putText(img, event_Foangle_r,(700,70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
            if int(Foangle_r) <=170:
                img = cv2.putText(img, "You're using Forehand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
            else:
                img = cv2.putText(img, "You're using Backhand",(1300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 255, 0), 1, cv2.LINE_AA)
            
        
        else:
            print("wrong choice: choose left or right.")
        img = cv2.putText(img, event_Rangle_l,(700,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, event_Rangle_r,(700,110),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 1, cv2.LINE_AA)
         
        if int(Rangle_l) & int(Rangle_r) >=160:
            img = cv2.putText(img, "Coach: Posture is too high",(1300,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 255), 1, cv2.LINE_AA)
        result_img = img
        cv2.imwrite('/home/piai/c4web/images/'+f'IMG_{cnt}.jpg', result_img)
        cnt+=1
        while cnt>=20:
            break    
    return result_img

if __name__ == '__main__':
    configs = parse_configs()
    left_right = input("choose left or right: ")
    demo(configs=configs)
    




