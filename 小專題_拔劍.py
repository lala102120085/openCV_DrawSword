import cv2
import mediapipe as mp
import numpy as np
import os
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
img_path = "catch_face"
delay_time = 300
need_num = 200
need_num_half = 100
timer = 0       # 初始化計時器
threshold = 3   # 設定閾值為3秒
# 定義影片寬高、FPS 等參數
fps = 30
cap = cv2.VideoCapture(0)
bg = np.zeros((205,183,181), dtype=np.uint8)    # 將背景設置為黑色
mu = 1
count = 0
x=300
y=300
tracker = cv2.TrackerCSRT_create()  # 創建追蹤器
tracking = 0                        # 設定 False 表示尚未開始追蹤

# 定義變數
timer = 0
threshold = 3  # 舉手超過 3 秒
stop_time = 0  # 停留在相對點位上超過 3 秒
jpg_abc = cv2.imread('abc.jpg')
jpg_a = cv2.imread('a.jpg')
x1, y1 = 0, 0
x2, y2 = jpg_a.shape[1], jpg_a.shape[0]   
resized_jpg_a = cv2.resize(jpg_a, (180, 52))


def is_horizontal(right_shoulder, right_elbow, right_wrist, threshold=10):  #使用三角函數 tan
    # 計算右手臂角度
    angle_rad = math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x) - math.atan2(right_shoulder.y - right_elbow.y, right_shoulder.x - right_elbow.x)
    angle_deg = math.degrees(angle_rad)
    # 判斷角度是否接近水平線
    if abs(angle_deg - 90) < threshold:
        return True
    else:
        return False

def track_object(jpg_a, img, left_index):
    while True:
        # 在這裡寫下你原本在 while count == 10 時想執行的無限迴圈內容
        # 當要結束追蹤時，就在內部使用 break
        if cv2.waitKey(1) == ord('27'):  
            break
    
def img_slideshow(img_path, delay_time):
    # 讀取照片序列
    img_sequence = sorted(os.listdir(img_path), key=lambda x: int(x.split('.')[0]))

    print("len(img_sequence)",len(img_sequence))

    # 判斷照片張數是否達到 100 張
    if len(img_sequence) >= need_num:
        # 分成前一半和後一半兩個部分
        img_s1 = img_sequence[:need_num_half]
        img_s2 = img_sequence[need_num_half:need_num]
        idx1 = 0  # 前半段圖片的索引
        idx2 = 0  # 後半段圖片的索引
        play_times = 50  # 循環撥放次數
        play_count = 0  # 已撥放次數        
        while play_count < play_times:
            while idx1 < need_num_half or idx2 < need_num_half:
                if idx1 < need_num_half:
                    # 加載前一半的圖像
                    img1 = cv2.imread(os.path.join(img_path, img_s1[idx1]))
                    cv2.imshow("oxxostudio", img1)
                    idx1 += 1
                if idx2 < need_num_half:
                    # 加載後一半的圖像
                    img2 = cv2.imread(os.path.join(img_path, img_s2[idx2]))
                    cv2.imshow("Image Slideshow", img2)
                    idx2 += 1
                cv2.waitKey(delay_time)  # 暫停一段時間
            play_count=play_count+1
            print("play_count:",play_count)
        return True
    else:
        return False

run = True         # 設定是否更動觸碰區位置
with mp_pose.Pose(
    min_detection_confidence=0.5,
    enable_segmentation=True,       # 額外設定 enable_segmentation 參數
    min_tracking_confidence=0.5) as pose:
    
    if not cap.isOpened():
        exit()
        
    while True:
        ret, img = cap.read()
        img = cv2.flip(img,1)
        if not ret:
            break
            
        img = cv2.resize(img,(520,300))
        size = img.shape   # 取得攝影機影像尺寸
        w = size[1]        # 取得畫面寬度
        h = size[0]        # 取得畫面高度 
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # 將 BGR 轉換成 RGB
        results = pose.process(img2)                    # 偵測全身點點
        # 畫出姿勢關鍵點和連線
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(53, 174, 191), thickness=3, circle_radius=5), connection_drawing_spec=mp_drawing.DrawingSpec(color=(191, 174, 53), thickness=5))
        # time_value = next(three_sec_timer())                             # 取得計時器的回傳值
        if results.pose_landmarks is not None:
            # 擷取手腕與耳朵
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
            right_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            # 只做手腕是否高於耳朵的y軸
            left_y = (left_wrist.y < left_ear.y)
            right_y = (right_wrist.y < right_ear.y)
            yy = int(left_index.x * w)-10  # 取得手的x軸
            xx = int(left_index.y * h)-80   
            
            if mu!=need_num+1:
                img[:] = (0,0,0)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(53, 174, 191), thickness=3, circle_radius=5),connection_drawing_spec=mp_drawing.DrawingSpec(color=(191, 174, 53), thickness=5))            
                if tracking ==1:
                    img[xx:xx + jpg_a.shape[0], yy:yy + jpg_a.shape[1]] = jpg_a
                cv2.imwrite(f'catch_face/{mu}.jpg', img) # 建立 VideoWriter 物件
                mu=mu+1
                if left_y ==1 :
                    # 讀取新的背景圖片
                    img = cv2.imread("abc.jpg")
                    if count == 5 and tracking==0:
                        print("aaa")
                        img[:] = (0,0,0)
                        tracking = 1
                    elif count!= 5 and tracking==0:
                        print("bbb")
                        count +=1
                    elif count >= 5 and tracking == 1:
                        print("ccc")
                        img[:] = (0,0,0)
                        img[xx:xx + jpg_a.shape[0], yy:yy + jpg_a.shape[1]] = jpg_a
                        
                        print("jpg_a.shape",jpg_a.shape)
                        print("xx",xx)
                        print("yy",yy)
                        # if jpg_a.shape[1]<=yy or jpg_a[0]<=xx :
                        #     # jpg_a 超出畫布，取消追蹤
                        #     img[:] = (0,0,100)
                        #     print("ddd")
                        #     tracking = 0
                    else:
                        print("超出條件")
                  
                else:
                    img[:] = (0,0,0)
            elif mu==need_num+1:
                triggered = img_slideshow(img_path, delay_time)
                img[:] = (100,100,100)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(53, 174, 191), thickness=3, circle_radius=5),connection_drawing_spec=mp_drawing.DrawingSpec(color=(191, 174, 53), thickness=5))

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(53, 174, 191), thickness=3, circle_radius=5),connection_drawing_spec=mp_drawing.DrawingSpec(color=(191, 174, 53), thickness=5))
        if tracking ==1:
            img[xx:xx + jpg_a.shape[0], yy:yy + jpg_a.shape[1]] = jpg_a
        cv2.imshow('oxxostudio', img)
        
        k = cv2.waitKey(5)
        if k == 27 or k == ord('q'):
            break     # 按下 q 鍵停止
# 釋放資源
cap.release()
cv2.destroyAllWindows()