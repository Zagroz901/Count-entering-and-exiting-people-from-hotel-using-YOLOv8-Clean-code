from Yolo_Loading import Yolo_Loading
from Video_Setting import Video_Setting
# from ParkingMonitor import ParkingMonitor
import cv2
import numpy as np
class ParkingMonitor:
    def __init__(self, video_path, model_path):
        self.area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
        self.area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]
        self.people_entering = {}
        self.entering = set()
        self.people_exiting = {}
        self.exiting = set()
        self.yolo_loading = Yolo_Loading(model_path)
        self.yolo_loading.load_model()  # Call the load_model method here
        self.video_setting = Video_Setting(video_path)
        
    def run_video(self):
        self.video_setting.load_video()
        while True:    
            ret,frame = self.video_setting.cap.read()
            if not ret:
                break
            frame=cv2.resize(frame,(1020,500))
            self.yolo_loading.procees_frame(frame,self.people_entering,self.entering,self.people_exiting,self.exiting,self.area1,self.area2)
            cv2.polylines(frame,[np.array(self.area1,np.int32)],True,(255,0,0),2)
            cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,0),2)

            cv2.polylines(frame,[np.array(self.area2,np.int32)],True,(255,0,0),2)
            cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,0),2)
            
            i = len(self.entering)
            o = len(self.exiting)
            cv2.putText(frame,'Number of entering people= '+str(i),(20,44),cv2.FONT_HERSHEY_COMPLEX,(1),(0,255,0),2)
            cv2.putText(frame,'Number of exiting people= '+str(o),(20,82),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,255),2)
            cv2.imshow("Monitoring", frame)  # Show frame
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break
        self.video_setting.release_widnow()
