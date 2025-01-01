from Yolo_Loading import *
from Video_Setting import *
from ParkingMonitor import * 

video_path = r"Video\Test_1.mp4"
model_path = r"Weight\yolov8n.pt"
parking=ParkingMonitor(video_path,model_path)
parking.run_video()