Before start the training, you need to clean run below .py files
clean_yolo_labels.py and validate_yolo_labels.py

Initial training:

(.venv) upendra@Upendras-MacBook-Air agritech-train % yolo train \  
  model=yolov8s.pt \                           
  data=/Users/upendra/Downloads/Jowar.v1i.yolov8/data.yaml \     
  epochs=10 \
  imgsz=640 \
  batch=4 \   
  device=mps \
  mosaic=0 \
  rect=False \
  workers=2 \
  project=runs/detect \
  name=jowar_e10

Then you can increment training:

(.venv) upendra@Upendras-MacBook-Air agritech-train % yolo train \  
  model=/Users/upendra/Desktop/agritech-train/runs/detect/runs/detect/jowar_e10/weights/best.pt \                                                
  data=/Users/upendra/Downloads/Jowar.v1i.yolov8/data.yaml \
  epochs=10 \
  imgsz=640 \
  batch=4 \   
  device=mps \
  mosaic=0 \
  rect=False \
  workers=2 \
  project=runs/detect \
  name=jowar_e20

Then you can validate:

(.venv) upendra@Upendras-MacBook-Air agritech-train % yolo val \    
  model=/Users/upendra/Desktop/agritech-train/runs/detect/runs/detect/jowar_e70/weights/best.pt \                                                
  data=/Users/upendra/Downloads/Jowar.v1i.yolov8/data.yaml \
  imgsz=640 \
  batch=1 \   
  device=mps \
  rect=False \
  half=False
Then you can predict:

(.venv) upendra@Upendras-MacBook-Air agritech-train % yolo predict \
  model=/Users/upendra/Desktop/agritech-train/runs/detect/runs/detect/jowar_e80/weights/best.pt \                                                
  source=/Users/upendra/Downloads/Jowar.v1i.yolov8/valid/images \
  imgsz=640 \
  device=mps  
