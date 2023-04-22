import time
import os
import torch
import cv2
import pafy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, text, patheffects
from IPython.display import clear_output
import re
from OpticalCharacterRecognition import myOCR

class DetectObjects:
    """
    Load a custom YOLO model from the pytorch hub and run inference on an input stream from (mp4, rtsp,
    youtube, etc).
    """
    
    def __init__(self, model_weights, stream_url, show_frames):
        """
        Initialize class with location to model weights, video location, and the option to print the frames
        out in the IDE.
        """
        self.stream_url = stream_url
        self.model_weights = model_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.show_frames = show_frames
        self.model = self.load_model()
        self.classes = self.model.names
        self.OCR = myOCR(model_weights='./ocr/best.pt')
        
    def video_stream(self):
        """
        Creates a video stream object from the input url or device that we can read and iterate through.
        """
        
        #read youtube or http link if needed
        if "http" in str(self.stream_url):
            video = pafy.new(self.stream_url)
            best = video.getbest(preftype="mp4")
            video_path = best.url
            assert video_path is not None
        else:
            video_path = self.stream_url
        #load stream with cv2 and return
        cap = cv2.VideoCapture(video_path)
        return cap
    
    def load_model(self):
        """
        Load pretrained model from PyTorch hub.
        """
        model = torch.hub.load("ultralytics/yolov5", "custom", self.model_weights,
                               _verbose=False, device=self.device)
        #set model confidence to 0.7
        model.conf = 0.7
        return model
    
    def score_frame(self, frame):
        """
        Run inference on the input frame using the loaded YOLO model from the PyTorch Hub.
        """
        self.model.to(self.device)
        frame = [frame]
        #send frame to model for inference
        results = self.model(frame)
        #extract model prediction labels and bounding boxes
        labels = results.xyxyn[0].cpu()[:, -1].numpy()
        bboxes = results.xyxyn[0].cpu()[:, :-1].numpy()
        return labels, bboxes, results
    
    def class_to_labels(self, x):
        """
        For a given label value, return the corresponding string name given to the model at training time.
        """
        return self.classes[int(x)]
    
    def get_plate_text(self, image, bbox_coords):
        """
        Take an input image and bounding box location predicted from YOLOv5 model; convert the image to
        greyscale, use several blurs, image filters and thresholding and then apply dilation to the image. 
        """
        x1, y1, x2, y2 = bbox_coords

        #subset image based on location of license plate bounding box
        img = image[y1:y2, x1:x2]
        text = self.OCR.OCR(img)
        return text
    
    def plot_bboxes(self, results, frame):
        """
        Plot the bounding box and label results from model inference on the corresponding image frame from
        the camera stream.
        """
        #unpack tuple
        labels, bbox, res = results
        
        #number of detections by model
        n = len(labels)
        
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        
        for i in range(n):
            #get normalized bounding box coordinates
            row = bbox[i]
            #multiply normalized coordinates by image shape to get exact location of bounding box
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            
            #extract plate number from text
            plate_num = self.get_plate_text(frame, (x1, y1, x2, y2))
            
            #set color of bounding box output
            bgr = (0, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            #use plate num instead of label if we can
            plate_num if plate_num != '' else self.class_to_labels(int(labels[i]))
            #put the text into the image frame
            cv2.putText(frame, plate_num, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2)
        return frame
    
    def __call__(self):
        """
        Execute when the class is called and iterate through all frames of stream while running inference with 
        object detection model until no more frames or user cancels.
        """
        #try to read the camera stream as an image
        cap = self.video_stream()
        assert cap.isOpened()

        fps = cap.get(cv2.CAP_PROP_FPS)

        #get x and y shape of video stream
        x_shape, y_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        avg_inference_time = 0
        frame_num = 1
        while True:
            start = time.time()
            ret, frame = cap.read()
            #break if frame can't be read or no more frames
            if not ret:
                break

            #send frame to model for inference
            results = self.score_frame(frame)
            #plot the image and output through imshow
            frame = self.plot_bboxes(results, frame)
            end = time.time()

            #save inference time and frame count
            avg_inference_time += (end - start)
            frame_num += 1

            #show frames in IDE
            if self.show_frames:
                cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        #release video stream
        cap.release()            