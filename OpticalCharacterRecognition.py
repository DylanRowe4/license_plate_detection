import cv2
import re
import numpy as np
import torch

class myOCR:
    """
    Load a custom YOLOv5 model trained to identify characters and numbers for optical character
    recognition on license plates.
    """
    def __init__(self, model_weights):
        """
        Initialize class with location to model weights, device and class names from model.
        """
        self.model_weights = model_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.classes = self.model.names
        
    def load_model(self):
        """
        Load pretrained model from PyTorch hub.
        """
        model = torch.hub.load("ultralytics/yolov5", "custom", './ocr/best.pt',
                               _verbose=False, device='cpu')
        #set model confidence to 0.7
        model.conf = 0.7
        return model

    def class_to_labels(self, x):
        """
        For a given label value, return the corresponding string name given to the model at training time.
        """
        return self.classes[int(x)]
    
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
        #get class of detections
        classes = [self.class_to_labels(int(label)) for label in labels]
        #get bounding box as well
        bboxes = results.xyxyn[0].cpu()[:, :-1].numpy()
        return classes, bboxes
    
    def OCR(self, frame):
        """
        Sort class labels and bouding boxes, then output the class labels into a string.
        """
        #infer on frame
        classes, bboxes = self.score_frame(frame)
        #put class and bbox into a tuple
        class_bbox = [(classes[i], bboxes[i]) for i in range(len(classes))]
        #sort them by the x1 location
        sorted_classes = sorted(class_bbox, key=lambda tup: tup[1][0])
        text = ''.join([tup[0] for tup in sorted_classes])
        return text