# License Plate Reader

In this project I use two object detection models created using the train_model.ipynb script to make a license plate reader. The script trains two YOLOv5 models with datasets found and obtained from Roboflow.com; one to identify and segment the location of a license plate and the next to read the characters of the license plate and return the plate numbers.

The script ocr_image shows a comparison between the Optical Character Recognition (OCR) model from pytesseract and the object detection OCR I created along with some image manipulations to make the characters easier to read.

If you run the scripts here the model weights will be found in the ```./yolov5/trained_models``` folder.

An output from the LicensePlateDetector model used in ```main.py``` can be seen below. The script can take any image or video file as well as youtube links.

<img width="960" alt="image" src="https://user-images.githubusercontent.com/43864012/233797006-47e56f91-d34d-44de-86d2-fed0c082fc7f.png">
