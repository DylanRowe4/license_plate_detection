import LicensePlateDetector

#class can take in an image instead of a video stream as well
if __name__=="__main__":
    object_detector = LicensePlateDetector.DetectObjects(model_weights="./license_plate_detection/best.pt",
                                                         stream_url='https://www.youtube.com/watch?v=9db5tH5t-bg&t=15s',
                                                         show_frames=True)
    object_detector()