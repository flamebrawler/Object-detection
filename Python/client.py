from communicator import Client
import cv2
import numpy as np

if __name__ == '__main__':
    client = Client('GT5000')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        image = np.array(frame)
        height, width, _ = image.shape
        client.send_int(height)
        client.send_int(width)
        client.send(list(image.reshape(-1)), height * width * 3)
        print(height, width, image)

    cap.release()
