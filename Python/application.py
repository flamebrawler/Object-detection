import numpy as np
import cv2
from model_loader import ObjectDetectionModel, generate_image
from tracker import reduce
from communicator import Server


model = ObjectDetectionModel()
server = Server()

while True:
    # get image from the client
    height = server.recv_int(0)
    width = server.recv_int(0)
    data = server.recv(0, height * width * 3)
    image = np.array(data).reshape((height, width, 3))

    # detect objects in the image
    detections = model(image)
    print(image)
    reduced_list = reduce(detections, [1])
    print(reduced_list)
    new_image = generate_image(image, detections, model.index)

    cv2.imshow('frame', new_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

