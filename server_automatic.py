import socket

import numpy as np
from PIL import Image
from keras.models import load_model
import keras.losses

from src.preprocessing import preprocessor
from src.losses import rmse

keras.losses.rmse = rmse
model = load_model('models/1-0.0001.depper.09-0.10.hdf5')

HEIGHT, WIDTH, CHANNELS = 80, 320, 3
BYTES_FOR_IMAGE = HEIGHT * WIDTH * CHANNELS
count = 0

s = socket.socket()

host = "192.168.2.14"
port = 8000
s.bind((host, port))
s.listen(5)

while True:
    
    print("Looking for connections.")
        
    client_socket, address = s.accept()
    print('Got connection from', address)
    
    bytes = b''
    
    expected = BYTES_FOR_IMAGE
    received = 0
    
    while True:
    
        recv = client_socket.recv(1024)
        
        if recv == b'': # The socket has been closed on the other end.
            break
        
        received += len(recv)
        bytes += recv
        
        if received >= expected:
                
            image_array = np.fromstring(bytes[:BYTES_FOR_IMAGE], dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
            
            count += 1
    
            prediction = model.predict(preprocessor(image_array))
            print(prediction)
            angle = round((((-prediction[0,0] - -1) / (+1 - -1)) * (140 - 40) + 40), 2)
            print(angle)
            angle = np.clip(angle, 40, 140)
            print(angle)

            Image.fromarray(image_array).save('dataset/test/image#{}-{}.png'.format(count, angle))

            angle_bytes = str(angle).encode('utf-8')
            if len(angle_bytes) < 6:
                angle_bytes = b' '*(6 - len(angle_bytes)) + angle_bytes

            client_socket.sendall(angle_bytes)

            received -= expected
            bytes = bytes[expected:]

    client_socket.close()

        