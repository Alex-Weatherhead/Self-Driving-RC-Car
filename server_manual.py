import socket

import numpy as np
from PIL import Image

s = socket.socket()

host = "192.168.2.14"
port = 8000
s.bind((host, port))
s.listen(5)

counter = 0
folder = 'dataset/training'
while True:
    
    print("Looking for connections.")
        
    client_socket, address = s.accept()
    print('Got connection from', address)
    
    bytes = b''
    
    image_dimensions = (80, 320, 3)
    
    bytes_for_image = 1
    for dimension in image_dimensions:
        bytes_for_image *= dimension
    
    bytes_for_angle = 6
    bytes_for_speed = 3
    
    expected = bytes_for_image + bytes_for_angle + bytes_for_speed
    received = 0
    
    while True:
    
        recv = client_socket.recv(1024)
        
        if recv == b'': # The socket has been closed on the other end.
            break
        
        received += len(recv)
        bytes += recv
        
        if received >= expected:
                
            image_array = np.fromstring(bytes[:bytes_for_image], dtype=np.uint8).reshape(image_dimensions)
            
            if len(image_dimensions) == 2:
                image = Image.fromarray(image_array, mode='L')
            elif len(image_dimensions) == 3:
                image = Image.fromarray(image_array, mode='RGB')
                
            angle = bytes[bytes_for_image:(bytes_for_image + bytes_for_angle)].decode('utf-8')
            speed = bytes[(bytes_for_image + bytes_for_angle):(bytes_for_image + bytes_for_angle + bytes_for_speed)].decode('utf-8')
                
            image.save('{}/image#{}-{}-{}.png'.format(folder, counter, angle, speed))
            counter += 1
    
            received -= expected
            bytes = bytes[expected:]
    
    """
    else:
        
        print('Incorrect number of bytes received. Expected {} but received {}.'.format(expected, received))
        quit()
    """
           
    client_socket.close()

        