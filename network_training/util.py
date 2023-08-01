import os
import zipfile
import tempfile
import numpy as np

def compressDecompress(file):
    """Compress and decompress a file

    Args:
        file (string): file to compress

    Returns:
        string: decompressed file
    """    
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    size = os.path.getsize(zipped_file)

    with zipfile.ZipFile(zipped_file, 'r') as f:
        f.extractall('evalutation')
    
    return size

def areaOfGate(corner1,corner2,corner3,corner4):
    """Compute area of gate

    Args:
        corner1 (tuple): x,y of corner
        corner2 (tuple): x,y of corner
        corner3 (tuple): x,y of corner
        corner4 (tuple): x,y of corner

    Returns:
        float: area of gate
    """    
    # compute the area of the gate
    area = 0.5*np.abs(corner1[0]*corner2[1] + corner2[0]*corner3[1] + corner3[0]*corner4[1] + corner4[0]*corner1[1] - corner2[0]*corner1[1] - corner3[0]*corner2[1] - corner4[0]*corner3[1] - corner1[0]*corner4[1])
    return area

def centerOfGate(corner1,corner2,corner3,corner4):
    """Compute center of gate

    Args:
        corner1 (tuple): x,y of corner
        corner2 (tuple): x,y of corner
        corner3 (tuple): x,y of corner
        corner4 (tuple): x,y of corner

    Returns:
        tuple: x,y of center
    """    
    # compute the center of the gate
    x = (corner1[0] + corner2[0] + corner3[0] + corner4[0])/4
    y = (corner1[1] + corner2[1] + corner3[1] + corner4[1])/4
    return (x,y)

def skewOfGate(corner1,corner2,corner3,corner4):
    """Compute skew of gate

    Args:
        corner1 (tuple): x,y of corner
        corner2 (tuple): x,y of corner
        corner3 (tuple): x,y of corner
        corner4 (tuple): x,y of corner

    Returns:
        float: skew of gate
    """    
    # compute the skew of the gate
    
    # Convert vertices to numpy array for easy calculations
    vertices = np.array(corner1,corner2,corner3,corner4)
    
    # Calculate the centroid of the quadrilateral
    centroid = np.mean(vertices, axis=0)
    
    # Calculate the angle deviations from the centroid for each vertex
    angle_deviations = []
    for vertex in vertices:
        dx = vertex[0] - centroid[0]
        dy = vertex[1] - centroid[1]
        angle_deviations.append(np.arctan2(dy, dx))
    pihalf = np.pi/2
    skew = np.max(((np.max(angle_deviations) -pihalf)/pihalf), ((pihalf - np.min(angle_deviations))/pihalf))

    return skew

def rgb2bayer(rgb_image):
    height, width, _ = rgb_image.shape
    print(height, width)
    bayer_image = np.zeros((height, width ), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Determine the color channel of the current pixel
            if i % 2 == 0:
                if j % 2 == 0:
                    channel = 'R'  # Red pixel
                else:
                    channel = 'G'  # Green pixel in even rows
            else:
                if j % 2 == 0:
                    channel = 'G'  # Green pixel in odd rows
                else:
                    channel = 'B'  # Blue pixel

            # Assign the color channel value to the Bayer image
            if channel == 'R':
                bayer_image[i, j] = rgb_image[i, j, 0]
            elif channel == 'G':
                bayer_image[i, j] = rgb_image[i, j, 1]
            else:  # channel == 'B'
                bayer_image[i, j] = rgb_image[i, j, 2]

    return bayer_image

def coord_out_of_bounds(width, height, x, y):
    return x < 0 or x >= width or y < 0 or y >= height

if __name__ == "__main__":
    #perform unit tests
    
    #test bayer conversion
    import cv2
    import tensorflow as tf

    img_dir = 'dataset/CNN/Austin1/img_1.png'
    image_file = tf.io.read_file(img_dir)
    img = tf.io.decode_png(image_file, channels = 3)
    img = tf.image.resize(img, [120,180])
    bayer = rgb2bayer(img)
    rgb_image = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2BGR)
    cv2.imshow('bayer',rgb_image)
    cv2.waitKey(0)