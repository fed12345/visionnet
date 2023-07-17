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
    vertices = [corner1,corner2,corner3,corner4]
    
    # Calculate the centroid of the quadrilateral
    centroid = centerOfGate(corner1,corner2,corner3,corner4)
    
    # Calculate the angle deviations from the centroid for each vertex
    angle_deviations = []
    for vertex in vertices:
        dx = vertex[0] - centroid[0]
        dy = vertex[1] - centroid[1]
        angle_deviations.append(np.arctan2(dy, dx))
    pihalf = np.pi/2
    skew = np.max(np.array([((np.max(angle_deviations) -pihalf)/pihalf), ((pihalf - np.min(angle_deviations))/pihalf)]))

    return skew
