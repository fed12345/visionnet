import os
import zipfile
import tempfile

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

print(compressDecompress('evalutation/models/dronet_base.tflite'))
print(compressDecompress('evalutation/models/dronet_pruned.tflite'))
