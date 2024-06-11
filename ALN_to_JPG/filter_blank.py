import PIL
from PIL import Image
import sys
from PIL import ImageChops
import os
import shutil


directory = sys.argv[1]
outputdir = "./blank"

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
       img = Image.open(filename)
       if not ImageChops.invert(img).getbbox(): shutil.move(os.path.join(directory, filename), os.path.join(directory, outputdir)) 
       else: continue 
    else:
        continue

