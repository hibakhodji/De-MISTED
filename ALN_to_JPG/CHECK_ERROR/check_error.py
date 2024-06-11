import os
import sys
import shutil

directory = sys.argv[1]
xml_file = sys.argv[2]
txt_file = sys.argv[3]
outputdir = "../NO_ERRORS"


f = open(txt_file, "r")
lines = [line.strip() for line in f if line.strip()]
if(len(lines)==2): shutil.move(os.path.join(directory, xml_file), os.path.join(directory, outputdir)) 

f.close()
