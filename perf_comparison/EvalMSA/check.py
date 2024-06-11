import shutil
import os
import sys
import re

file = sys.argv[1]
errors = "./errors"
noerrors = "./noerrors"


with open(file) as f:
    for line in f:
        if "Outliers" in line:

            if len(line.split())>1: shutil.move(os.path.join("./", file), os.path.join("./", errors))

            else: shutil.move(os.path.join("./", file), os.path.join("./", noerrors))
