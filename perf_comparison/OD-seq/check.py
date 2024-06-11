import shutil
import os
import sys

file = sys.argv[1]
errors = "./errors"
noerrors = "./noerrors"


if os.stat(file).st_size == 0 : shutil.move(os.path.join("./", file), os.path.join("./", noerrors))
else : shutil.move(os.path.join("./", file), os.path.join("./", errors))
