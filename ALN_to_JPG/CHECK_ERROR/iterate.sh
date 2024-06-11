for f in $1/*.xml
do 
  ../../../annot_error/annot_error $f > result.txt
  python3 check_error.py . $f result.txt
done
rm result.txt
