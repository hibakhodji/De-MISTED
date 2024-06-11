for f in $1/*.xml
do 
  f="$(basename -- $f)"
  v="${f%.xml}_f"
  ../../filter_error/filter_error $f $v
  mv $v.xml ../NO_ERRORS
done
