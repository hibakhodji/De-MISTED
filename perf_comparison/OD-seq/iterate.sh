for f in $1/*.xml
do 
  f="$(basename -- $f)"
  f="${f%.xml}"
  ./generate.sh $f
done
