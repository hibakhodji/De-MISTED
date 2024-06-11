for f in ./*.xml
do 
  f="$(basename -- $f)"
  f="${f%.xml}"
  ./generate.sh $f
done
