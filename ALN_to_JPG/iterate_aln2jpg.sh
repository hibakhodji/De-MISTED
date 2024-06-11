for f in $1/*.xml
do 
  f="$(basename -- $f)"
  f="${f%.xml}"
  ./aln2jpg.sh $f
done
