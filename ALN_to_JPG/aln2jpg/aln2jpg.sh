/biolo/clustalw $1.xml -convert -output=fasta
./adoma.sh -i $1.tfa -p $1 -prot -color
wkhtmltoimage --format jpg $1.html $1.jpg
\rm $1.fa $1.aln
mv $1.jpg images
mv $1.xml  $1.html $1.tfa xfiles
