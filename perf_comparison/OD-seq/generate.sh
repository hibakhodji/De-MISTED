biolo/clustalw $1.xml -convert -output=fasta
OD-seq -i $1.tfa  -o $1
python3 check.py $1
rm $1.tfa
