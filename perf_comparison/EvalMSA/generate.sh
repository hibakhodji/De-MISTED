biolo/clustalw $1.xml -convert -output=fasta
mv $1.tfa $1.fa
EvalMSA $1.fa blosum62
python3 check.py $1_results.txt
rm $1.fa
