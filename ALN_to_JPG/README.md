In the directory : /aln2jpg

Usage: aln2jpg.sh [input]

Input: alignment in XML format (possibly with error annotations?). 

An example alignment is available in ref.xml. 
To call the script use : aln2jpg.sh ref
Output is a JPG image in ref.jpg

Method:
The aln2jpg.sh script calls three programs :

1.	clustalw to convert alignment from XML to FASTA format
2.	adoma.sh to convert FASTA alignment to HTML
3.	wkhtmltoimage to convert HTML to JPG (or other image format if you prefer)

clustalw is developed inhouse and installed here : /biolo/clustalw

adoma.sh is free software, but was slightly modified to create a colored alignment. The script calls a python program convert_aln.py, and a C program installed in aln2jpg/convseq

wkhtmltoimage (0.12.6) is an open source precompiled binary, downloaded from https://wkhtmltopdf.org, and uses the Qt WebKit rendering engine.

