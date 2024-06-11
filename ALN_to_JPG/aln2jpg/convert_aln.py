"""
--------------------------------------------------------------------------------

With this script a ClustalW alignment file is converted to a html file.
If defined, their will also be a colored html file from the alignment.
The alignent from ClustalW is also converted to a simplified alignment.
This output can be in html and txt. The simplified output will come in handy when
there is a lot of homology in the aligned sequences. Single mismatches can then
be identified a lot easier.

This script is part of ADOMA. 
Six arguments are given from adoma.sh to convert_aln.py:
- ClustalW alignment file (.aln)
- Defined prefix for the output file ([string])
- Type of sequences (DNA or protein)
- If the user wants a colored output (0 or 1)
- If the user wants a txt for the simplified output (0 or 1)
- The path to the files

--------------------------------------------------------------------------------

Copyright (C) 2014  	Dionne Zaal

ADOMA is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. 

ADOMA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. 

You should have received a copy of the GNU General Public License along with
ADOMA.  If not, see <http://www.gnu.org/licenses/>

--------------------------------------------------------------------------------
"""

import sys  # Import sys for input via commandline
import os   # Import the operating system


# This function creates the two different alignment files: normal and colored
def clustalw_alignment(aln_file, num_file, enhanced, sequence, files, seq_file, cons_file, last_line):
    
                
    num_aln = len(aln_file) + 1     # Number of sequences in alignment

    # Create a row for each sequence in the alignment
    for sample in range (1, num_aln):
        row = ""  # Append the sequence in this variable
            
        # Therefore the file with the sequence must be opened
        read_sample_file = open("%s_aln.txt" %(sample), "r")
        sample_file = read_sample_file.readlines()
        len_aln = len(sample_file[0])      # Length of the alignment

        # Append the normal sequence to the row for creating the regular HTML file
        row += sample_file[0]
        row += "<td WIDTH=%s>\n\t\t\t\t%s\t\t\t\t\n\t\t\t</td>" %("200", num_file[sample-1])

        # If colored output is requested, give each nucleotide/amino acids a color
        # according to the clustalw color scheme from:
        # https://www.cgl.ucsf.edu/chimera/docs/ContributedSoftware/multalignviewer/colprot.par
        if int(enhanced) == 1:
            row1 = ""   # Sequence will be appended here
            for nuc in range (1, len_aln):
                color = "background-color:white;"

                # Colors if the sequences are DNA
                if sequence == "DNA":
                    if sample_file[0][nuc - 1] == "A":
                        color = "background-color:#FE2E2E;"
                    elif sample_file[0][nuc - 1] == "C":
                        color = "background-color:#0080FF;"
                    elif sample_file[0][nuc - 1] == "T" or sample_file[0][nuc - 1] == "U":
                        color = "background-color:#01DF01;"
                    elif sample_file[0][nuc - 1] == "G":
                        color = "background-color:#FF8000;"

                # Colors if the sequences are protein
                if sequence == "PROTEIN":
                    if sample_file[0][nuc - 1] == "A" or sample_file[0][nuc - 1] == "C" or sample_file[0][nuc - 1] == "S" or sample_file[0][nuc - 1] == "T":
                        color = "background-color:#CC66FF;"
                    elif sample_file[0][nuc - 1] == "R" or sample_file[0][nuc - 1] == "K":
                        color = "background-color:#6666FF;"
                    elif sample_file[0][nuc - 1] == "P":
                        color = "background-color:#08FAF3" #RD9D8D4
                    elif sample_file[0][nuc - 1] == "N":
                        color = "background-color:#FBF484;" #DCF7F7
                    elif sample_file[0][nuc - 1] == "D" or sample_file[0][nuc - 1] == "Q" or sample_file[0][nuc - 1] == "E":
                        color = "background-color:#009933;"
                    elif sample_file[0][nuc - 1] == "G":
                        color = "background-color:#FF9900;"
                    elif sample_file[0][nuc - 1] == "H":
                        color = "background-color:#F50909;" #D7F0F0
                    elif sample_file[0][nuc - 1] == "I" or sample_file[0][nuc - 1] == "L" or sample_file[0][nuc - 1] == "M" or sample_file[0][nuc - 1] == "V":
                        color = "background-color:#FF99FF;"
                    elif sample_file[0][nuc - 1] == "F" or sample_file[0][nuc - 1] == "W" or sample_file[0][nuc - 1] == "Y":
                        color = "background-color:#FF6666;"
                    elif sample_file[0][nuc - 1] == "B" or sample_file[0][nuc - 1] == "Z" or sample_file[0][nuc - 1] == "X":
                        color = "background-color:#FFFFFF;"
                    else:
                        color = "background-color:white;"
                        
                row1 += "<span style=%s>%s</span>" %(color, sample_file[0][nuc - 1]) # Append backgroundcolor for each nucleotide/aminoacid
                        
        if int(enhanced) == 1:
            HTML_color.write("\t\t<tr>\n\t\t\t<td WIDTH=%s>%s</td>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t%s\n\t\t\t\t\n\t\t\t</td>\n\t\t</tr>\n" %("150", seq_file[sample -1],"500", row1))

        # Remove some temporary files
        os.system("""rm %s.txt %s_aln.txt""" %(sample,sample))
        
    # Count the amount of spaces in between the alignment and the sequence name
    spaces = last_line[0][len(seq_file[sample -1]):(len(seq_file[sample -1]) + len_aln)].count("@")

    if int(enhanced) == 1:
        # Write the conservation line to the HTML file
        HTML_color.write("\t\t<tr>\n\t\t\t<td>""</td>\n\t\t\n\t\t\t<td>\n\t\t\t\t\n\t\t\t\t%s\t\t\t\t\n\t\t\t</td>\n\t\t\n\t\t\t<td>""</td>\n\t\t" %(cons_file[0][(len(seq_file[sample -1]))+ spaces:].replace("@", "\b ")))
        if sequence == "PROTEIN":
            # One file is only created when colored output is requested and the sequences are protein, also removes this file  
            os.system("""rm full_aln_%s.txt""" %(files))

# This function gets information from the alignment file
# The functions simplified_alignment and clustalw_alignment need this information 
def get_alignment(alignment, enhanced, txt, sequence):
    # Get the lines in the alignment that are empty or contain *
    os.system("""cat %s | grep -vn [A-Za-z] | tr ":" " " | awk '{ print $1 }' > temp1.txt""" %(alignment))
    # There are two (empty) lines between the alignments, but only one must be reported for the splitting of the alignments
    os.system("""awk 'NR % 2' temp1.txt > line_pos.txt """)
    # Remove temporary file
    os.system("""rm temp1.txt""")

    # Open file with positions where the alignments are located
    read_pos_file = open("line_pos.txt", "r")
    pos_file = read_pos_file.readlines()

    # Determine how many sequences where aligned to eachother 
    num_seq = int(pos_file[1]) - int(pos_file[0]) - 2

    # Create file for each alignment part with the obtained positions
    for x in range(0, len(pos_file)):
        os.system("""cat %s | head -n %s | tail -n %s > aln_%s.txt""" %(alignment,(int(pos_file[x]) -1), int(num_seq), x))
        # Get the alignment conservation for the ClustalW alignment
        os.system("""cat %s | head -n %s | tail -n 1 | tr " " "@" > cons_%s.txt""" %(alignment, int(pos_file[x]), x))
        # Get the last row of alignment
        os.system("""cat %s | head -n %s | tail -n 2 | head -n 1 | tr " " "@" > last_%s.txt""" %(alignment, int(pos_file[x]), x))

    # Create file with names of the sequences that where aligned
    os.system("""cat aln_1.txt | grep -n [A-Za-z\-] | tr ":" "\t" | awk '{ print $2 }' > seq_aln.txt""")
    
    # Open the file with sequence names so they can be written to the alignment file later on
    read_seq_file = open("seq_aln.txt", "r")
    seq_file = read_seq_file.readlines() 

    # Get information for each alignment part 
    for files in range(1, len(pos_file)):
        
        # Open the file with the alignment part
        read_aln_file = open("aln_%s.txt" %(files), "r")
        aln_file = read_aln_file.readlines()

        # Get all the rows from the alignment part and write every line to a different file
        os.system("""cat aln_%s.txt | grep -n [A-Za-z\-] | tr ":" "\t" > rows_aln_%s.txt""" %(files, files))
        os.system("""nawk '{ print $0 >> $1 }' rows_aln_%s.txt""" %(files))

        # Save the third colunn in the alignment file (this are for example length of sequences, can be empty)
        os.system("""cat rows_aln_%s.txt | awk '{ print $4 }' > num_aln_%s.txt""" %(files,files))
        read_num_file = open("num_aln_%s.txt" %(files), "r")
        num_file = read_num_file.readlines()

        # For each line in alignment get the alignment part (so without the sequence name) and save in a file
        for line in range(1, len(aln_file) + 1): 
            os.system("""cat %s | awk '{ print $2,$3 }' > %s.txt""" %(line,line))
            os.system("""rm %s""" %(line))  # Remove old file
            os.system("""cat %s.txt | awk '{ print $2 }' > %s_aln.txt""" %(line,line))

        # Open the alignment conservation file for this alignment part
        read_cons_file = open("cons_%s.txt" %(files), "r")
        cons_file = read_cons_file.readlines()

        # Open the last line of the alignment
        read_last_line = open("last_%s.txt" %(files), "r")
        last_line = read_last_line.readlines()

        # Get the ClustalW alignment in a HTML file and if specified in colored HTML file
        clustalw_alignment(aln_file, num_file, enhanced, sequence, files, seq_file, cons_file, last_line)

        # Remove some temporary files        
        os.system("""rm rows_aln_%s.txt cons_%s.txt""" %(files, files))

        if int(enhanced) == 1:
            HTML_color.write("\t\t<tr>\n\t\t\t<td><br/></td>\n\t\t\n\t\t\t<td><br/></td>\n\t\t\n\t\t\t<td><br/></td>\n\t\t")

# Main function, calls the different functions in the script
def main():
    # Create global variables for the different HTML files, so there can be lines written to the files
    # during the whole script. 
    global HTML_cw
    global HTML_color
    global HTML
    
    alignment = sys.argv[1] # Input via commandline of alignmentfile
    prefix = sys.argv[2]    # Input of the prefix for the files from the command line
    enhanced = sys.argv[3]  # Input if the users would like a colored html output
    sequence = sys.argv[4]  # Type of sequences (DNA or protein)
    txt = sys.argv[5]       # Input if the users would like a txt output of the simplified alignment
    file_path = sys.argv[6] # The path to the output files

    # Open two or three different HTML files for writing the output
    # Also writing the header, etc. to the HTML files


    # HTML file for colored ClustalW alignment
    if int(enhanced) == 1:
        HTML_color = open("%s%s.html" %(file_path, prefix), "w")
        HTML_color.write("<HTML>\n\n<HEAD>\n\t<TITLE>\n\tClustalW alignment\n\t</TITLE>\n\t<style>\ntable {\n"
                       "border-collapse: collapse;\n"
                       "font-family:'Courier New';\nfont-size:10px;\n"
                       "color:rgba(0, 0, 0, 0);\n" # set transparent text
                       # "color:grey;\n" # set color of text
                       "}\n" 
                       "td {\nborder: none;\n;padding: 0px;\n}\n" 
                       "</style>\n</HEAD>\n\n<BODY>\n"
                       "\t<BR>\n\t<table %s>\n" %("WIDTH=800"))

    # Go to function get_alignment
    get_alignment(alignment, enhanced, txt, sequence)

    # Remove the last temporary files
    os.system("""rm aln_* line_pos.txt pos.txt num_aln_* seq_aln.txt cons_0.txt last_*.txt""")


    if int(enhanced) == 1:
        HTML_color.write("\n\t</table>\n</BODY>\n</HTML>")
        HTML_color.close()

    
main()
