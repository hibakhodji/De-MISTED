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

# This function creates the simplified alignment of the
# regular ClustalW alignment file.
def simplified_alignment(aln_file, seq_file, txt, num_file, files):
    
    # Set the first line of the alignment as the reference
    # The sequence of the reference will be shown in the simplified alignment
    # And will be used to determine the mismatches
    read_ref_file = open("1_aln.txt", "r")
    ref_file = read_ref_file.readlines()
    
    num_aln = len(aln_file) + 1     # Number of sequences in alignment
    len_aln = len(ref_file[0])      # Length of the alignment

    ref = ""   # The reference will be append in this string

    # Append the reference to a string, this way the reference can easily be written to the HTML file
    # And there can be checked if the reference contains a deletion and make this better visible in the
    # alignment file
    for length in range(0,len(ref_file[0])):
        # If the reference contains a deletion, append a * on this spot in the reference sequence
        if ref_file[0][length] == "-":
            ref += "*"
        # Otherwise append the nucleotide from that location
        else:
            ref += ref_file[0][length]
            
    # Write the reference sequence to the txt file if this is requested by the user
    # If the sequence names are too long, these will be shortened to 12 characters
    # This way the alignment parts from the different sequences will align nicely
    if int(txt) == 1:
        seq = seq_file[0]
        if len(seq) == 12:
            pass
        elif len(seq) < 12:
            for length in range(len(seq), 12):
                seq += " "
        elif len(seq_file[0]) > 13:
            seq = seq_file[0][0:12]
        os.system("""printf -- "%s\t%s   %s" >> simplified_aln.txt""" %(seq.replace("\n", ""), ref.replace("\n", ""), num_file[0]))
            
    # Write the reference sequence to the HTML file. 
    ref += "<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>" %("200","Courier", num_file[0])
    HTML.write("\t\t<tr>\n\t\t\t<td><br/></td>\n\t\t</tr>\n\t\t<tr>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t</tr>\n" %("150", "Courier", seq_file[0], "500", "Courier", ref))
    
    # Check if the sequences are the same as the reference
    for sample in range (2, num_aln):
        i = 0   # Iteration
        row = ""    # Simplified alignment will be appended here
        
        # Compare the two sequences, if there are differences the positions of these differences are written to pos.txt
        os.system("""cmp -b -l 1_aln.txt %s_aln.txt | awk '{ print $1 }' > pos.txt""" %(sample))

        # Open the positions file
        read_pos_file = open("pos.txt", "r")
        pos_file = read_pos_file.readlines()

        # If there are no differences the positions file will be empty and the parameter 'same' will be written to the simplified alignment
        if len(pos_file) == 0:
            same = ""   # If the reference and the sequence are the same this parameter will be printed in the alignment
            # Fill alignment with "-" to show that sequence and reference are the same
            # Only if the reference contains a deletion, the deletion will also be written to the sequence
            for x in range(0, (len_aln - 1)):
                if ref[x] == "*":
                    same += "*"
                else:
                    same += "-"

            # If the user requested a txt output, write the sample name and sequence to the txt file
            # If the sequence name is bigger than 12 characters, the name will be shortened to 12 characters
            # If the sequence name is smaller than 12 characters, spaces are appended
            # This will lead to a nicely aligned multiple sequence alignment
            if int(txt) == 1:
                seq = seq_file[sample -1]
                if len(seq) == 12:
                    pass
                elif len(seq) < 12:
                    for length in range(len(seq), 12):
                        seq += " "
                elif len(seq_file[sample -1]) > 13:
                    seq = seq_file[sample -1][0:12]
                os.system("""printf -- "%s\t%s   %s" >> simplified_aln.txt""" %(seq.replace("\n", ""), same, num_file[sample-1]))

            # Write the sequence to the HTML file, these will also be append to the table
            same += "<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>" %("200","Courier", num_file[sample-1])
            HTML.write("\t\t<tr>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\n\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t</tr>\n" %("150", "Courier", seq_file[sample -1], "500", "Courier", same))

        # If there are differences between the reference and the sample sequence
        # the differences will be shown in the simplified alignment    
        else:
            
            # Therefore the file with the sequence must be opened
            read_sample_file = open("%s_aln.txt" %(sample), "r")
            sample_file = read_sample_file.readlines()

            # Check for every position if the sequence and the reference are the same
            for nuc in range (1, len_aln):

                # If they differences are 'empty' append "-" or a "*" if there is a deletion in the reference sequence
                # The differences are empty when the loop already detected the amount of differences that was found
                if i >= len(pos_file):
                        if ref[nuc - 1] == "*":
                            row += "*"
                        else:
                            row += "-"

                # Check if there is a difference between the sequence and the reference
                else:

                    # If the position of the difference is called, check whether it is a deletion or a mismatch
                    if nuc == int(pos_file[i]):
                        
                        # If it is a deletion, print "*"
                        if sample_file[0][nuc - 1] == "-":
                            row += "*"
                            i += 1  # Count the differences that are implemented

                        # If it is a mismatch, print the nucleotide or amino acid
                        else:
                            row += sample_file[0][nuc - 1]
                            i += 1  # Count the differences that are implemented
                            
                    # If there is no difference on this position print "-" or "*" depending on the reference sequence
                    else:
                        if ref[nuc - 1] == "*":
                            row += "*"
                        else:
                            row += "-"
                        
            # Write the simplified alignment to a txt file if requested            
            # With the same limitation of 12 characters for the sequence name
            if int(txt) == 1:
                seq = seq_file[sample -1]
                if len(seq) == 12:
                    pass
                elif len(seq) < 12:
                    for length in range(len(seq), 12):
                        seq += " "
                elif len(seq_file[sample -1]) > 13:
                    seq = seq_file[sample -1][0:12]
                os.system("""printf -- "%s\t%s   %s" >> simplified_aln.txt""" %(seq.replace("\n", ""), row.replace("\n", ""), num_file[sample-1]))

            # Write the sequence to the HTML file
            row += "<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>" %("200","Courier", num_file[sample-1])
            HTML.write("\t\t<tr>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\n\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t</tr>\n" %("150", "Courier", seq_file[sample -1], "500", "Courier", row))

    # Print a newline after each part of the alignment in the txt file
    if int(txt) == 1:
        os.system("""printf "\n" >> simplified_aln.txt""")

# This function creates the two different alignment files: normal and colored
def clustalw_alignment(aln_file, num_file, enhanced, sequence, files, seq_file, cons_file, last_line):
    
    # When colored output is requested and the sequences are protein, there must be calculated
    # how much of each amino acid is on a certain position in the alignment. With DNA/RNA sequences
    # this is not needed according to the color scheme. 
    if int(enhanced) == 1 and sequence == "PROTEIN":
        # Get only the alignment part of the alignment
        os.system("""cat aln_%s.txt | awk '{ print $2 }' > full_aln_%s.txt""" %(files, files))
        read_full_aln_file = open("full_aln_%s.txt" %(files), "r")
        full_aln_file = read_full_aln_file.readlines()

        # list with all the amino acids for counting of the amount of amino acids for every position
        amino_acids = ["G", "C", "A", "V", "L", "I", "M", "F", "W", "P", "S", "T", "Y", "N", "Q", "D", "E", "K", "R", "H"]
        percentages = {}    # Dictionary where percentages of amino acids on a specific location will be appended

        # Create a list with all the aminoacids on a specific position in one string
        # (instead of the whole sequence of one entry)(whole column instead of row)
        zipped = list(zip(*full_aln_file))

        # Per position count the amount of amino acids on that specific location
        for x in range(0, len(zipped)):
            # Also count the gaps and empty spots, these may not be included in the
            # calculation of the percentages per specific location
            gap_count = zipped[x].count("-") + zipped[x].count(" ")
            percentages[x] = {}  # Create a dict in the dict for a specific location

            # For every amino acid, count the amount on the specific location
            # And save this in the percentages dictionary
            for y in range(0, len(amino_acids)):
                zip_count = zipped[x].count(amino_acids[y])
                zip_percentage = float(zip_count) / float((len(zipped[x])) - gap_count) * float(100)
                percentages[x][amino_acids[y]] = zip_percentage
                
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
        row += "<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>" %("200","Courier", num_file[sample-1])

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
                    calc_1 = percentages[nuc -1]["W"] + percentages[nuc -1]["L"] + percentages[nuc -1]["V"] + percentages[nuc -1]["I"] + percentages[nuc -1]["M"] + percentages[nuc -1]["A"] + percentages[nuc -1]["F"] + percentages[nuc -1]["C"] + percentages[nuc -1]["Y"] + percentages[nuc -1]["H"]+ percentages[nuc -1]["P"]
                    if sample_file[0][nuc - 1] == "P":
                        color = "background-color:#F7FE2E;"
                    elif sample_file[0][nuc - 1] == "G":
                        color = "background-color:#FF8000;"
                    elif sample_file[0][nuc - 1] == "C":
                        if percentages[nuc -1]["C"] >= 85:
                            color = "background-color:#F781BE;"
                        elif calc_1 >= 60:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["P"] >= 50:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["W"] >= 85 or percentages[nuc - 1]["Y"] >= 85 or percentages[nuc - 1]["A"] >= 85 or percentages[nuc - 1]["S"] >= 85 or percentages[nuc - 1]["P"] >= 85 or percentages[nuc - 1]["F"] >= 85 or percentages[nuc - 1]["H"] >= 85 or percentages[nuc - 1]["I"] >= 85 or percentages[nuc - 1]["L"] >= 85 or percentages[nuc - 1]["M"] >= 85 or percentages[nuc - 1]["V"] >= 85:
                            color = "background-color:#2E9AFE;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "A":
                        if calc_1 >= 60:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["P"] >= 50:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["S"] >= 85 or percentages[nuc - 1]["T"] >= 85 or percentages[nuc - 1]["G"] >= 85:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["W"] >= 85 or percentages[nuc - 1]["Y"] >= 85 or percentages[nuc - 1]["A"] >= 85 or percentages[nuc - 1]["C"] >= 85 or percentages[nuc - 1]["P"] >= 85 or percentages[nuc - 1]["F"] >= 85 or percentages[nuc - 1]["H"] >= 85 or percentages[nuc - 1]["I"] >= 85 or percentages[nuc - 1]["L"] >= 85 or percentages[nuc - 1]["M"] >= 85 or percentages[nuc - 1]["V"] >= 85:
                            color = "background-color:#2E9AFE;"
                        elif (percentages[nuc - 1]["T"] + percentages[nuc - 1]["S"]) >= 50:
                            color = "background-color:#2E9AFE;" 
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "I" or sample_file[0][nuc - 1] == "L" or sample_file[0][nuc - 1] == "M" or sample_file[0][nuc - 1] == "F" or sample_file[0][nuc - 1] == "W" or sample_file[0][nuc - 1] == "V":
                        if calc_1 >= 60:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["P"] >= 50:
                            color = "background-color:#2E9AFE;"
                        elif percentages[nuc - 1]["W"] >= 85 or percentages[nuc - 1]["Y"] >= 85 or percentages[nuc - 1]["A"] >= 85 or percentages[nuc - 1]["C"] >= 85 or percentages[nuc - 1]["P"] >= 85 or percentages[nuc - 1]["F"] >= 85 or percentages[nuc - 1]["H"] >= 85 or percentages[nuc - 1]["I"] >= 85 or percentages[nuc - 1]["L"] >= 85 or percentages[nuc - 1]["M"] >= 85 or percentages[nuc - 1]["V"] >= 85:
                            color = "background-color:#2E9AFE;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "R" or sample_file[0][nuc - 1] == "K":
                        if (percentages[nuc - 1]["K"] + percentages[nuc - 1]["R"]) >= 60:
                            color = "background-color:#FE2E2E;"
                        elif percentages[nuc - 1]["K"] >= 85 or percentages[nuc - 1]["R"] >= 85 or percentages[nuc - 1]["Q"] >= 85:
                            color = "background-color:#FE2E2E;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "Q":
                        if (percentages[nuc - 1]["K"] + percentages[nuc - 1]["R"]) >= 60:
                            color = "background-color:#2EFE2E;"
                        elif (percentages[nuc - 1]["Q"] + percentages[nuc - 1]["E"]) >= 50:
                            color = "background-color:#2EFE2E;"
                        elif percentages[nuc - 1]["Q"] >= 85 or percentages[nuc - 1]["E"] >= 85 or percentages[nuc - 1]["K"] >= 85 or percentages[nuc - 1]["R"] >= 85:
                             color = "background-color:#2EFE2E;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "E":
                        if (percentages[nuc - 1]["E"] + percentages[nuc - 1]["D"]) >= 50:
                            color = "background-color:#D358F7;"
                        elif (percentages[nuc - 1]["Q"] + percentages[nuc - 1]["E"]) >= 50:
                            color = "background-color:#D358F7;"
                        elif percentages[nuc - 1]["Q"] >= 85 or percentages[nuc - 1]["E"] >= 85 or percentages[nuc - 1]["D"] >= 85:
                            color = "background-color:#D358F7;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "D":
                        if (percentages[nuc - 1]["E"] + percentages[nuc - 1]["D"]) >= 50:
                            color = "background-color:#D358F7;"
                        elif percentages[nuc - 1]["N"] >= 50:
                            color = "background-color:#D358F7;"
                        elif percentages[nuc - 1]["N"] >= 85 or percentages[nuc - 1]["E"] >= 85 or percentages[nuc - 1]["D"] >= 85:
                            color = "background-color:#D358F7;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "H" or sample_file[0][nuc - 1] == "Y":
                        if calc_1 >= 60:
                            color = "background-color:#01DFA5;"
                        elif percentages[nuc - 1]["P"] >= 50:
                            color = "background-color:#01DFA5;"
                        elif percentages[nuc - 1]["W"] >= 85 or percentages[nuc - 1]["Y"] >= 85 or percentages[nuc - 1]["A"] >= 85 or percentages[nuc - 1]["C"] >= 85 or percentages[nuc - 1]["P"] >= 85 or percentages[nuc - 1]["F"] >= 85 or percentages[nuc - 1]["H"] >= 85 or percentages[nuc - 1]["I"] >= 85 or percentages[nuc - 1]["L"] >= 85 or percentages[nuc - 1]["M"] >= 85 or percentages[nuc - 1]["V"] >= 85:
                            color = "background-color:#01DFA5;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "T":
                        if calc_1 >= 60:
                            color = "background-color:#2EFE2E;"
                        elif (percentages[nuc - 1]["T"] + percentages[nuc - 1]["S"]) >= 50:
                            color = "background-color:#2EFE2E;"
                        elif percentages[nuc - 1]["S"] >= 85 or percentages[nuc - 1]["T"] >= 85:
                            color = "background-color:#2EFE2E;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "S":
                        if calc_1 >= 80:
                            color = "background-color:#2EFE2E;"
                        elif (percentages[nuc - 1]["T"] + percentages[nuc - 1]["S"]) >= 50:
                            color = "background-color:#2EFE2E;"
                        elif percentages[nuc - 1]["S"] >= 85 or percentages[nuc - 1]["T"] >= 85:
                            color = "background-color:#2EFE2E;"
                        else:
                            color = "background-color:white;"
                    elif sample_file[0][nuc - 1] == "N":
                        if percentages[nuc - 1]["N"] >= 50:
                            color = "background-color:#2EFE2E;"
                        elif percentages[nuc - 1]["N"] >= 85 or percentages[nuc - 1]["D"] >= 85:
                            color = "background-color:#2EFE2E;"
                        else:
                            color = "background-color:white;"
                        
                row1 += "<span style=%s>%s</span>" %(color, sample_file[0][nuc - 1]) # Append backgroundcolor for each nucleotide/aminoacid
            row1 += "<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>" %("200", "Courier", num_file[sample-1])       
                        
        # Write the sequence to the HTML file(s)
        HTML_cw.write("\t\t<tr>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\n\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t</tr>\n" %("150", "Courier", seq_file[sample -1], "500", "Courier", row))
        if int(enhanced) == 1:
            HTML_color.write("\t\t<tr>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\t<td WIDTH=%s>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\n\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t</tr>\n" %("150", "Courier", seq_file[sample -1],"500", "Courier", row1))

        # Remove some temporary files
        os.system("""rm %s.txt %s_aln.txt""" %(sample,sample))
        
    # Count the amount of spaces in between the alignment and the sequence name
    spaces = last_line[0][len(seq_file[sample -1]):(len(seq_file[sample -1]) + len_aln)].count("@")
    # Write the conservation line to the HTML file
    HTML_cw.write("\t\t<tr>\n\t\t\t<td>""</td>\n\t\t\n\t\t\t<td>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\n\t\t\t<td>""</td>\n\t\t" %("Courier", cons_file[0][(len(seq_file[sample -1]))+ spaces:].replace("@", "\b ")))

    if int(enhanced) == 1:
        # Write the conservation line to the HTML file
        HTML_color.write("\t\t<tr>\n\t\t\t<td>""</td>\n\t\t\n\t\t\t<td>\n\t\t\t\t<FONT FACE=%s SIZE=2>\n\t\t\t\t%s\t\t\t\t</FONT>\n\t\t\t</td>\n\t\t\n\t\t\t<td>""</td>\n\t\t" %("Courier", cons_file[0][(len(seq_file[sample -1]))+ spaces:].replace("@", "\b ")))
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

        # Get the simplified alignment
        simplified_alignment(aln_file, seq_file, txt, num_file, files)
        # Get the ClustalW alignment in a HTML file and if specified in colored HTML file
        clustalw_alignment(aln_file, num_file, enhanced, sequence, files, seq_file, cons_file, last_line)

        # Remove some temporary files        
        os.system("""rm rows_aln_%s.txt cons_%s.txt""" %(files, files))

        # Write an empty row to the alignment HTML files
        HTML_cw.write("\t\t<tr>\n\t\t\t<td><br/></td>\n\t\t\n\t\t\t<td><br/></td>\n\t\t\n\t\t\t<td><br/></td>\n\t\t")
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

    # HTML file for regular ClustalW alignment
    HTML_cw = open("%sclustalw_%s_aln.html" %(file_path, prefix), "w")
    HTML_cw.write("<HTML>\n\n<HEAD>\n\t<TITLE>\n\tClustalW alignment\n\t</TITLE>\n</HEAD>\n\n<BODY>\n"
                       "\t<H3><FONT face=%s>\nClustalW alignment: %s\n\t</FONT></H3>\n\n\t"
                       "<BR>\n\n\t<table style=%s>\n\t\t<tr>\n" %("Courier", alignment, "WIDTH=800"))

    # HTML file for colored ClustalW alignment
    if int(enhanced) == 1:
        HTML_color = open("%scolor_%s_aln.html" %(file_path, prefix), "w")
        HTML_color.write("<HTML>\n\n<HEAD>\n\t<TITLE>\n\tClustalW alignment\n\t</TITLE>\n</HEAD>\n\n<BODY>\n"
                       "\t<H3><FONT face=%s>\nColored ClustalW alignment: %s\n\t</FONT></H3>\n\n\t"
                       "<BR>\n\n\t<table style=%s>\n\t\t<tr>\n" %("Courier", alignment, "WIDTH=800"))

    # HTML file for simplified ClustalW alignment
    HTML = open("%ssimplified_%s_aln.html" %(file_path, prefix), "w")
    HTML.write("<HTML>\n\n<HEAD>\n\t<TITLE>\n\tSimplified alignment\n\t</TITLE>\n</HEAD>\n\n<BODY>\n"
                   "\t<H3><FONT face=%s>\nSimplified ClustalW alignment: %s\n\t</FONT></H3>\n\n\t"
                   "<BR>\n\n\t<table style=%s>\n" %("Courier", alignment, "WIDTH=800"))

    # When the user would like a txt file, also write a header for that file
    if int(txt) == 1:
        os.system("""printf -- "Simplified ClustalW alignment: %s\n\n" >> simplified_aln.txt""" %(alignment))

    # Go to function get_alignment
    get_alignment(alignment, enhanced, txt, sequence)

    # Remove the last temporary files
    os.system("""rm aln_* line_pos.txt pos.txt num_aln_* seq_aln.txt cons_0.txt last_*.txt""")

    # Close the table, body, etc. for the different HTML files
    HTML_cw.write("\n\t</table>\n</BODY>\n</HTML>")
    HTML_cw.close()

    HTML.write("\n\t</table>\n</BODY>\n</HTML>")
    HTML.close()

    if int(enhanced) == 1:
        HTML_color.write("\n\t</table>\n</BODY>\n</HTML>")
        HTML_color.close()

    # Move the simplified alignment txt to a file with more specific name and the right location
    if int(txt) == 1:
        os.system("""mv simplified_aln.txt %ssimplified_%s_aln.txt""" %(file_path, prefix))
    
main()
