#!/bin/bash

# This scripts fetches all the parameters from the commandline, checks these and
# give them to the different programs and scripts. This script is part of ADOMA. 


# Copyright (C) 2014  	Dionne Zaal

# ADOMA is free software: you can redistribute it and/or modify it under the terms 
# of the GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or (at your option) any later version. 

# ADOMA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
# PURPOSE.  See the GNU General Public License for more details. 

# You should have received a copy of the GNU General Public License along 
# with ADOMA.  If not, see <http://www.gnu.org/licenses/>



# If only the program is called, output where the user can find the help document 
if [ $# -eq 0 ] 
then
    echo 
    echo For help: type --help or -h after the command
    echo
    exit
fi

# If the user asks for the help file by typing -h or --help display the help file
if [ $1 == "-h" ] || [ $1 == "--help" ]
then 
    echo
    echo "ADOMA: Alternative Display Of Multiple Alignment"
    echo  
    echo "Usage:     ./adoma.sh -i [input_files] -p [prefix] [extra_options]"
    echo 
    echo "There are two mandatory options:"
    echo "-p         Define the prefix for the output files"
    echo "-i         Input files in fasta format"
    echo "           More than one file can be put in for alignment"
    echo "           For example: -i sequence_1.fasta sequence_2.fasta" 
    echo
    echo "Extra options for ADOMA are: "
    echo "-dna       Define that the input sequences are DNA [default]"
    echo "-prot      Define that the input sequences are protein"
    echo "-txt       Define for a txt output of the simplified alignment"
    echo "-color     Define for a colored ClustalW alignment in HTML"
    echo "-clustalw  Options for the ClustalW alignment can be defined here."
    echo "           For example for output with length of sequences:" 
    echo "           -clustalw -seqnos=ON"
    echo
    echo "For more details check the README file or the manual in the"
    echo "adoma-[version] folder"
    echo
    exit
fi 

# If input is not -h, --help or empty: start with defining the parameters
iteration=0     # Counts the amount of arguments from the command line input
m_parameter=0   # Counts the mandatory parameters from the command line input
parameter=1     # Counts all the parameters from the command line input
TYPE="DNA"      # Contains the type of sequences: DNA and PROTEIN. Default is DNA. 
TXT=0           # Checks if the user would like a txt output
COLOR=0         # Checks if the user would like a colored HTML output
type_DNA=0      # Counts amount of times type DNA is defined
type_PROT=0     # Counts amount of times type protein is defined

# Check all the arguments that are defined in the commandline
for i in "$@"
do 
    iteration=$(($iteration+1)) # Appends one every time the loop starts over
    old=$(($parameter))  # Remembers the amount of parameters before checking for parameters

    # Define the prefix parameter
    if [ "$i" == "-p" ]
    then
        PREFIX=$(echo ${@:iteration+1:1} | tr "/" " "| awk '{ print $NF }')  # Determine prefix from the input
        file_path=$(echo ${@:iteration+1:1} | sed s/$PREFIX/""/)  # Determine destination for output files, if not defined, parameter will be empty

        m_parameter=$(($m_parameter+1)) # Count +1 for the mandatory parameters
        parameter=$(($parameter+1)) #  Count +1 for the parameters
    fi

    # If the parameter -dna is defined by the user, set the type of sequences to DNA
    if [ "$i" == "-dna" ]
    then 
        TYPE="DNA"
        parameter=$(($parameter+1))   # Count +1 for the parameters
        type_DNA=$(($type_DNA+1))  # Count +1 for amount of times DNA is defined
    fi

    # If the parameter -prot is defined by the user, set the type of sequences to PROTEIN
    if [ "$i" == "-prot" ]
    then 
        TYPE="PROTEIN"
        parameter=$(($parameter+1))  # Count +1 for the parameters
        type_PROT=$(($type_PROT+1))  # Count +1 for amount of times protein is defined
    fi

    # If the user defined -txt with the arguments in the command line, set TXT to 1
    if [ "$i" == "-txt" ] 
    then 
        TXT=1
        parameter=$(($parameter+1))   # Count +1 for the parameters
    fi

    # If the users specified the -color parameter, set COLOR to 1
    if [ "$i" == "-color" ]
    then 
        COLOR=1
        parameter=$(($parameter+1))  # Count +1 for the parameters
    fi

    # If the parameter input is defined, get the filenames and merged them into one fasta file
    if [ "$i" == "-i" ]
    then
        m_parameter=$(($m_parameter+1))  # Count +1 for the mandatory parameters

        count_par=0  # In this variable the parameters before the input parameter are counted
        next_par=0   # In this variable contains the location of the first parameter after the input parameter

        # Loop through the arguments for the commandline to get the filenames
        # This is necessary because the amount of input files is not yet defined 
        for y in "$@"
        do  
            count_par=$(($count_par+1))
 
            # If the amount of parameters before the input parameter is bigger than the parameter (we are behind
            # the input parameter now), set the location of the first parameter behind the input parameter
            if [ $count_par -ge $parameter ]
            then
                # Check if the argument in the commandline is a parameter that can be defined with this program
                if [ "$y" == "-dna" ] || [ "$y" == "-prot" ] || [ "$y" == "-txt" ] || [ "$y" == "-color" ] || [ "$y" == "-p" ] || [ "$y" == "-clustalw" ]
                then
                    next_par=$count_par # If it is a parameter that can be defined, save the location
                    break  # Exit the for loop, no more looping is needed, location is determined
                fi
            fi
        done
  
        # If no parameter is found behind the input parameter (because it is the last one 
        # in the arguments): set the next parameter to the end of the row of arguments. 
        if [ $next_par -eq 0 ]
        then
            next_par=$(($#+1))
        fi   

        parameter=$(($parameter+1))  # +1 for the input parameter variable
        # Determine the amount of files that are at the input by substracting amount
        # of parameters before the input parameter with first parameter behind the input parameter
        num_files=$(($next_par-$parameter))  # Number of files defined by the user
        input_par=$parameter  # Remember the location of the input parameter for later use
        cat ${@:$parameter:$num_files} > Merged.fa # Merged the fasta files together to one file
    fi

    # If the user defines -clustalw at input, append the options for clustalw to a string
    if [ "$i" == "-clustalw" ]
    then
        cw_options=""  # Append defined ClustalW options in this string
        count_par1=0   # Count amount of parameters before the clustalw parameter
        next_par1=0    # Set the location for the first parameter behind the clustalw options

        # Loop through the commandline arguments to get the clustalw input options
        # This is necessary because the amount of options is not defined
        for x in "$@"
        do 
           count_par1=$(($count_par1+1))

           # If we looped through the parameters before the clustalw, check if the argument is a defined parameter for this program
           if [ $count_par1 -ge $parameter ]
           then 
               if [ "$x" == "-dna" ] || [ "$x" == "-prot" ] || [ "$x" == "-txt" ] || [ "$x" == "-color" ] || [ "$x" == "-p" ] || [ "$x" == "-i" ]
               then 
                   next_par1=$count_par1 # If it is a defined parameter for this program, set the location
                   break  # Exit the for loop because the location for the first parameter behind the clustalw parameter is defined
               fi
           fi
        done

        # If the location for the next parameter is 0, the clustalw parameter is the last in the commandline arguments
        # So the next parameter then has to be set as the last argument in the commandline
        if [ $next_par1 -eq 0 ]
        then
            next_par1=$(($#+1))
        fi

        # Add 1 to the parameter variable for the clustalw argument
        parameter=$(($parameter+1))
        # Define the amount of options that are given by the user for clustalw by substracting the amount
        # of parameters before the clustalw parameter from the first parameter behind the clustalw parameter
        num_options=$(($next_par1-$parameter))

	count_par2=0  # Count amount of parameters before the clustalw parameter
        count_cw_par=0  # Count amount of cw options that are appended to a string
       
        # Once again loop through the commandline arguments to get the options for clustalw
        # These can now be appended to a string because the amount of parameters is known
        for z in "$@"
        do
          count_par2=$(($count_par2+1))  # Add one for a parameter before the clustalw parameter

          # If amount of parameters is smaller than the first parameter behind the clustalw parameter
          # and bigger than the location of the clustalw parameter, append options for clustalw to a string
          if [ $count_par2 -lt $next_par1 ] && [ $count_par2 -ge $parameter ] 
          then
              # If the amount of options appended to the cw options string equals the amount of options defined
              # Exit the for loop because all the options are defined
              if [ $count_cw_par -eq $num_options ]
              then 
                  break
              else
                  cw_options+=" $z" # Append the user defined clustalw options to a string
                  count=$(($count_cw_par+1)) # Add one for the appended clustalw option
              fi
          fi
        done 
    fi

    # If no parameter was found, append 1 to the parameter variable
    if [ $old -eq $parameter ] 
    then
        parameter=$(($parameter+1))
    fi

done

# Checks if all the mandatory parameters are defined by the user.
# If not all the mandatory parameters are defined the program quits.
if [ $m_parameter -lt  2 ]
then 
    echo
    echo 'Not all the mandatory parameters are assigned (-p and -i)'
    echo 'The help file can be assessed by typing -h or --help after the command'
    echo
    exit
fi

# If both DNA type and PROTEIN type is defined by the user: 
# Print a error message and quit. 
if [ $type_DNA -ge 1 ] && [ $type_PROT -ge 1 ]
then 
     echo 
     echo 'More than one type of sequences is defined (-dna and -prot)'
     echo 'Sequences can only be one of the two types'
     echo 'Try again with only one type of sequences'
     echo
     exit
fi

# Output processes to user
echo
echo 'ADOMA: Alternative Display Of Multiple Alignment'
echo
echo '-> Merging fasta files into one fasta file:' "$PREFIX".fa
echo

mv Merged.fa $file_path"$PREFIX".fa   # Move the merged fasta file to a fasta file with the specified prefix

# Write defined parameters to the logfile
echo Input files: ${@:$input_par:$num_files} were written to one fasta file: $file_path"$PREFIX".fa > "$file_path"logfile.txt
echo Prefix is defined as: "$PREFIX", type of sequences is defined as: "$TYPE" >> "$file_path"logfile.txt 

# Write the clustalw options to the logfile if they were defined to the logfile
if [ ${#cw_options} -ge 1 ]
then
    echo There are extra ClustalW options defined: $cw_options >> "$file_path"logfile.txt
fi

# Write that a txt output is requested if the user defines it to the logfile
if [ $TXT -eq 1 ]
then
    echo Extra txt output for the simplified alignment is requested >> "$file_path"logfile.txt
fi

# Write that a colored output is requested if the user defines it to the logfile
if [ $COLOR -eq 1 ]
then
    echo Extra colored output for the Clustal W alignment is requested >> "$file_path"logfile.txt
fi

# Output processes to user and logfile
echo '-> Performing ClustalW alignment'
echo >> "$file_path"logfile.txt
echo 'ClustalW alignment:' >> "$file_path"logfile.txt

# Performing clustalw alignment.
clustalw -infile=$file_path"$PREFIX".fa -align $cw_options -type="$TYPE" -quiet &>>"$file_path"logfile.txt
echo

# Determine the path where ADOMA was called, so the python script can be called properly
scriptpath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Output processes to user
echo '-> Converting alignment'
echo 'Convert ClustalW alignment to different output files with python script: convert_aln.py' >> "$file_path"logfile.txt

# Convert the alignment with the convert_aln python script 
python $scriptpath/convert_aln.py $file_path"$PREFIX".aln "$PREFIX" "$COLOR" "$TYPE" "$TXT" "$file_path" &>>"$file_path"logfile.txt
echo

# Output processes to user
echo ADOMA is done.
echo If the output is not as expected, check the logfile.txt for errors that
echo might have occured during the program and check the manual for troubleshooting.
echo
