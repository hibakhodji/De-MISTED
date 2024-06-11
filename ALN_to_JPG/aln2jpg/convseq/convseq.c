#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include "clustalw.h"

#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

int main(int argc,char **argv)
{
	sint i,ii,j,jj,k,kk,l,s,e,n,g,seq;
	sint t;
	sint seq1;
	char infile[FILENAMELEN+1];
	char outfile[FILENAMELEN+1];
	ALN mult_aln;
	OPT opt;
	sint nseqs;
	char name[MAXNAMES+1];
	char c;




	if(argc!=3) {
		fprintf(stdout,"Usage: %s input_aln output_aln\n",argv[0]);
		exit(1);
	}
	strcpy(infile,argv[1]);
	strcpy(outfile,argv[2]);

        init_options(&opt);

	(*opt.alnout_opt).output_clustal=TRUE;

/* read in the sequences */
	seq_input(infile,opt.explicit_type,FALSE,&mult_aln);
	if(mult_aln.nseqs<=0) {
		error("No sequences in %s\n",infile);
		exit(1);
	}
	nseqs=mult_aln.nseqs;

/* write out the sequences */
	strcpy((*opt.alnout_opt).clustal_outname, outfile);

	if(!open_alignment_output(infile,opt.alnout_opt)) exit(1);
        create_alignment_output(mult_aln,*opt.alnout_opt);
}



