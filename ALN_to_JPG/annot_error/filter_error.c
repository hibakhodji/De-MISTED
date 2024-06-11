#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include "clustalw.h"

/* global data */

Boolean verbose;

int main(int argc, char **argv)
{
	FILE *ifd;
	int  n,i,j,k,feature;
	int start_res,end_res;
	int start_col,end_col;
        char infile[FILENAMELEN+1];
        char outfile[FILENAMELEN+1];
	char line[MAXLINE+1];
	ALN mult_aln;
	OPT opt;

	if(argc!=3) {
		fprintf(stderr,"Usage: %s aln_file output_file\n",argv[0]);
		exit(1);
	}
        strcpy(infile,argv[1]);
        strcpy(outfile,argv[2]);

        init_options(&opt);

	(*opt.alnout_opt).output_clustal=FALSE;
        (*opt.alnout_opt).output_relacs=TRUE;

/* open the XML aln file */
        seq_input(infile,opt.explicit_type,FALSE,&mult_aln);

        if(mult_aln.nseqs<=0) exit(1);

	fprintf(stdout,"Number of sequences : %d\n",mult_aln.nseqs);
	fprintf(stdout,"Number of columns : %d\n",mult_aln.seqs[0].len);

/* for each sequence */
	for(i=0;i<mult_aln.nseqs;i++) {
		feature=SEQERRBLOCK;
/* check for errors */
		if(mult_aln.ft[i].nentries[feature] > 0) {
			mult_aln.seqs[i].output_index=(-1);
		}
	}

/* write out the sequences */
	n=0;
	for(i=0;i<mult_aln.nseqs;i++) {
		if(mult_aln.seqs[i].output_index!=(-1)) mult_aln.seqs[i].output_index=n++;
	}

	if(!open_alignment_output(outfile,opt.alnout_opt)) exit(1);
        create_alignment_output(mult_aln,*opt.alnout_opt);
        fprintf(stdout,"\n");
}

