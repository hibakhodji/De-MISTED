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
	char line[MAXLINE+1];
	ALN mult_aln;
	OPT opt;

	if(argc!=2) {
		fprintf(stderr,"Usage: %s aln_file\n",argv[0]);
		exit(1);
	}
        strcpy(infile,argv[1]);

        init_options(&opt);

/* open the XML aln file */
        seq_input(infile,opt.explicit_type,FALSE,&mult_aln);

        if(mult_aln.nseqs<=0) exit(1);

	fprintf(stdout,"Number of sequences : %d\n",mult_aln.nseqs);
	fprintf(stdout,"Number of columns : %d\n",mult_aln.seqs[0].len);

/* for each sequence */
	for(i=0;i<mult_aln.nseqs;i++) {
		feature=SEQERRBLOCK;
/* for each error annotation */
                for(j=0;j<mult_aln.ft[i].nentries[feature];j++) {
			start_res=mult_aln.ft[i].data[feature][j].start-1;
			if(start_res<0) start_res=0;
			end_res=mult_aln.ft[i].data[feature][j].end;
			if(strcmp(mult_aln.ft[i].data[feature][j].name,"SEQ_ERROR2")==0) end_res=start_res+1;
			pos2col(mult_aln.seqs[i].data,start_res,end_res,&start_col,&end_col);
			if(strcmp(mult_aln.ft[i].data[feature][j].name,"SEQ_ERRORN2")==0) start_col=0;
			if(strcmp(mult_aln.ft[i].data[feature][j].name,"SEQ_ERRORC2")==0) end_col=mult_aln.seqs[0].len;
			fprintf(stdout,"%s %d %s %d %d\n",mult_aln.seqs[i].name,i,mult_aln.ft[i].data[feature][j].name,start_col,end_col);
		}
	}

}

