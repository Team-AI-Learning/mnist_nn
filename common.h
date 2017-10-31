#ifndef __COMMON__
#define __COMMON__

#define UINT unsigned int

#include "omp.h"
#include<time.h>

#define FOR(x, n) \
for(UINT x = 0; x < n; ++ x)

#define FOR2D(x1, x2, n1, n2) for(int x1 = 0; x1 < n1; x1++) for(UINT x2 = 0; x2 < n2; x2++) 

#define FOR2D_OMP(x1, x2, n1, n2, looper) \
for(int looper = 0; looper < n1*n2; looper++) \
{	int x1 = looper / n2; int x2 = looper % n2; 
#define FOR_OMP_END }

#define FOR3D(x1, x2, x3, n1, n2, n3) for(int x1 = 0; x1 < n1; x1++) for(UINT x2 = 0; x2 < n2; x2++) for(UINT x3 = 0; x3 < n3; x3++) 

#define FOR4D(x1, x2, x3, x4, n1, n2, n3, n4) \
for(int x1 = 0; x1 < n1; x1++) for(UINT x2 = 0; x2 < n2; x2++) for(UINT x3 = 0; x3 < n3; x3++) for(UINT x4 = 0; x4 < n4; x4++)

// compare max value
#define CMP_MAX(a, b) (a > b) ? (a) : (b);

#define TIMESTAMP_START(tStart) \
double tStart = clock();

#define TIMESTAMP_SAVE(tStart, toSave) \
toSave += clock() - tStart;

#define TIMESTAMP_PRINT(tSave, str) \
printf(str \
	" %.3f sec.\n", (double)(tSave) / CLOCKS_PER_SEC);

#define TIMESTAMP_END(tStart, str) \
printf( str \
" %.3f sec.\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);


#endif