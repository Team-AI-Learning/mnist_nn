#ifndef __COMMON__
#define __COMMON__

#define UINT unsigned int

// for loop with increasing integer
#define FOR(x, n) for(UINT x = 0; x < n; ++ x)
#define FOR2D(x1, x2, n1, n2) for(UINT x1 = 0; x1 < n1; x1++) for(UINT x2 = 0; x2 < n2; x2++) 
#define FOR3D(x1, x2, x3, n1, n2, n3) for(UINT x1 = 0; x1 < n1; x1++) for(UINT x2 = 0; x2 < n2; x2++) for(UINT x3 = 0; x3 < n3; x3++) 
#define FOR4D(x1, x2, x3, x4, n1, n2, n3, n4) \
for(UINT x1 = 0; x1 < n1; x1++) for(UINT x2 = 0; x2 < n2; x2++) for(UINT x3 = 0; x3 < n3; x3++) for(UINT x4 = 0; x4 < n4; x4++)

// compare max value
#define CMP_MAX(a, b) (a > b) ? (a) : (b);

#endif
