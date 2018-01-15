#ifndef MATRIX
#define MATRIX

#include <stdio.h>
#include <stdlib.h>

#define Number double

#define ASSERT(x, s) \
if ((x) == 0) { \
	printf(s);	\
	exit(EXIT_FAILURE);	\
}

struct Matrix {
	int m;
	int n;
	Number *a;
};

struct Matrix const_mat(int m, int n, Number c);
struct Matrix rand_mat(int m, int n, Number low, Number high);

struct Matrix mat_add(const struct Matrix *lhs, const struct Matrix *rhs);
struct Matrix mat_mul(const struct Matrix *lhs, const struct Matrix *rhs);

void mat_add_to(struct Matrix *self, const struct Matrix *rhs);

#endif
