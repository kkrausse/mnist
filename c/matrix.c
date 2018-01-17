#include "matrix.h"

#define new(m, n) {m, n, malloc(sizeof(Number) * m * n)}

struct Matrix const_mat(int m, int n, Number c)
{
	struct Matrix r = new(m, n);

	for (int i = 0; i < m*n; ++i)
		r.a[i] = c;

	return r;
}

struct Matrix rand_mat(int m, int n, Number low, Number high)
{
	struct Matrix r = new(m, n);

	for (int i = 0; i < m*n; i++)
		r.a[i] = ((Number)rand() / (Number)RAND_MAX) * (high - low) + low;

	return r;
}

struct Matrix new_mat(int m, int n)
{
	retrun new(m, n);
}

void delete_mat(struct Matrix *mat)
{
	free(mat->a);
}

void resize_mat(struct Matrix *mat, int m, int n)
{
	if (m * n > mat->m * mat->n)
		mat->a = realloc(mat->a, m * n);
}

struct Matrix mat_add(const struct Matrix *lhs, const struct Matrix *rhs)
{
	int m = lhs->m;
	int n = lhs->n;

	ASSERT(m == rhs->m && n == rhs->n, "Error: cant add matricies\
		 dimensions dont match");
	
	struct Matrix r = new(m, n);

	for (int i = 0; i < m * n; i++)
		r.a[i] = lhs->a[i] + rhs->a[i];

	return r;
}	

/* 
 * add matrix to another, modifying the left one
 */
void mat_add_to(struct Matrix *self, const struct Matrix *rhs)
{
	ASSERT(self->m == rhs->m && self->n == rhs->n, "Error: cant add matricies\
                 dimensions dont match");

	for (int i = 0; i < self->m * self->n; i++)
		self->a[i] += rhs->a[i];
}

struct Matrix mat_mul(const struct Matrix *lhs, const struct Matrix *rhs)
{
	ASSERT(lhs->n == rhs->m, "Error: bad dimensions for matrix multiplication");
	
	struct Matrix r = new(lhs->m, rhs->n);

	Number sum;
	for (int i = 0; i < lhs->m; i++) {
		for (int j = 0; j < rhs->n; j++) {
			sum = 0.0;
			for (int k = 0; k < lhs->n; k++)
				sum += lhs->a[i * lhs->n + k] * rhs->a[k * rhs->n + j];
			r.a[i * r.n + j] = sum;
		}
	}
	
	return r;
}

struct Matrix hada_prod(const struct Matrix *lhs, const struct Matrix *rhs)
{
	int m = lhs->m;
	int n = lhs->n;

	ASSERT(m == rhs->m && n == rhs->n, "Error: cant add matricies\
		 dimensions dont match");
	
	struct Matrix r = new(m, n);

	for (int i = 0; i < m * n; i++)
		r.a[i] = lhs->a[i] * rhs->a[i];

	return r;
}
