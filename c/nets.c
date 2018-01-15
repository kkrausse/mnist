#include "matrix.h"

int main() {
	struct Matrix a = rand_mat(20, 20, 0.0, 10.0);
	for (int i = 0; i < a.m; i++) {
		for (int j = 0; j < a.n; j++)
			printf("%.1lf, ", a.a[i * a.m + j]);
		printf("\n");
	}
}
