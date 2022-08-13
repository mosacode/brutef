#include "kernel.cuh"

#define TX(x) (x <= 9 ? x+'0':x-10+'a')

__global__ void kernel(char* wordl, cuuint32_t* wsize, cuuint32_t* ind, cuuint32_t* indsize, cuuint32_t* hash, cuuint32_t* hashsize, char* ans, cuuint32_t* jmpc) {
	cuuint32_t a, b, c, d;
	for (cuuint32_t i = threadIdx.x; i + 1 < *indsize; i += *jmpc) {
		if (ind[i + 1] > ind[i]+1) {
			computemd5(wordl + ind[i], ind[i + 1] - ind[i] - 1, &a, &b, &c, &d);
			for (cuuint32_t j = 0; j < *hashsize; j+=4) {
				if (a == hash[j * 4] && b == hash[j * 4 + 1] && c == hash[j * 4 + 2] && d == hash[j * 4 + 3]) {
					cuuint32_t k = j * 33;
					for (cuuint32_t p = ind[i]; p < ind[i + 1]; p++, k++) ans[k] = wordl[p];
					k--; ans[k] = '\0';
				}
			}
		}
	}
}