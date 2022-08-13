#include "md5func.cuh"

__global__ void kernel(char* wordl, cuuint32_t* wsize, cuuint32_t* ind, cuuint32_t* indsize, cuuint32_t* hash, cuuint32_t* hashsize, char* ans, cuuint32_t* jmpc);