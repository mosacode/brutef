#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATEL(x, n) ((x<<n) | (x>>(32-n)))

#define FF(a, b, c, d, x, rc, ac) \
{ a+=F(b, c, d) + x + ac;\
  a = ROTATEL(a, rc);\
  a+=b;\
}
#define GG(a, b, c, d, x, rc, ac) \
{ a+=G(b, c, d) + x + ac;\
  a = ROTATEL(a, rc);\
  a+=b;\
}
#define HH(a, b, c, d, x, rc, ac) \
{ a+=H(b, c, d) + x + ac;\
  a = ROTATEL(a, rc);\
  a+=b;\
}
#define II(a, b, c, d, x, rc, ac) \
{ a+=I(b, c, d) + x + ac;\
  a = ROTATEL(a, rc);\
  a+=b;\
}

#define a0 ((cuuint32_t)0x67452301)
#define b0 ((cuuint32_t)0xEFCDAB89)
#define c0 ((cuuint32_t)0x98BADCFE)
#define d0 ((cuuint32_t)0x10325476)

__device__ inline void computemd5(char* word, cuuint32_t length, cuuint32_t* a1, cuuint32_t* b1, cuuint32_t* c1, cuuint32_t* d1) {
	cuuint32_t a, b, c, d;
	a = b = c = d = 0;

	cuuint32_t vals[14] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

	int i = 0;
	for (i = 0; i < length; i++) {
		vals[i / 4] |= word[i] << ((i % 4) * 8);
	}

	vals[i / 4] |= 0x80 << ((i % 4) * 8);

	cuuint32_t bitlen = length * 8;

#define in0  (vals[0])//x
#define in1  (vals[1])//y
#define in2  (vals[2])//z
#define in3  (vals[3])
#define in4  (vals[4])
#define in5  (vals[5])
#define in6  (vals[6])
#define in7  (vals[7])
#define in8  (vals[8])
#define in9  (vals[9])
#define in10 (vals[10])
#define in11 (vals[11])
#define in12 (vals[12])
#define in13 (vals[13])
#define in14 (bitlen) //w = bit length
#define in15 (0)
	a = a0, b = b0, c = c0, d = d0;

#define S11 7
#define S12 12
#define S13 17
#define S14 22
	FF(a, b, c, d, in0, S11, 3614090360);
	FF(d, a, b, c, in1, S12, 3905402710);
	FF(c, d, a, b, in2, S13, 606105819);
	FF(b, c, d, a, in3, S14, 3250441966);
	FF(a, b, c, d, in4, S11, 4118548399);
	FF(d, a, b, c, in5, S12, 1200080426);
	FF(c, d, a, b, in6, S13, 2821735955);
	FF(b, c, d, a, in7, S14, 4249261313);
	FF(a, b, c, d, in8, S11, 1770035416);
	FF(d, a, b, c, in9, S12, 2336552879);
	FF(c, d, a, b, in10, S13, 4294925233);
	FF(b, c, d, a, in11, S14, 2304563134);
	FF(a, b, c, d, in12, S11, 1804603682);
	FF(d, a, b, c, in13, S12, 4254626195);
	FF(c, d, a, b, in14, S13, 2792965006);
	FF(b, c, d, a, in15, S14, 1236535329);

#define S21 5
#define S22 9
#define S23 14
#define S24 20
	GG(a, b, c, d, in1, S21, 4129170786);
	GG(d, a, b, c, in6, S22, 3225465664);
	GG(c, d, a, b, in11, S23, 643717713);
	GG(b, c, d, a, in0, S24, 3921069994);
	GG(a, b, c, d, in5, S21, 3593408605);
	GG(d, a, b, c, in10, S22, 38016083);
	GG(c, d, a, b, in15, S23, 3634488961);
	GG(b, c, d, a, in4, S24, 3889429448);
	GG(a, b, c, d, in9, S21, 568446438);
	GG(d, a, b, c, in14, S22, 3275163606);
	GG(c, d, a, b, in3, S23, 4107603335);
	GG(b, c, d, a, in8, S24, 1163531501);
	GG(a, b, c, d, in13, S21, 2850285829);
	GG(d, a, b, c, in2, S22, 4243563512);
	GG(c, d, a, b, in7, S23, 1735328473);
	GG(b, c, d, a, in12, S24, 2368359562);

#define S31 4
#define S32 11
#define S33 16
#define S34 23
	HH(a, b, c, d, in5, S31, 4294588738);
	HH(d, a, b, c, in8, S32, 2272392833);
	HH(c, d, a, b, in11, S33, 1839030562);
	HH(b, c, d, a, in14, S34, 4259657740);
	HH(a, b, c, d, in1, S31, 2763975236);
	HH(d, a, b, c, in4, S32, 1272893353);
	HH(c, d, a, b, in7, S33, 4139469664);
	HH(b, c, d, a, in10, S34, 3200236656);
	HH(a, b, c, d, in13, S31, 681279174);
	HH(d, a, b, c, in0, S32, 3936430074);
	HH(c, d, a, b, in3, S33, 3572445317);
	HH(b, c, d, a, in6, S34, 76029189);
	HH(a, b, c, d, in9, S31, 3654602809);
	HH(d, a, b, c, in12, S32, 3873151461);
	HH(c, d, a, b, in15, S33, 530742520);
	HH(b, c, d, a, in2, S34, 3299628645);

#define S41 6
#define S42 10
#define S43 15
#define S44 21
	II(a, b, c, d, in0, S41, 4096336452);
	II(d, a, b, c, in7, S42, 1126891415);
	II(c, d, a, b, in14, S43, 2878612391);
	II(b, c, d, a, in5, S44, 4237533241);
	II(a, b, c, d, in12, S41, 1700485571);
	II(d, a, b, c, in3, S42, 2399980690);
	II(c, d, a, b, in10, S43, 4293915773);
	II(b, c, d, a, in1, S44, 2240044497);
	II(a, b, c, d, in8, S41, 1873313359);
	II(d, a, b, c, in15, S42, 4264355552);
	II(c, d, a, b, in6, S43, 2734768916);
	II(b, c, d, a, in13, S44, 1309151649);
	II(a, b, c, d, in4, S41, 4149444226);
	II(d, a, b, c, in11, S42, 3174756917);
	II(c, d, a, b, in2, S43, 718787259);
	II(b, c, d, a, in9, S44, 3951481745);

	a += a0;
	b += b0;
	c += c0;
	d += d0;

	*a1 = a;
	*b1 = b;
	*c1 = c;
	*d1 = d;
}