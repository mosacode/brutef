#include "kernel.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <bits/stdc++.h>

std::string wldir;
std::vector<std::string> wlists;
std::vector<std::string> hashes;
std::vector<cuuint32_t> hashval;
cuuint32_t* hashes_device;
cuuint32_t* hashes_device_size;
char* wordlist;
char* ans;
cuuint32_t anssz;
std::string ansh;
cuuint32_t* wordlistsize;
cuuint32_t* indecies;
cuuint32_t* indeciessize;
cuuint32_t* workingth;
std::string wordlisthost;
std::vector<cuuint32_t> wordlisthostind;
bool isw = false;
cudaError_t cudastatus;
cudaDeviceProp prp;

bool commandparse(int argc, char* argv[]) {
	for (int i = 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			return std::cout << "invalid command " << argv[i] << '\n', 0;
		}
		std::string commandline = argv[i] + 1, command;
		uint64_t j = 0;
		for (; j < commandline.size(); j++) {
			if (commandline[j] == '=' || commandline[j] == ':') break;
			command.push_back(commandline[j]);
		}
		if (command == "hash") {
			if (j + 1ll >= commandline.size() || commandline[j] != '=' || commandline.size() != 37)
				return std::cout << "invalid usage of command -hash\nusage: -hash=X where X is a valid hash\n", 0;
			command.clear(); j++; for (; j < commandline.size(); j++) command.push_back(commandline[j]);
			for (int j2 = 0; j2 < command.size(); j2++) {
				if (command[j2] >= 'A' && command[j2] <= 'Z') command[j2] |= 32;
				if (!((command[j2] >= '0' && command[j2] <= '9') || (command[j2] >= 'a' && command[j2] <= 'f')))
					return std::cout << "invalid hash " << command << " at " << i << "th command" << '\n', 0;
			}
			hashes.push_back(command);
		}
		/*else if (command == "wldir") {
			if (j + 1ll >= commandline.size() || commandline[j] != '=')
				return std::cout << "invalid usage of command -wldir\nusage: -wldir=X where X is a directory\n", 0;
			wldir.clear(); j++;
			for (; j < commandline.size(); j++) wldir.push_back(commandline[j]);
		}*/
		else if (command == "wl") {
			if (j + 1ll >= commandline.size() || commandline[j] != '=')
				return std::cout << "invalid usage of command -wl\nusage: -wl=X where X is a file\n", 0;
			j++; command.clear();
			for (; j < commandline.size(); j++)
				command.push_back(commandline[j]);
			wlists.push_back(command);
		}
		else {
			return std::cout << "unknown command " << commandline << '\n', 0;
		}
	}
	return 1;
}
inline cuuint32_t hextodec(char x) { return (x >= 'a' ? x - 'a' + 10 : x - '0'); }

void setinds() {
	wordlisthostind = { 0 };
	for (cuuint32_t i = 0; i < wordlisthost.size(); i++) if (wordlisthost[i] == '\n') wordlisthostind.push_back(i + 1);
	wordlisthostind.push_back(wordlisthost.size()+1);
}

void cleangmem() {
	cudaFree(wordlist);
	cudaFree(wordlistsize);
	cudaFree(indecies);
	cudaFree(indeciessize);
}

void copydatatogmem() {
	cuuint32_t tmp;
	cudaMalloc(&wordlist, wordlisthost.size());
	cudaMemcpy(wordlist, wordlisthost.data(), wordlisthost.size(), cudaMemcpyHostToDevice);
	cudaMalloc(&wordlistsize, 4);
	tmp = wordlisthost.size();
	cudaMemcpy(wordlistsize, &tmp, 4, cudaMemcpyHostToDevice);
	cudaMalloc(&indecies, wordlisthostind.size() * 4);
	cudaMemcpy(indecies, wordlisthostind.data(), wordlisthostind.size() * 4, cudaMemcpyHostToDevice);
	cudaMalloc(&indeciessize, 4);
	tmp = wordlisthostind.size();
	cudaMemcpy(indeciessize, &tmp, 4, cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[]) {
	cudastatus = cudaSetDevice(0);
	if (cudastatus) return std::cerr << "no cuda device found!\n", 0;
	cudaGetDeviceProperties(&prp, 0);
	{cuuint32_t tmp = prp.maxThreadsPerBlock; cudaMalloc(&workingth, 4); cudaMemcpy(workingth, &tmp, 4, cudaMemcpyHostToDevice); }

	if (!commandparse(argc, argv)) return 0;

	for (int i = 0; i < hashes.size(); i++) {
		for (int i2 = 0; i2 < 4; i2++) {
			cuuint32_t phash = 0, tmp = 0;
			phash |= ((hextodec(hashes[i][i2 * 8 + 6]) << 4) | hextodec(hashes[i][i2 * 8 + 7])) << 24;
			phash |= ((hextodec(hashes[i][i2 * 8 + 4]) << 4) | hextodec(hashes[i][i2 * 8 + 5])) << 16;
			phash |= ((hextodec(hashes[i][i2 * 8 + 2]) << 4) | hextodec(hashes[i][i2 * 8 + 3])) << 8;
			phash |= ((hextodec(hashes[i][i2 * 8 + 0]) << 4) | hextodec(hashes[i][i2 * 8 + 1])) << 0;
			hashval.push_back(phash);
		}
	}
	cudaMalloc(&hashes_device, hashval.size() * 4);
	cudaMemcpy(hashes_device, hashval.data(), hashval.size() * 4, cudaMemcpyHostToDevice);
	cudaMalloc(&hashes_device_size, 4); {
		cuuint32_t tmp = hashval.size()/4; cudaMemcpy(hashes_device_size, &tmp, 4, cudaMemcpyHostToDevice);
	}
	cudaMalloc(&ans, hashes.size() * 33); anssz = hashes.size();
	cudaMemset(ans, '\0', hashes.size() * 33);
	for (int i = 0; i < wlists.size(); i++) {
		std::ifstream input;
		input.open(wlists[i], std::ios_base::ate | std::ios_base::in);
		if (!input.is_open()) {
			std::cerr << "couldnt open file \"" << wlists[i] << "\". skipping file\n";
			continue;
		}
		size_t sz = input.tellg();
		wordlisthost.resize(sz);
		input.seekg(0);
		input.read((char*)wordlisthost.data(), sz);
		for (int i = 0; i < wordlisthost.size(); i++) if (wordlisthost[i] == '\0') { wordlisthost.resize(i); break; }
		if (wordlisthost.size() == 0) continue;
		setinds();
		if (isw) {
			cudastatus = cudaDeviceSynchronize();
			cleangmem();
			if (cudastatus) {
				return std::cout << "cuda error on cudadevicesynchorize\n" << cudaGetErrorString(cudastatus) << '\n' << cudaGetErrorName(cudastatus), 0;
				cudaFree(hashes_device);
				cudaFree(hashes_device_size);
			}
			copydatatogmem();
		}
		else copydatatogmem(), isw = true;
		kernel << <1, prp.maxThreadsPerBlock >> > (wordlist, wordlistsize, indecies, indeciessize, hashes_device, hashes_device_size, ans, workingth);
	}
	cudastatus = cudaDeviceSynchronize();
	cleangmem();
	if (cudastatus) {
		cudaFree(hashes_device);
		cudaFree(hashes_device_size);
		return std::cout << "cuda error on cudadevicesynchorize\n" << cudaGetErrorString(cudastatus) << '\n' << cudaGetErrorName(cudastatus), 0;
	}
	cudaFree(hashes_device);
	cudaFree(hashes_device_size);
	ansh.resize(anssz*33);
	cudaMemcpy((char*)ansh.data(), ans, anssz * 33, cudaMemcpyDeviceToHost);
	for (int i = 0; i < anssz; i++) {
		std::cout << hashes[i] << " : ";
		for (int j = 0; j < 33; j++) {
			if (ansh[i * 33 + j] == '\0') break;
			std::cout << ansh[i * 33 + j];
		}
		std::cout << std::endl;
	}
	cudaFree(ans);
	return 0;
}

//wef : 0e22f493c7480c3d7e38919ed8a72f6b
//few : 7c3daa31f887c333291d5cf04e541db5