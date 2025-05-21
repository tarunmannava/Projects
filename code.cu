#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <unordered_map>

#define MAX_TOKENS        123
#define PAD_ID            -1
#define TRIGRAM_LUT_SIZE  96  /* Number of trigram combinations */
#define OUTPUT_FILENAME   "tokenization_results.txt"

/* Data structure for DNA sequences */
typedef struct {
    char* data;
    int length;
} dna_sequence;

/* Data structure for trigram lookup */
typedef struct {
    char key[3];
    int value;
} trigram_entry;

/* Global variables */
trigram_entry *trigram_lut;
char **sequences;
int *seq_lengths;
int num_sequences;
int *cpu_tokens, *gpu_tokens;

/* Device constant lookup table */
__device__ __constant__ char LUT_KEYS[TRIGRAM_LUT_SIZE][3];
__device__ __constant__ int  LUT_VALUES[TRIGRAM_LUT_SIZE];

/* Timing variables */
struct timezone Idunno;
struct timeval startTime, endTime;

/*
   GPU kernel: multiple sequences per block for better memory coalescing
   Each thread processes trigrams for 4 sequences
*/
#define SEQS_PER_BLOCK 4

__global__ void trigram_tokenizer_gpu(
    const char* __restrict__ dna,
    const int*  __restrict__ seq_offsets,
    const int*  __restrict__ seq_lengths,
    int*        __restrict__ token_ids,
    int         num_seqs
) {
    int block_seq_id = blockIdx.x * SEQS_PER_BLOCK;
    int t = threadIdx.x;

    /* Load LUT into shared memory */
    __shared__ char lut_k[TRIGRAM_LUT_SIZE][3];
    __shared__ int  lut_v[TRIGRAM_LUT_SIZE];
    for (int idx = t; idx < TRIGRAM_LUT_SIZE; idx += blockDim.x) {
        lut_k[idx][0] = LUT_KEYS[idx][0];
        lut_k[idx][1] = LUT_KEYS[idx][1];
        lut_k[idx][2] = LUT_KEYS[idx][2];
        lut_v[idx] = LUT_VALUES[idx];
    }
    __syncthreads();

    /* Process multiple sequences per thread */
    for (int s = 0; s < SEQS_PER_BLOCK; s++) {
        int seq_id = block_seq_id + s;
        if (seq_id >= num_seqs || t >= MAX_TOKENS) continue;

        int start = seq_offsets[seq_id];
        int len = seq_lengths[seq_id];
        int out_base = seq_id * MAX_TOKENS;

        /* Form trigram characters */
        char a, b, c;
        if (t == 0) {
            a = '<';
            b = (len > 0 ? dna[start] : '>');
            c = (len > 1 ? dna[start+1] : '>');
        }
        else if (t < len-1) {
            a = dna[start + t - 1];
            b = dna[start + t];
            c = dna[start + t + 1];
        }
        else if (t == len-1) {
            a = dna[start + t - 1];
            b = dna[start + t];
            c = '>';
        }
        else {
            token_ids[out_base + t] = PAD_ID;
            continue;
        }

        /* Lookup in LUT */
        int id = PAD_ID;
        for (int j = 0; j < TRIGRAM_LUT_SIZE; ++j) {
            if (a == lut_k[j][0] && b == lut_k[j][1] && c == lut_k[j][2]) {
                id = lut_v[j];
                break;
            }
        }
        token_ids[out_base + t] = id;
    }
}

/*
   CPU reference implementation
*/
void trigram_tokenizer_cpu(
    std::vector<std::string>& seqs,
    const std::vector<std::array<char,3>>& keys,
    const std::vector<int>& vals,
    int* output_tokens
) {
    /* Build map */
    std::unordered_map<std::string,int> lut;
    for (int i = 0; i < TRIGRAM_LUT_SIZE; ++i) {
        std::string k { keys[i][0], keys[i][1], keys[i][2] };
        lut[k] = vals[i];
    }
    int n = seqs.size();
    for (int i = 0; i < n; ++i) {
        const auto &s = seqs[i];
        int L = s.size();
        for (int t = 0; t < MAX_TOKENS; ++t) {
            std::string tri(3, '>');
            if (t == 0) {
                tri[0] = '<';
                if (L > 0) tri[1] = s[0];
                if (L > 1) tri[2] = s[1];
            }
            else if (t < L-1) {
                tri[0] = s[t-1];
                tri[1] = s[t];
                tri[2] = s[t+1];
            }
            else if (t == L-1) {
                tri[0] = s[t-1];
                tri[1] = s[t];
                tri[2] = '>';
            } else {
                output_tokens[i*MAX_TOKENS + t] = PAD_ID;
                continue;
            }
            output_tokens[i*MAX_TOKENS + t] = lut.count(tri) ? lut[tri] : PAD_ID;
        }
    }
}

/*
   Load DNA sequences from file
*/
void load_dna(const char* fn, std::vector<std::string>& v) {
    std::ifstream f(fn);
    std::string s;
    while (f >> s) v.push_back(s);
}

/*
   Report CPU running time
*/
double report_running_time() {
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff = endTime.tv_usec - startTime.tv_usec;
    if (usec_diff < 0) {
        sec_diff--;
        usec_diff += 1000000;
    }
    printf("CPU runtime: %ld.%06ld seconds\n", sec_diff, usec_diff);
    return sec_diff + usec_diff / 1000000.0;
}

/*
   Compare CPU and GPU results
*/
void compare_results(int* cpu_result, int* gpu_result, int total_elements) {
    bool match = true;
    int num_diffs = 0;

    for (int i = 0; i < total_elements; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            match = false;
            num_diffs++;
            if (num_diffs <= 5) {
                printf("Mismatch at position %d: CPU=%d, GPU=%d\n",
                       i, cpu_result[i], gpu_result[i]);
            }
        }
    }

    printf("\nOutputs match: %s\n", match ? "YES" : "NO");
    if (!match) {
        printf("Total mismatches: %d out of %d elements\n",
               num_diffs, total_elements);
    }
}

/*
   Write tokenization results to a file in a format suitable for PyTorch
*/
void write_results_to_file(int* tokens, int num_sequences,
                           const std::vector<std::string>& seqs) {
    std::ofstream output(OUTPUT_FILENAME);
    if (!output) {
        std::cerr << "Error: Could not open output file " << OUTPUT_FILENAME << std::endl;
        return;
    }

    // Write all tokens for each sequence (n x 123 array format)
    for (int i = 0; i < num_sequences; i++) {
        output << i;  // Sequence number
        for (int j = 0; j < MAX_TOKENS; j++) {
            output << " " << tokens[i*MAX_TOKENS + j];
        }
        output << std::endl;
    }

    output.close();
    printf("Results written to %s\n", OUTPUT_FILENAME);
}

/*
   Main function
*/
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    /* Load sequences */
    std::vector<std::string> seqs;
    load_dna(argv[1], seqs);
    num_sequences = seqs.size();

    printf("Processing %d DNA sequences\n", num_sequences);

    /* Build LUT keys & vals according to provided mapping */
    std::vector<std::array<char,3>> keys = {
        {'<','A','A'},{'<','A','C'},{'<','A','G'},{'<','A','T'},
        {'<','C','A'},{'<','C','C'},{'<','C','G'},{'<','C','T'},
        {'<','G','A'},{'<','G','C'},{'<','G','G'},{'<','G','T'},
        {'<','T','A'},{'<','T','C'},{'<','T','G'},{'<','T','T'},
        {'A','A','>'},{'A','A','A'},{'A','A','C'},{'A','A','G'},{'A','A','T'},
        {'A','C','>'},{'A','C','A'},{'A','C','C'},{'A','C','G'},{'A','C','T'},
        {'A','G','>'},{'A','G','A'},{'A','G','C'},{'A','G','G'},{'A','G','T'},
        {'A','T','>'},{'A','T','A'},{'A','T','C'},{'A','T','G'},{'A','T','T'},
        {'C','A','>'},{'C','A','A'},{'C','A','C'},{'C','A','G'},{'C','A','T'},
        {'C','C','>'},{'C','C','A'},{'C','C','C'},{'C','C','G'},{'C','C','T'},
        {'C','G','>'},{'C','G','A'},{'C','G','C'},{'C','G','G'},{'C','G','T'},
        {'C','T','>'},{'C','T','A'},{'C','T','C'},{'C','T','G'},{'C','T','T'},
        {'G','A','>'},{'G','A','A'},{'G','A','C'},{'G','A','G'},{'G','A','T'},
        {'G','C','>'},{'G','C','A'},{'G','C','C'},{'G','C','G'},{'G','C','T'},
        {'G','G','>'},{'G','G','A'},{'G','G','C'},{'G','G','G'},{'G','G','T'},
        {'G','T','>'},{'G','T','A'},{'G','T','C'},{'G','T','G'},{'G','T','T'},
        {'T','A','>'},{'T','A','A'},{'T','A','C'},{'T','A','G'},{'T','A','T'},
        {'T','C','>'},{'T','C','A'},{'T','C','C'},{'T','C','G'},{'T','C','T'},
        {'T','G','>'},{'T','G','A'},{'T','G','C'},{'T','G','G'},{'T','G','T'},
        {'T','T','>'},{'T','T','A'},{'T','T','C'},{'T','T','G'},{'T','T','T'}
    };

    /* Prepare values array according to provided mapping */
    std::vector<int> vals = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        91, 92, 93, 94, 95
    };

    /* Allocate memory for results */
    int total_tokens = num_sequences * MAX_TOKENS;
    cpu_tokens = (int*)calloc(total_tokens, sizeof(int));
    gpu_tokens = (int*)calloc(total_tokens, sizeof(int));

    /* CPU run & time */
    gettimeofday(&startTime, &Idunno);
    trigram_tokenizer_cpu(seqs, keys, vals, cpu_tokens);
    double cpu_time = report_running_time();

    /* Flatten and copy to GPU */
    std::vector<char> flat; flat.reserve(1000000);
    std::vector<int> offs(num_sequences), lens(num_sequences);
    int off = 0;
    for (int i = 0; i < num_sequences; ++i) {
        offs[i] = off;
        lens[i] = seqs[i].size();
        flat.insert(flat.end(), seqs[i].begin(), seqs[i].end());
        off += lens[i];
    }

    /* GPU memory allocation */
    char* dna_d;
    int*  offs_d;
    int*  lens_d;
    int*  out_d;
    cudaMalloc(&dna_d,  flat.size());
    cudaMalloc(&offs_d, num_sequences*sizeof(int));
    cudaMalloc(&lens_d, num_sequences*sizeof(int));
    cudaMalloc(&out_d,  total_tokens*sizeof(int));

    /* Copy data to GPU */
    cudaMemcpy(dna_d, flat.data(), flat.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(offs_d, offs.data(), num_sequences*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lens_d, lens.data(), num_sequences*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out_d, 0, total_tokens*sizeof(int));

    /* Copy LUT to constant memory */
    char hkeys[TRIGRAM_LUT_SIZE][3];
    int  hvals[TRIGRAM_LUT_SIZE];
    for (int i = 0; i < TRIGRAM_LUT_SIZE; ++i) {
        hkeys[i][0] = keys[i][0];
        hkeys[i][1] = keys[i][1];
        hkeys[i][2] = keys[i][2];
        hvals[i] = vals[i];
    }
    cudaMemcpyToSymbol(LUT_KEYS, hkeys, sizeof(hkeys));
    cudaMemcpyToSymbol(LUT_VALUES, hvals, sizeof(hvals));

    /* GPU run & time */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    /* Calculate grid dimensions for memory coalescing */
    int threadsPerBlock = 128;
    int numBlocks = (num_sequences + SEQS_PER_BLOCK - 1) / SEQS_PER_BLOCK;
    printf("Grid dimensions: %d blocks x %d threads/block (processing %d sequences per block)\n",
           numBlocks, threadsPerBlock, SEQS_PER_BLOCK);
    trigram_tokenizer_gpu<<<numBlocks, threadsPerBlock>>>(dna_d, offs_d, lens_d, out_d, num_sequences);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    printf("GPU runtime: %.6f ms\n", gpu_ms);
    printf("Speedup: %.2fx\n", (cpu_time * 1000) / gpu_ms);

    /* Copy results back to host */
    cudaMemcpy(gpu_tokens, out_d, total_tokens*sizeof(int), cudaMemcpyDeviceToHost);

    /* Compare results */
    compare_results(cpu_tokens, gpu_tokens, total_tokens);


    /* Write full results to file */
    write_results_to_file(gpu_tokens, num_sequences, seqs);

    /* Clean up */
    cudaFree(dna_d);
    cudaFree(offs_d);
    cudaFree(lens_d);
    cudaFree(out_d);
    free(cpu_tokens);
    free(gpu_tokens);

    return 0;
}