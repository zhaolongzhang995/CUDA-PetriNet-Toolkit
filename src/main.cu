#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <cstring>
//#include <sys/time.h>

//m: the number of places 
//n: the number of transitions

#ifndef HASH_TABLE_LENGTH
#define HASH_TABLE_LENGTH 0x1ffffffU
#endif

struct link {
    int point_m;    // 第一个元素为整数
    int trans;
    struct link* ptr;    // 第二个元素为整数指针
};

void Getsize(int* rows, int* cols, const char* name) {
    FILE* fid = fopen(name, "r");
	if (fid == NULL) {//文件打开失败
        perror("Error opening file");
        return;
    }
    int m = 0, n = 0;
    char line0[256], line[256];
	if (fgets(line0, sizeof(line0), fid) == NULL) {//读取第一行(为TINA导出ndr文件信息)
        perror("Error reading first line");
        fclose(fid);
        return;
    }

	while (1) {
        const char delimiters[] = " \t";
        // Read a line
        fgets(line, sizeof(line), fid);

        // Break if line contains '@'
        if ((line[0] == '@')) {
            break;
        }

        // First call strtok, input string
        char* token = strtok(line, delimiters);

		// Continue to call strtok to get the second token
        token = strtok(NULL, delimiters);


        token = strtok(NULL, delimiters);
        while (token != NULL) {
            if (atoi(token) > n)
                n = atoi(token);
            if (strchr(token, ':') != NULL) {
                token = strtok(NULL, delimiters);
            }
            token = strtok(NULL, delimiters);
        }
        m++;
    }
    fclose(fid);
    *rows = m;
    *cols = n;
}

void LY_pnt2NW(const char* name, int* input, int* output, int* N, 
    char** P_name, int* M, int m, int n, char** T_name) {
    FILE* fid = fopen(name, "r");
    char line0[256], line[256];
    //int** output, ** input, * M;
    int  k = 0;

    if (fid == NULL)
    {
        printf(" %s warn", name);
        exit(1);
    }
    // Skip the first line
    fgets(line0, sizeof(line0), fid);


    while (1) {
        char* prestr, * poststr;
        int c;
        const char delimiters[] = " \t";
        // Read a line
        fgets(line, sizeof(line), fid);

        // Break if line contains '@'
        if ((line[0] == '@')) {
            break;
        }

        // Find comma positions
        prestr = strtok(line, ",");
        poststr = strtok(NULL, ",");
        //printf("%s  \n", prestr);

        // First call strtok, input string
        char* token = strtok(prestr, delimiters);

        // Continue to call strtok to get the second token
        token = strtok(NULL, delimiters);

        // Extract values from prestr
        if (token != NULL)
            c = atoi(token);  // Skip the first two characters
            //printf("%d  \n", c);
        M[k] = c;

        // Process prestr
        if (strlen(prestr) > 0) {
            token = strtok(NULL, delimiters);
            while (token != NULL) {
                if (strchr(token, ':') == NULL) {//不含冒号（权值为1）
					output[k * n + atoi(token) - 1] = 1;
                }
				else {//含冒号（权值不为1）
					int T = atoi(token);//变迁序号
					char* w = strtok(NULL, delimiters);//权值
                    int W = atoi(w);
					output[k * n + T - 1] = W;//权值
                }
				token = strtok(NULL, delimiters);//继续读取下一个
            }
        }

        // Process poststr
        if (strlen(poststr) > 0) {
            char* token = strtok(poststr, delimiters);
			while (token != NULL) {//读取变迁信息
				if (strchr(token, ':') == NULL) {//不含冒号（权值为1）
                    input[k * n + atoi(token) - 1] = 1;
                }
				else {//含冒号（权值不为1）
                    int T = atoi(token);
					char* w = strtok(NULL, delimiters);//权值
                    int W = atoi(w);
                    input[k * n + T - 1] = W;
                }
                token = strtok(NULL, delimiters);
            }
        }

        k++;
    }
    fgets(line, sizeof(line), fid);
    k = 0;
    while (1) {
        const char delimiters[] = " \t";
        // Read a line
        fgets(line, sizeof(line), fid);

        // Break if line contains '@'
        if ((line[0] == '@')) {
            break;
		}//读取库所信息
        char* token = strtok(line, delimiters);
        token = strtok(NULL, delimiters);
		P_name[k] = strdup(token);//库所名
        k++;

    }
    fgets(line, sizeof(line), fid);
    k = 0;
	while (1) {//读取变迁信息
        const char delimiters[] = " \t";
        // Read a line
        fgets(line, sizeof(line), fid);

        // Break if line contains '@'
        if ((line[0] == '@')) {
            break;
        }
        char* token = strtok(line, delimiters);
        token = strtok(NULL, delimiters);
		T_name[k] = strdup(token);//变迁名
        k++;
    }

    // Clean up
    fclose(fid);

}

void deleteZeroColumns(int* I, int* O, int* rows, int* cols) {
    // Find the all-zero column
    for (int j = 0; j < *cols; ++j) {
        int allZero = 1;  // Assuming all zeros in the current column

        for (int i = 0; i < *rows; ++i) {
            if ((I[i * (*cols) + j] != 0) || (O[i * (*cols) + j] != 0)) {
                allZero = 0;  // When there are non-zero elements in the front column
                break;
            }
        }

        // If the current column is full of zeros, delete the column
        if (allZero) {
            // Move the following column to override the current column
            for (int k = j; k < *cols - 1; ++k) {
                for (int l = 0; l < *rows; ++l) {
                    I[l * (*cols) + k] = I[l * (*cols) + k + 1];
                    O[l * (*cols) + k] = O[l * (*cols) + k + 1];
                }
            }

            // Reduction in the number of columns
            --(*cols);
            --j;  // Recheck the current column
        }
    }
}

__device__ uint32_t XXHash(int const* M_pro, int n_place, uint32_t seed, int k) {
    uint32_t h32 = seed + 0x165667b1 + n_place * 4;
    const uint32_t prime1 = 0x9e3779b1;
    const uint32_t prime2 = 0x85ebca77;
    const uint32_t prime3 = 0xc2b2ae3d;
    const uint32_t prime4 = 0x27d4eb2f;//初始化

    for (int i = 0; i < n_place; i++) {
        uint32_t k1 = (uint32_t)M_pro[k * n_place + i];
        k1 *= prime2;
        k1 = (k1 << 13) | (k1 >> 19);
        k1 *= prime1;
        h32 ^= k1;
        h32 = (h32 << 17) | (h32 >> 15);
        h32 = h32 * prime3 + prime4;
    }

    h32 += n_place * 4;

	// finalization
    h32 ^= h32 >> 15;
    h32 *= prime2;
    h32 ^= h32 >> 13;
    h32 *= prime3;
    h32 ^= h32 >> 16;

    h32 %= HASH_TABLE_LENGTH; // 0~HASH_TABLE_LENGTH

    return h32;
}


__device__ uint32_t MurmurHash3(int const* M_pro, int n_place, uint32_t seed, int k) {
    uint32_t h1 = seed;//0 

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593; //n_place=3
    for (int i = 0; i < n_place; i++) {
        uint32_t k1 = (uint32_t)M_pro[k * n_place + i];//k1=0

        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;

        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }
    // finalization
    h1 ^= n_place;
    h1 ^= (h1 >> 16);
    h1 *= 0x85ebca6b;
    h1 ^= (h1 >> 13);
    h1 *= 0xc2b2ae35;
    h1 ^= (h1 >> 16);

    h1 %= HASH_TABLE_LENGTH;//0~HASH_TABLE_LENGTH

    return h1;

}

__global__ void init_bloom(int* M0, int n_place, uint32_t* bloom, int* M_all) {
    int hash;
	hash = XXHash(M0, n_place, 0, 0);//第0行的M0计算哈希
    for (int j = 0; j < n_place; j++) {
		M_all[hash * n_place + j] = M0[j];//将初始标识写入M_all
    }
	bloom[hash] = 0;//将初始标识写入bloom所对应的位置
}

__global__ void Produce(int* M_new, int* d_N, int* d_I, int* M_pro, int n_place, int n_all,
    int* n_fired_d, int* n_trans_d, int* trans_pro, int n_trans) {
    extern __shared__ int key[];
    int thid = blockDim.x * blockIdx.x + threadIdx.x;
    int bid = blockIdx.x, p_from;
    int tid = threadIdx.x;
    int Mrow = gridDim.x * blockDim.x;
    uint32_t j, k;
    
    //检查是否可以发射
    for (j = 0; j < n_place; j++) {
        if (M_new[bid * n_place + j] - d_I[j * blockDim.x + tid] < 0) {
            return;
        }
    }

    atomicAdd(n_trans_d, 1);//可达图内发射的总变迁数
    k = atomicAdd(n_fired_d, 1);//本次循环内发射的变迁数

    //可发射后产生M_pro集合
    for (j = 0; j < n_place; j++) {
        M_pro[k * n_place + j] = d_N[j * blockDim.x + tid] + M_new[bid * n_place + j];
    }

    //确认发射标识的序号与发射变迁的序号
    p_from = n_all - gridDim.x + bid;
	atomicExch(&trans_pro[k * 3], p_from);//发射标识序号
	atomicExch(&trans_pro[k * 3 + 1], tid);//发射变迁序号
    /*trans_pro[(n_trans + k )* 3] = p_from;
    trans_pro[(n_trans + k) * 3 + 1] = tid;*/
}

__global__ void Check(int const* M_pro, int n_place, int* M_all, int n_all,
    uint32_t* bloom, int* n_pro_d, int* M_new, int* mutex, int* trans_pro,int n_trans) {
    uint32_t hash1;
    int n_pro, j, flag = 1;

    hash1 = XXHash(M_pro, n_place, 0, blockIdx.x);//第k行的M_pro计算哈希
    while (1) {
        flag = 1;
        while (atomicCAS(&mutex[hash1], 0, 1) != 0);
		__threadfence();//等待上面的步骤完成读取，保证读取的是最新的值
        if (bloom[hash1] == 0xffffffffU) {
			n_pro = atomicAdd(n_pro_d, 1);//产生标识的序号
            //atomicExch(&bloom[hash1], n_pro + n_all);

            bloom[hash1] = n_pro + n_all;
            for (j = 0; j < n_place; j++) {
				M_new[n_pro * n_place + j] = M_pro[blockIdx.x * n_place + j];//将新标识写入M_new
				M_all[hash1 * n_place + j] = M_pro[blockIdx.x * n_place + j];//将新标识写入M_all
            }

            __threadfence();
            atomicExch(&mutex[hash1], 0);//释放锁
            break;
        }
        else {
            for (j = 0; j < n_place; j++) {
                if (M_all[hash1 * n_place + j] != M_pro[blockIdx.x * n_place + j]) {
					flag = 0;//标识已存在
                    break;
                }
            }
            if (flag == 0) {
                __threadfence();
				atomicExch(&mutex[hash1], 0);//释放锁
				hash1 = (hash1 + 1) % (HASH_TABLE_LENGTH);//哈希冲突，改变哈希值继续查找
            }
            else {
                __threadfence();
                atomicExch(&mutex[hash1], 0);//释放锁
                break;
            }
        }
    }
    /*if (blockIdx.x == 0)
    {
        for (j = 0; j < n_place; j++) {
			printf("%d\t", M_pro[j]);
        }
		printf("\n");
    }*/
	atomicExch(&trans_pro[blockIdx.x * 3 + 2], bloom[hash1]);//标识发射后指向的标识序号
    //trans_pro[(n_trans + blockIdx.x) * 3 + 2] = bloom[hash1];
}



void print_result(int* M_all, int n_place, int n_all, int* trans,
    int n_trans, int n_arc, char** P_name, char** T_name, uint32_t* bloom) {
    int i, j, k, n, * index;
    FILE* file = fopen("file.txt", "w");
    index = (int*)calloc(HASH_TABLE_LENGTH, sizeof(int));

    fprintf(file, "n_place: %d n_arc: %d n_all: %d n_trans: %d\n", n_place, n_arc, n_all, n_trans);
    for (int ii = 0; ii < HASH_TABLE_LENGTH; ii++) {
        if (bloom[ii] != 0xffffffffU) {
            n = bloom[ii];
            index[n] = ii;
        }
    }
    for (i = 0; i < n_all; i++) {
        n = index[i];
        fprintf(file, "\nstate %d\n", i);
        fprintf(file, "props");
        for (j = 0; j < n_place; j++) {
            int token = M_all[n * n_place + j];
            if (token != 0) {
                if (token == 1)
                    fprintf(file, " %s", P_name[j]);
                else
                    fprintf(file, " %s*%d", P_name[j], token);
            }
        }
        fprintf(file, "\ntrans");
        k = 0;
        for (j = 0; j < n_trans; j++) {
            if (trans[j * 3] == i) {
                int n = trans[j * 3 + 1];
                fprintf(file, " %s/%d", T_name[n], trans[j * 3 + 2]);
                k++;
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

void bloomMarking(int* I, int* O, int* M0, int* N, int n_place,
    int n_arc, char** P_name, char** T_name) {
    uint32_t* bloom, * bloom_h;     // 在设备上定义char类型数组
    int* trans_d, * trans_1, * trans_pro;
    int i;
    int n_all_h = 1, n_new_h = 1, n_trans_h = 0, n_pro_h = 0, n_fired_h = 0;
    int* n_trans_d, * n_pro_d, * n_fired_d;
    int* M_all_d, * M_new, * M_pro, * d_N, * d_I, state = 0;
    int* M_all_h, * trans_h, * mutex;
    double copy = 0;
    clock_t start, end, copy_s, copy_e;
    size_t pitch;
    //link* d_link_heads0, * d_link_heads1, * d_link_new_head,* d_link;


	// Allocate memory on the device
    cudaMalloc((void**)&n_pro_d, sizeof(int));
    cudaMemset(n_pro_d, 0, sizeof(int));

    cudaMalloc((void**)&n_trans_d, sizeof(int));
    cudaMemset(n_trans_d, 0, sizeof(int));

    cudaMalloc((void**)&bloom, (HASH_TABLE_LENGTH) * sizeof(uint32_t));
    cudaMemset(bloom, 0xffffffffU, (HASH_TABLE_LENGTH) * sizeof(uint32_t));

    cudaMalloc((void**)&mutex, (HASH_TABLE_LENGTH) * sizeof(uint32_t));
    cudaMemset(mutex, 0, (HASH_TABLE_LENGTH) * sizeof(uint32_t));

    cudaMalloc(&M_new, sizeof(int) * n_place * n_new_h);
    cudaMemset(M_new, 0, sizeof(int) * n_place * n_new_h);

    cudaMalloc(&d_N, sizeof(int) * n_arc * n_place);
    cudaMemset(d_N, 0, sizeof(int) * n_arc * n_place);

    cudaMalloc(&d_I, sizeof(int) * n_arc * n_place);
    cudaMemset(d_I, 0, sizeof(int) * n_arc * n_place);

    cudaMalloc(&M_all_d, sizeof(int) * n_place * (HASH_TABLE_LENGTH));
    cudaMemset(&M_all_d, 0, sizeof(int) * n_place * (HASH_TABLE_LENGTH));

    //n_trans_h为链接的总数，n_fired_h为新产生链接的数量
    cudaMalloc(&trans_d, sizeof(int) * (HASH_TABLE_LENGTH) * 3 * 3);
    cudaMemset(&trans_d, 0, sizeof(int) * (HASH_TABLE_LENGTH) * 3 * 3);

    //cudaMalloc(&trans_d, sizeof(int) * 3);
    cudaMemcpy(M_new,  M0, n_place * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, n_arc * sizeof(int) * n_place, cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, I, n_arc * sizeof(int) * n_place, cudaMemcpyHostToDevice);
    
    //cudaprint << <1, 1 >> > (d_N, n_arc, n_place);

    cudaMalloc((void**)&n_fired_d, sizeof(int));
    cudaMemset((void**)&n_fired_d, 0, sizeof(int));

    cudaDeviceSynchronize();
    init_bloom << <1, 1 >> > (M_new, n_place, bloom, M_all_d);
    start = clock();
    while (1) {
        //将产生的新标识与旧标识合并

        cudaDeviceSynchronize();
        n_pro_h = 0;

        //n_pro_d和n_fired_d在本次循环起始时均为0
        cudaMemcpy(n_pro_d, &n_pro_h, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(n_fired_d, &n_pro_h, sizeof(int), cudaMemcpyHostToDevice);

        //为本次循环的产生标识分配内存
        cudaMalloc(&M_pro, sizeof(int) * n_place * n_new_h * n_arc);
        //cudaMemset(&M_pro, 0, sizeof(int) * n_place * n_new_h * n_arc);
        //number of new markings 

        //计算M_pro和trans_pro部分内容
        Produce << <n_new_h, n_arc >> > (M_new, d_N, d_I, M_pro, n_place, n_all_h,
            n_fired_d, n_trans_d, trans_d, n_trans_h);
        cudaDeviceSynchronize();
        cudaMemcpy(&n_fired_h, n_fired_d, sizeof(int), cudaMemcpyDeviceToHost);

        //准备好M_new与exist判断标识内存
        cudaFree(M_new);
        cudaMalloc(&M_new, sizeof(int) * n_place * n_fired_h);
        //cudaMemset(&M_new, 0, sizeof(int) * n_place * n_fired_h);
        copy_s = clock();
        Check << <n_fired_h, 1 >> > (M_pro, n_place, M_all_d, n_all_h, 
            bloom, n_pro_d, M_new, mutex, trans_d,n_trans_h);
        cudaDeviceSynchronize();
        copy_e = clock();
        copy = (double)(copy_e - copy_s) / CLOCKS_PER_SEC + copy;
        printf("total time:%fs\n", copy);

        //计算trans_pro指向标识序号
        cudaMemcpy(&n_pro_h, n_pro_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&n_trans_h, n_trans_d, sizeof(int), cudaMemcpyDeviceToHost);
        n_new_h = n_pro_h;
        n_all_h += n_new_h;


        cudaFree(M_pro);

        printf("  n_all_h: %d,n_new_h: %d,n_fired_h:%d\n", n_all_h, n_new_h, n_fired_h);

        if (n_new_h == 0)
            break;
    }
    end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("total time:%fs\n", elapsed);

	/*int* count, count_h;
	cudaMalloc(&count, sizeof(int));
	cudaMemset(count, 0, sizeof(int));
	countAll << <HASH_TABLE_LENGTH, 1 >> > (bloom, count);
	cudaDeviceSynchronize();
	cudaMemcpy(&count_h, count, sizeof(int), cudaMemcpyDeviceToHost);
	printf("count:%d\n", count_h);*/

    /*M_all_h = (int*)calloc(HASH_TABLE_LENGTH * n_place, sizeof(int));
    bloom_h = (uint32_t*)calloc(HASH_TABLE_LENGTH, sizeof(uint32_t));
    trans_h = (int*)calloc(n_trans_h * 3, sizeof(int));

    cudaMemcpy(M_all_h, M_all_d, HASH_TABLE_LENGTH * n_place * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(bloom_h, bloom, HASH_TABLE_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(trans_h, trans_d, n_trans_h * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("n_place: %d n_arc: %d n_all: %d n_trans: %d\n", n_place, n_arc, n_all_h, n_trans_h);
    start = clock();
    //print_result(M_all_h, n_place, n_all_h, trans_h, n_trans_h, n_arc, P_name, T_name,bloom_h);
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("write time:%fs\n", elapsed);*/
    
    printf("n_place: %d n_arc: %d n_all: %d n_trans: %d\n", n_place, n_arc, n_all_h, n_trans_h);
    
    cudaFree(n_pro_d);
    cudaFree(M_all_d);
    cudaFree(M_new);
    cudaFree(d_N);
    cudaFree(M_pro);
    cudaFree(d_I);
    cudaFree(n_trans_d);
}

int main(int argc,char* argv[]) {
    char* filename = "pnt\\recipe11.pnt";
    if (argc > 1)
    {
        filename = argv[1];
    }
    int* I, * O, * N, * M0;
    int m, n, * rows, * cols;
    char* P_name[5000], * T_name[5000];
    rows = &m;
    cols = &n;
    clock_t start, end;

    start = clock();
    // Determine m and n based on the second line
    Getsize(rows, cols, filename);

    // Allocate memory for input and output matrices
    O = (int*)malloc(n * m * sizeof(int));
    I = (int*)malloc(n * m * sizeof(int));
    N = (int*)malloc(n * m * sizeof(int));

    // 使用 memset 将矩阵初始化为全零
    memset(O, 0, n * m * sizeof(int));
    memset(I, 0, n * m * sizeof(int));

    // Allocate memory for M0
    M0 = (int*)malloc(m * sizeof(int));

    LY_pnt2NW(filename, I, O, N, P_name, M0, m, n, T_name);
    end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("read time:%fs\n", elapsed);
    // Use I, O, and M0 as needed

    deleteZeroColumns(I, O, rows, cols);
    for (int i = 0; i < m * n; ++i) {
        N[i] = O[i] - I[i];
    }
    
    bloomMarking(I, O, M0, N, m, n, P_name, T_name);

    

    return 0;
}
