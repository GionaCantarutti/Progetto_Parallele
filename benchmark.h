#ifndef BENCHMARK_H
#define BENCHMARK_H

typedef struct {
    //Times in ms
    double serial_time;
    double cuda_time;
    int is_matching; //0 false, 1 true, -1 did not check
} BenchmarkInstance;

typedef struct {
    BenchmarkInstance* results;
    char** tests;
    int nTests;
} BenchmarkRun;

void benchmarkSerial(const char** tests, int testCount);
BenchmarkRun runAndBenchmark(const char** tests, int testCount, bool verify);
void runAndVerify(const char** tests, int testCount);
void batchBenchmark(const char** tests, int testCount, int reps, bool verify, const char* batchName);

#endif