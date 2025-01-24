#include <stdio.h>
#include <stdlib.h>
#include "benchmark.h"

int main() {

    const int nTests = 4;
    const char* tests[] = {"all_black.txt", "random_noise.txt", "random_streaks.txt", "sparse_streaks.txt"};

    benchmarkSerial(tests, nTests);

    return 0;
}