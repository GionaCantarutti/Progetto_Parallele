#include <stdio.h>
#include <stdlib.h>
#include "benchmark.h"

int main() {

    const int nTests = 9;
    const char* tests[] = {"all_black.txt", "random_noise.txt", "random_streaks.txt", "sparse_streaks.txt", "long_snake.txt", "chessboard.txt", "small_streaks.txt", "big_streaks.txt", "32m_streaks.txt"};

    batchBenchmark(tests, nTests, 20, true, "chess_v8.txt");

    return 0;
}