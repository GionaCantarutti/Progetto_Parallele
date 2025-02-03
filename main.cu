#include <stdio.h>
#include <stdlib.h>
#include "benchmark.h"

int main() {

    const int nTests = 6;
    const char* tests[] = {"all_black.txt", "random_noise.txt", "random_streaks.txt", "sparse_streaks.txt", "long_snake.txt", "chessboard.txt"};

    //runAndBenchmark(tests, nTests, true);
    //runAndVerify(tests, nTests);

    batchBenchmark(tests, nTests, 20, true, "chess_v1.txt");

    return 0;
}