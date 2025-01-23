#include <stdio.h>
#include <stdlib.h>
#include "char_matrix.h"
#include "serial_cc.h"

__global__ void hello() {
    printf("Hello World!\n");
}

int main() {

    CharMatrix test = readInputFromFile("Inputs/random_streaks.txt");

    GroupMatrix res = cc_bfs(&test);

    saveGroupMatrixToFile(&res, "Outputs/test.txt");

    freeMat(&test); freeGroups(&res);

    return 0;
}