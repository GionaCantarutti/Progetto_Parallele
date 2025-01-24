#include "benchmark.h"
#include "char_matrix.h"
#include "serial_cc.h"

void benchmarkSerial(const char** tests, int testCount) {

    for (int i = 0; i < testCount; i++) {

        char* inPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 8) ); //7 for "Inputs/" and 1 for the "\0"
        inPath = strcat(strcpy(inPath, "Inputs/"), tests[i]);

        CharMatrix test = readInputFromFile(inPath);

        free(inPath);

        //ToDo: add timing
        GroupMatrix res = cc_bfs(&test);

        char* outPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 16) ); //15 for "Outputs/Serial/" and 1 for the "\0"
        inPath = strcat(strcpy(outPath, "Outputs/Serial/"), tests[i]);

        saveGroupMatrixToFile(&res, outPath);

        free(outPath);

        freeMat(&test); freeGroups(&res);

    }

}