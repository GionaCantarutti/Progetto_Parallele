#include "benchmark.h"
#include "char_matrix.h"
#include "serial_cc.h"
#include "cuda_cc.h"

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

typedef struct Mapping Mapping;
struct Mapping {
    int from;
    int into;
    Mapping* next;
};

bool isMapped(Mapping* map, int n) {

    while (map != NULL) {
        if (n == map->from) return true;
        map = map->next;
    }

    return false;

}

int mapped(Mapping* map, int n) {

    while (map != NULL) {
        if (n == map->from) return map->into;
        map = map->next;
    }

    perror("Mapping not found!");
    return -1;

}

bool addMapping(Mapping* map, int from, int into) {

    //Force injective mapping
    if (map->into == into) return false;

    while (map->next != NULL) {
        if (map->into == into) return false;
        map = map->next;
    }

    Mapping newmap;
    newmap.from = from; newmap.into = into;
    map->next = &newmap;

    return true;
}

bool isMatching(GroupMatrix* a, GroupMatrix* b) {

    if (a->width != b->width || a->height != b->height) return false;

    Mapping map;
    map.from = a->groups[0];
    map.into = b->groups[0];

    for (int x = 0; x < a->width; x++) {
        for (int y = 0; y < a->height; y++) {

            if (isMapped(&map, a->groups[y * a->width + x])) {
                if (mapped(&map, a->groups[y * a->width + x]) != b->groups[y * b->width + x]) return false;
            } else {
                if (!addMapping(&map, a->groups[y * a->width + x], b->groups[y * b->width + x])) return false;
            }

        }
    }

    return true;

}

//Runs the tests both on CPU and GPU and verifies that both produce the same result
void runAndVerify(const char** tests, int testCount) {

    for (int i = 0; i < testCount; i++) {

        char* inPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 8) ); //7 for "Inputs/" and 1 for the "\0"
        inPath = strcat(strcpy(inPath, "Inputs/"), tests[i]);

        CharMatrix test = readInputFromFile(inPath);

        free(inPath);

        GroupMatrix cpu_res = cc_bfs(&test);
        GroupMatrix cuda_res = cuda_cc(&test);

        bool match = isMatching(&cpu_res, &cuda_res);

        printf("%s", match ? "Results match" : "Results do not match");

        freeMat(&test); freeGroups(&cpu_res); freeGroups(&cuda_res);

    }

}