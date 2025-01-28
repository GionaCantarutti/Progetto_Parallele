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

bool addMapping(Mapping** mapPtr, int from, int into) {
    Mapping *current = *mapPtr;
    // Check for injectivity: ensure 'into' isn't already mapped
    while (current != NULL) {
        if (current->into == into) return false; // 'into' already exists
        current = current->next;
    }

    // Create new node on the heap
    Mapping *newmap = (Mapping*)malloc(sizeof(Mapping));
    newmap->from = from;
    newmap->into = into;
    newmap->next = *mapPtr;  // Prepend to list
    *mapPtr = newmap;        // Update head pointer
    return true;
}

// Helper to free the linked list
void freeMappings(Mapping* map) {
    while (map != NULL) {
        Mapping* next = map->next;
        free(map);
        map = next;
    }
}

bool isMatching(GroupMatrix* a, GroupMatrix* b) {
    if (a->width != b->width || a->height != b->height) return false;

    Mapping* map = NULL; // Start with empty list

    // Initialize with first element's mapping
    if (!addMapping(&map, a->groups[0], b->groups[0])) {
        freeMappings(map);
        return false;
    }

    for (int i = 0; i < a->width * a->height; i++) {
        int a_val = a->groups[i];
        int b_val = b->groups[i];

        Mapping* current = map;
        bool found = false;
        while (current != NULL) {
            if (current->from == a_val) {
                if (current->into != b_val) {
                    freeMappings(map);
                    return false;
                }
                found = true;
                break;
            }
            current = current->next;
        }

        if (!found) {
            if (!addMapping(&map, a_val, b_val)) {
                freeMappings(map);
                return false;
            }
        }
    }

    freeMappings(map);
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

        printf("%s", match ? "Results match\n" : "Results do not match\n");

        if (!match) {
            char* errPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 30) ); //21 for "Outputs/Errors/missmatch_cpu_" and 1 for the "\0"
            errPath = strcat(strcpy(errPath, "Outputs/Errors/missmatch_cpu_"), tests[i]);
            saveGroupMatrixToFile(&cpu_res, errPath);
            errPath = strcat(strcpy(errPath, "Outputs/Errors/missmatch_gpu_"), tests[i]);
            saveGroupMatrixToFile(&cuda_res, errPath);
            free(errPath);
        }

        freeMat(&test); freeGroups(&cpu_res); freeGroups(&cuda_res);

    }

}