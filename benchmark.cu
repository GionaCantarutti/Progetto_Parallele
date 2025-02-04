#include <time.h>
#include <sys\timeb.h>
#include "benchmark.h"
#include "char_matrix.h"
#include "serial_cc.h"
#include "cuda_cc.cuh"

#define SEC_TO_NS(sec) ((sec)*1000000000)

/// Get a time stamp in nanoseconds.
uint64_t nanos()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    uint64_t ns = SEC_TO_NS((uint64_t)ts.tv_sec) + (uint64_t)ts.tv_nsec;
    return ns;
}

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
                    printf("\nFound missmatch at (%d, %d)\n", i % a->width, i / a->width);
                    printf("A's value maps to %d but %d is expected\n", current->into, b_val);
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

//Doesn't check for injectivity!
bool halfMatch(GroupMatrix* a, GroupMatrix* b) {
    if (a->width != b->width || a->height != b->height) return false;

    int* mapping;
    int max_ID = a->width * a->height + 10;
    mapping = (int*)malloc(max_ID * sizeof(int));
    for (int i = 0; i < max_ID; i++) mapping[i] = -1;

    for (int i = 0; i < a->width * a->height; i++) {
        if (mapping[a->groups[i]] == -1) { //If a's group ID is not mapped yet, map it
            mapping[a->groups[i]] = b->groups[i];
        } else { //Otherwise make sure that a's group maps to b's group
            if (mapping[a->groups[i]] != b->groups[i]) {
                printf("Missmatch found at (%d, %d). %d should map to %d but is instead mapped to %d... ", i % a->width, i / a->width, a->groups[i], b->groups[i], mapping[a->groups[i]]);
                free(mapping);
                return false;
            }
        }
    }

    free(mapping);

    return true;
}

bool isMatchingFast(GroupMatrix* a, GroupMatrix* b) {
    return halfMatch(a, b) && halfMatch(b, a);
}

BenchmarkRun runAndBenchmark(const char** tests, int testCount, bool verify) {

    BenchmarkRun run;
    run.results = (BenchmarkInstance*)malloc(testCount * sizeof(BenchmarkInstance));
    run.tests = (char**)malloc(testCount * sizeof(char*));
    run.nTests = testCount;

    for (int i = 0; i < testCount; i++) {

        char* inPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 8) ); //7 for "Inputs/" and 1 for the "\0"
        inPath = strcat(strcpy(inPath, "Inputs/"), tests[i]);

        CharMatrix test = readInputFromFile(inPath);

        free(inPath);
        printf("Benchmarking \"%s\"...\t", tests[i]);

        uint64_t start, end;
        BenchmarkInstance result;

        start = nanos();
        GroupMatrix cpu_res = cc_bfs(&test);
        end = nanos();
        result.serial_time = (double)(end - start)/(double)1000000;
        start = nanos();
        GroupMatrix cuda_res = cuda_cc(&test);
        end = nanos();
        result.cuda_time = (double)(end - start)/(double)1000000;

        if (verify) {
            bool match = isMatchingFast(&cpu_res, &cuda_res);
            result.is_matching = match;
            printf("%s", match ? "Results match!\n" : "Results do not match!\n");

            if (!match) {
                char* errPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 30) ); //21 for "Outputs/Errors/missmatch_cpu_" and 1 for the "\0"
                errPath = strcat(strcpy(errPath, "Outputs/Errors/missmatch_cpu_"), tests[i]);
                saveGroupMatrixToFile(&cpu_res, errPath);
                errPath = strcat(strcpy(errPath, "Outputs/Errors/missmatch_gpu_"), tests[i]);
                saveGroupMatrixToFile(&cuda_res, errPath);
                free(errPath);
            }
        } else {
            result.is_matching = -1;
            printf("\n");
        }

        printf("CPU time: %.1fms\t| Device time: %.3fms\n", result.serial_time, result.cuda_time);

        run.results[i] = result;
        run.tests[i] = (char*)tests[i];

        freeMat(&test); freeGroups(&cpu_res); freeGroups(&cuda_res);

    }

    return run;

}

void batchBenchmark(const char** tests, int testCount, int reps, bool verify, const char* batchName) {

    char* filePath = (char*)malloc(sizeof(char) * (strlen(batchName) + 20) ); //21 for "Outputs/Benchmarks/" and 1 for the "\0"
    filePath = strcat(strcpy(filePath, "Outputs/Benchmarks/"), batchName);

    FILE* file = fopen(filePath, "w"); //ToDo: support for name changing
    if (!file) {
        perror("Error opening file");
        return;
    }

    free(filePath);

    fprintf(file, "test_name, repetition, serial_time, cuda_time, is_matching\n");

    for (int i = 0; i < reps; i++) {

        printf("\nRepetition %d of %d (index %d)...\n", i+1, reps, i);

        BenchmarkRun run = runAndBenchmark(tests, testCount, verify);

        for (int j = 0; j < run.nTests; j++) {

            BenchmarkInstance res = run.results[j];

            fprintf(file, "%s, %d, %.1f, %.1f, %d\n", run.tests[j], i, res.serial_time, res.cuda_time, res.is_matching);

        }

    }

    fclose(file);

}

//Runs the tests both on CPU and GPU and verifies that both produce the same result
void runAndVerify(const char** tests, int testCount) {

    for (int i = 0; i < testCount; i++) {

        char* inPath = (char*)malloc(sizeof(char) * (strlen(tests[i]) + 8) ); //7 for "Inputs/" and 1 for the "\0"
        inPath = strcat(strcpy(inPath, "Inputs/"), tests[i]);

        CharMatrix test = readInputFromFile(inPath);

        free(inPath);
        printf("Testing \"%s\"...\t\t", tests[i]);

        GroupMatrix cpu_res = cc_bfs(&test);
        GroupMatrix cuda_res = cuda_cc(&test);

        bool match = isMatchingFast(&cpu_res, &cuda_res);
        printf("%s", match ? "Results match!\n" : "Results do not match!\n");

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