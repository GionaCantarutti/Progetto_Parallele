#include <stdio.h>
#include <stdlib.h>

__global__ void hello() {
    printf("Hello World!\n");
}

//Reads from filepath returning the boolean matrix and setting rows and cols to the appropriate amounts
bool** readInputFromFile(const char* filepath, int* rows, int* cols) {

    //Open file
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Error while opening the file");
        return NULL;
    }

    //Count rows and columns
    int colCount = 0;
    int rowCount = 0;
    char ch;
    while ((ch =fgetc(file)) != EOF) {
        if (ch == '\n') break;
        colCount++;
    }

    char* line = (char *)malloc((colCount + 1) * sizeof(char));
    rewind(file);

    while (fgets(line, colCount + 2, file)) {
        rowCount++;
    }
    rewind(file);

    //Actually reading the file now
    //Allocate matrix
    bool** matrix = (bool**)malloc(rowCount * sizeof(bool*));
    for (int i = 0; i < colCount; i++) {
        matrix[i] = (bool*)malloc(colCount * sizeof(bool));
    }

    //Populate matrix
    int row = 0;
    while (fgets(line, sizeof(line), file)) {
        for (int col = 0; col < colCount; col++) {
            matrix[row][col] = (line[col] == '1');
        }
        row++;
    }

    free(line);
    fclose(file);

    *rows = rowCount;
    *cols = colCount;

    return matrix;

}

int main() {
    hello<<<1, 4>>>();

    printf("CPU says hi!\n");

    cudaDeviceSynchronize();

    int r, c;
    readInputFromFile("Inputs/all_black.txt", &r, &c);

    return 0;
}