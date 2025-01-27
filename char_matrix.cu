#include "char_matrix.h"

// Reads from filepath returning the CharMatrix. In case of error returns a [0,0]-sized matrix.
CharMatrix readInputFromFile(const char* filepath) {
    // Open file
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Error while opening the file");
        CharMatrix errMat;
        errMat.height = 0;
        errMat.width = 0;
        return errMat;
    }

    // Count rows and columns
    int colCount = 0;
    int rowCount = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') break;
        colCount++;
    }

    char* line = (char*)malloc((colCount + 1) * sizeof(char));
    rewind(file);

    while (fgets(line, colCount + 2, file)) {
        rowCount++;
    }
    rewind(file);

    // Allocate matrix
    char** matrix = (char**)malloc(rowCount * sizeof(char*));
    for (int i = 0; i < rowCount; i++) {
        matrix[i] = (char*)malloc(colCount * sizeof(char));
    }

    // Populate matrix
    int row = 0;
    while (fgets(line, colCount + 2, file)) {
        for (int col = 0; col < colCount; col++) {
            matrix[row][col] = line[col];
        }
        row++;
    }

    free(line);
    fclose(file);

    CharMatrix mat;
    mat.width = colCount;
    mat.height = rowCount;
    mat.matrix = matrix;

    return mat;
}

// Frees the memory allocated for the CharMatrix
void freeMat(CharMatrix* mat) {
    if (mat->width != 0 && mat->height != 0) {
        for (int x = 0; x < mat->height; x++) {
            free(mat->matrix[x]);
        }
        free(mat->matrix);
    }
}

GroupMatrix initGroups(int width, int height) {

    GroupMatrix newMat = simpleInitGroups(width, height);

    // Init groups
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            newMat.groups[x][y] = -1; // -1 meaning "no group"
        }
    }

    return newMat;

}

//Init group matrix without setting group values
GroupMatrix simpleInitGroups(int width, int height) {

    GroupMatrix newMat;
    newMat.width = width;
    newMat.height = height;

    // Allocate matrix
    int** matrix = (int**)malloc(height * sizeof(int*));
    for (int i = 0; i < height; i++) {
        matrix[i] = (int*)malloc(width * sizeof(int));
    }

    newMat.groups = matrix;

    return newMat;

}

void freeGroups(GroupMatrix* mat) {
    if (mat->width != 0 && mat->height != 0) {
        for (int x = 0; x < mat->height; x++) {
            free(mat->groups[x]);
        }
        free(mat->groups);
    }
}

void saveGroupMatrixToFile(const GroupMatrix* groups, const char* filepath) {
    FILE* file = fopen(filepath, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }

    for (int y = 0; y < groups->height; y++) {
        for (int x = 0; x < groups->width; x++) {
            fprintf(file, "%d", groups->groups[y][x]);
            if (x < groups->width - 1) fprintf(file, " ");
        }
        fprintf(file, "\n");
    }

    fclose(file);
}