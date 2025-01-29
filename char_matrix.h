#ifndef CHAR_MATRIX_H
#define CHAR_MATRIX_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char* matrix;
    int width;
    int height;
} CharMatrix;

typedef struct {
    int* groups;
    int width;
    int height;
} GroupMatrix;

CharMatrix readInputFromFile(const char* filepath);
void freeMat(CharMatrix* mat);
GroupMatrix initGroups(int width, int height);
GroupMatrix initGroupsUnique(int width, int height);
GroupMatrix initGroupsVal(int width, int height, int val);
GroupMatrix simpleInitGroups(int width, int height);
void freeGroups(GroupMatrix* mat);
void saveGroupMatrixToFile(const GroupMatrix* groups, const char* filepath);

#endif // CHAR_MATRIX_H