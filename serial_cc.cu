#include "serial_cc.h"
#include "char_matrix.h"

typedef struct {
    int x,y;
} Pixel;

void bfs_run(CharMatrix* chars, GroupMatrix* groups, int x, int y, int groupID) {


    int maxQueue = chars->width * chars->height;
    Pixel* queue = (Pixel*)malloc(maxQueue * sizeof(Pixel));
    int front = 0, rear = 0;

    Pixel start; start.x = x; start.y = y;
    queue[0] = start;
    groups->groups[y * groups->width + x] = groupID;
    rear++;

    //Neighbour deltas
    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    while (front < rear) {
        Pixel p = queue[front++];

        for (int i = 0; i < 4; i++) {

            Pixel np; np.x = p.x + dx[i]; np.y = p.y + dy[i];
            //Skip pixels that exit the grid or that already have a group
            if (np.x >= chars->width || np.x < 0 || np.y >= chars->height || np.y < 0) continue;
            if (groups->groups[np.y * groups->width + np.x] != -1) continue;
            if (chars->matrix[p.y * chars->width + p.x] == chars->matrix[np.y * chars->width + np.x]) {
                groups->groups[np.y * groups->width + np.x] = groupID;
                queue[rear++] = np;
            }

        }
    }

    free(queue);

}

GroupMatrix cc_bfs(CharMatrix* mat) {


    GroupMatrix groups = initGroups(mat->width, mat->height);

    int nextID = 0;

    for (int x = 0; x < mat->width; x++) {
        for (int y = 0; y < mat->height; y++) {

            //If pixel has a group skip it
            if (groups.groups[y * groups.width + x] != -1) continue;

            bfs_run(mat, &groups, x, y, nextID++);

        }
    }

    return groups;

}