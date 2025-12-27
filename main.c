#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

int count_neighbors(char *grid, int r, int c, int rows, int cols) {
    int count = 0;
    for (int i = r - 1; i <= r + 1; i++) {
        for (int j = c - 1; j <= c + 1; j++) {
            if (i == r && j == c) continue;

            if (i >= 0 && i < rows && j >= 0 && j < cols) {
                if (grid[i * cols + j] == 'X') {
                    count++;
                }
            }
        }
    }
    return count;
}

int main(int argc, char *argv[])
{
    int rank, size;
    int N_rows, M_cols, Gen_count;
    char *global_grid = NULL;
    char *final_grid = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) printf("Usage: %s <input_file> <generations>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank==0) {
        FILE *f = fopen(argv[1], "r");
        if (!f) {
            perror("Error opening input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fscanf(f, "%d %d %d", &N_rows, &M_cols, &Gen_count);

        char dummy;
        fscanf(f, "%c", &dummy);
        global_grid = (char *)malloc(N_rows * M_cols * sizeof(char));
        for (int i = 0; i < N_rows; i++) {
            for (int j = 0; j < M_cols; j++) {
                char c;
                fscanf(f, "%c", &c);
                while(c == '\n' || c == '\r') fscanf(f, "%c", &c);
                global_grid[i * M_cols + j] = c;
            }
        }
        fclose(f);
    }
    MPI_Bcast(&N_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Gen_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}