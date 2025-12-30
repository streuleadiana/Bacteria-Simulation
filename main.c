#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

/*
Compilare:
gcc -Wall -o prog main.c -lmsmpi
mpiexec -n 4 prog input.txt output.txt
*/

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
        if (rank == 0)
            printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank==0) {
        FILE *f = fopen(argv[1], "r");
        if (!f) {
            perror("Error opening input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fscanf(f, "%d %d %d", &N_rows, &M_cols, &Gen_count) != 3) {
            printf("Error reading file header.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        global_grid = (char *)malloc(N_rows * M_cols * sizeof(char));
        if (!global_grid) {
            fprintf(stderr, "Memory error\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N_rows; i++) {
            for (int j = 0; j < M_cols; j++) {
                char c;
                do {
                    if (fscanf(f, "%c", &c) != 1) break;
                } while (c == '\n' || c == '\r' || c == ' ');
                global_grid[i * M_cols + j] = c;
            }
        }
        fclose(f);
    }
    MPI_Bcast(&N_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Gen_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int rows_per_proc = N_rows / size;
    int remainder = N_rows % size;
    int current_displ = 0;

    for (int i = 0; i < size; i++) {
        int rows;
        if (i < remainder) {
            rows = rows_per_proc + 1;
        } else {
            rows = rows_per_proc;
        }
        sendcounts[i] = rows * M_cols;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    int local_rows = sendcounts[rank] / M_cols;

    char *local_grid = (char *)malloc((local_rows + 2) * M_cols * sizeof(char));
    char *next_grid  = (char *)malloc((local_rows + 2) * M_cols * sizeof(char));

    char *actual_data_ptr = local_grid + M_cols;

    MPI_Scatterv(global_grid, sendcounts, displs, MPI_CHAR,
                 actual_data_ptr, sendcounts[rank], MPI_CHAR,
                 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int gen = 0; gen < Gen_count; gen++) {

        int top_neighbor;
        if (rank == 0) {
            top_neighbor = MPI_PROC_NULL;
        } else {
            top_neighbor = rank - 1;
        }

        int bottom_neighbor;
        if (rank == size - 1) {
            bottom_neighbor = MPI_PROC_NULL;
        } else {
            bottom_neighbor = rank + 1;
        }

        MPI_Sendrecv(local_grid + M_cols, M_cols, MPI_CHAR, top_neighbor, 0,
                     local_grid + (local_rows + 1) * M_cols, M_cols, MPI_CHAR, bottom_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(local_grid + local_rows * M_cols, M_cols, MPI_CHAR, bottom_neighbor, 1,
                     local_grid, M_cols, MPI_CHAR, top_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < M_cols; j++) {

                int neighbors = count_neighbors(local_grid, i, j, local_rows + 2, M_cols);

                char current_cell = local_grid[i * M_cols + j];
                char new_cell = '.';

                if (current_cell == '.') {
                    if (neighbors == 3) {
                        new_cell = 'X';
                    } else {
                        new_cell = '.';
                    }
                } else {
                    if (neighbors < 2 || neighbors > 3) {
                        new_cell = '.';
                    } else {
                        new_cell = 'X';
                    }
                }

                next_grid[i * M_cols + j] = new_cell;
            }
        }

        char *tmp = local_grid;
        local_grid = next_grid;
        next_grid = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Execution time for %d processes: %f seconds\n", size, end_time - start_time);
    }

    actual_data_ptr = local_grid + M_cols;

    if (rank == 0) {
        final_grid = (char *)malloc(N_rows * M_cols * sizeof(char));
    }

    MPI_Gatherv(actual_data_ptr, sendcounts[rank], MPI_CHAR,
                final_grid, sendcounts, displs, MPI_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *f_out = fopen(argv[2], "w");
        if (!f_out) {
            perror("Error opening output file");
        } else {
            fprintf(f_out, "%d %d %d\n", N_rows, M_cols, Gen_count);
            for (int i = 0; i < N_rows; i++) {
                for (int j = 0; j < M_cols; j++) {
                    fprintf(f_out, "%c", final_grid[i * M_cols + j]);
                }
                fprintf(f_out, "\n");
            }
            fclose(f_out);
        }

        free(global_grid);
        free(final_grid);
    }

    free(local_grid);
    free(next_grid);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}