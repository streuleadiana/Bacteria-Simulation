#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

/*
Compilare:
gcc -Wall -o prog main.c -lmsmpi
mpiexec -n 4 prog bacteria*NR*.txt output.txt
*/

//Function to count alive neighbors
int count_neighbors(char *grid, int r, int c, int rows, int cols) {
    int count = 0;
    for (int i = r - 1; i <= r + 1; i++) {
        for (int j = c - 1; j <= c + 1; j++) {
            if (i == r && j == c)
                continue;
            if (i >= 0 && i < rows && j >= 0 && j < cols) {
                if (grid[i * cols + j] == 'X') {
                    count++;
                }
            }
        }
    }
    return count;
}

//Serial simulation for verification
void run_serial_simulation(char *initial_grid, char *result_grid, int rows, int cols, int gens) {
    char *current = (char *)malloc(rows * cols * sizeof(char));
    char *next = (char *)malloc(rows * cols * sizeof(char));

    // Validate memory allocation before usage
    if (!current || !next) {
        fprintf(stderr, "Memory allocation error in serial check!\n");
        if(current) free(current);
        if(next) free(next);
        return;
    }

    // Initialize grids
    memset(next, '.', rows * cols * sizeof(char));
    memcpy(current, initial_grid, rows * cols * sizeof(char));

    // Simulation loop
    for (int g = 0; g < gens; g++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int neighbors = count_neighbors(current, i, j, rows, cols);
                char current_cell = current[i * cols + j];
                char new_cell = '.';
                if (current_cell == 'X') {
                    if (neighbors == 2 || neighbors == 3) new_cell = 'X';
                } else {
                    if (neighbors == 3) new_cell = 'X';
                }
                next[i * cols + j] = new_cell;
            }
        }
        // Swap pointers for next generation
        char *tmp = current;
        current = next;
        next = tmp;
    }

    // Copy final result
    memcpy(result_grid, current, rows * cols * sizeof(char));

    free(current);
    free(next);
}

int main(int argc, char *argv[])
{
    int rank, size;
    int N_rows, M_cols, Gen_count;
    char *global_grid = NULL;
    char *final_grid = NULL;
    char *serial_result_grid = NULL;

    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // Input Reading (Rank 0 only)
    if (rank==0) {
        FILE *f = fopen(argv[1], "r");
        if (!f) {
            printf("Error opening input file: %s\n", argv[1]);
            fflush(stdout);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Read dimensions
        if (fscanf(f, "%d %d %d", &N_rows, &M_cols, &Gen_count) != 3) {
            printf("Error reading file header.\n");
            fflush(stdout);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Validate input data
        if (N_rows <= 0 || M_cols <= 0 || Gen_count < 0) {
            printf("Invalid input data! Rows/Cols must be positive and Gens >= 0.\n");
            fflush(stdout);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (size > N_rows) {
            printf("Too many processes for too few rows!\n");
            fflush(stdout);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate and read global grid
        global_grid = (char *)malloc(N_rows * M_cols * sizeof(char));
        if (!global_grid) {
            fprintf(stderr, "Memory error\n");
            fflush(stdout);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < N_rows; i++) {
            for (int j = 0; j < M_cols; j++) {
                char c;
                int ret;
                do {
                    ret = fscanf(f, "%c", &c);
                    if (ret == EOF) {
                        printf("File ended too early\n");
                        fflush(stdout);
                        free(global_grid);
                        fclose(f);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                } while (c == '\n' || c == '\r' || c == ' ' || c == '\t');

                if (c != 'X' && c != '.') {
                    printf("Invalid character '%c' at row %d, col %d. Only 'X' and '.' are accepted.\n", c, i, j);
                    fflush(stdout);
                    free(global_grid);
                    fclose(f);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                global_grid[i * M_cols + j] = c;
            }
        }
        fclose(f);

        // Allocate memory for final results
        final_grid = (char *)malloc(N_rows * M_cols * sizeof(char));
        serial_result_grid = (char *)malloc(N_rows * M_cols * sizeof(char));
        if (!final_grid || !serial_result_grid) {
            printf("Insufficient memory for final grids.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // Broadcast Dimensions
    MPI_Bcast(&N_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Gen_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Safety check for other processes
    if (N_rows <= 0 || M_cols <= 0) {
        MPI_Finalize();
        return 1;
    }

    //Domain Decomposition (Calculate counts/displacements)
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    if (!sendcounts || !displs) {
        printf("Insufficient memory for rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = N_rows / size;
    int remainder = N_rows % size;
    int current_displ = 0;

    // Distribute rows as evenly as possible
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

    // Local Memory Allocation
    char *local_grid = (char *)malloc((local_rows + 2) * M_cols * sizeof(char));
    char *next_grid  = (char *)malloc((local_rows + 2) * M_cols * sizeof(char));

    if (!local_grid || !next_grid) {
        printf("Insufficient memory for local_grid to rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize with '.' to avoid garbage values in ghost cells
    memset(local_grid, '.', (local_rows + 2) * M_cols * sizeof(char));
    memset(next_grid,  '.', (local_rows + 2) * M_cols * sizeof(char));

    char *actual_data_ptr = local_grid + M_cols;

    // Scatter Data
    MPI_Scatterv(global_grid, sendcounts, displs, MPI_CHAR,
                 actual_data_ptr, sendcounts[rank], MPI_CHAR,
                 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Main Parallel Loop
    for (int gen = 0; gen < Gen_count; gen++) {

        // Determine neighbors
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

        // Send first row to top, receive from bottom
        MPI_Sendrecv(local_grid + M_cols, M_cols, MPI_CHAR, top_neighbor, 0,
                     local_grid + (local_rows + 1) * M_cols, M_cols, MPI_CHAR, bottom_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send last row to bottom, receive from top
        MPI_Sendrecv(local_grid + local_rows * M_cols, M_cols, MPI_CHAR, bottom_neighbor, 1,
                     local_grid, M_cols, MPI_CHAR, top_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Process local grid
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < M_cols; j++) {

                // Count neighbors including ghost cells
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

        // Swap pointers for next generation
        char *tmp = local_grid;
        local_grid = next_grid;
        next_grid = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Gather Results
    actual_data_ptr = local_grid + M_cols;

    MPI_Gatherv(actual_data_ptr, sendcounts[rank], MPI_CHAR,
                final_grid, sendcounts, displs, MPI_CHAR,
                0, MPI_COMM_WORLD);

    // Verification and Output (Rank 0)
    if (rank == 0) {
        // Run serial version for comparison
        double serial_start = MPI_Wtime();
        run_serial_simulation(global_grid, serial_result_grid, N_rows, M_cols, Gen_count);
        double serial_end = MPI_Wtime();

        // Print Statistics
        double serial_duration = serial_end - serial_start;
        printf("Serial Execution (P=1): %f seconds\n", serial_duration);

        double parallel_duration = end_time - start_time;
        printf("Parallel Execution (P=%d): %f seconds\n", size, parallel_duration);

        double speedup = serial_duration / parallel_duration;
        printf("Speedup: %.2f\n", speedup);

        // Verify correctness
        int match = 1;
        for(int i=0; i<N_rows * M_cols; i++) {
            if (final_grid[i] != serial_result_grid[i]) {
                match = 0;
                printf("NOT MATCH at index %d! Parallel: %c, Serial: %c\n", i, final_grid[i], serial_result_grid[i]);
                break;
            }
        }
        if (match) {
            printf("SUCCESS: Serial and Parallel results MATCH!\n");
        } else {
            printf("FAILURE: Results do not match!\n");
        }

        // Write output to file
        FILE *f_out = fopen(argv[2], "w");
        if (!f_out) {
            printf("Error opening output file");
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
        free(serial_result_grid);
    }

    // Cleanup
    free(local_grid);
    free(next_grid);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}