#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

//#define DUMP_GRID

#ifndef NDEBUG
#define NODE_PRINT_MSG( fmt, ...) printf( fmt, ##__VA_ARGS__)
#else
#define NODE_PRINT_MSG( fmt, ...) (void)
#endif

#if 0
#define NODE_PRINT_MSG_V NODE_PRINT_MSG
#else
#define NODE_PRINT_MSG_V(...)
#endif


struct GridConfig
{
    struct Dimension
    {
        int64_t size;
        double step;
    };

    Dimension x;
    Dimension t;
};

#ifdef DUMP_GRID
void
Dump_grid( int64_t x_size,
           int64_t t_size,
           const double* grid)
{
    for ( int64_t j = 0; j != t_size; ++j )
    {
        for ( int64_t i = 0; i != x_size; ++i )
        {
            int64_t idx = j * x_size + i;
            printf( "[%.2ld]%lf ", idx, grid[idx]);
        }

        printf( "\n");
    }
}
#else
void
Dump_grid( int64_t x_size,
           int64_t t_size,
           const double* grid) {}
#endif

static double
Func( double x, 
      double t)
{
    return 0;
}

static int 
Start_job( int rank,
           int n_nodes,
           const GridConfig* world_grid)
{
    NODE_PRINT_MSG( "Node %d: Started...\n", rank);

    int64_t block_size = world_grid->x.size / n_nodes;
    int64_t from = block_size * rank;
    int64_t to   = from + block_size;
    if ( rank == n_nodes - 1 )
    {
        to += world_grid->x.size % n_nodes; //block_size;
    }

    // Leader will merge all subgrids into one.
    // So allocate enough space for that.
    const int64_t grid_width = to - from;
    const size_t grid_alloc_count = ( rank == 0 ) 
                                  ? world_grid->t.size * world_grid->x.size
                                  : world_grid->t.size * grid_width;

    double* node_grid = (double*)malloc( grid_alloc_count * sizeof( double));
    assert( node_grid != nullptr );

    NODE_PRINT_MSG( "Node %d:\n"
                    "   Got interval: [%ld, %ld)\n"
                    "   Grid width  = %ld\n" 
                    "   Grid height = %ld\n" 
                    "   Allocated   = %ld doubles\n", 
                    rank, from, to, grid_width, world_grid->t.size, grid_alloc_count);

    // Initialize grid
    { 
        memset( node_grid, 0, grid_alloc_count * sizeof( double));

        for ( int64_t i = 0; i != grid_width; ++i )
        {
            node_grid[i] = exp( - (i + from) * world_grid->x.step);
        }

        // Initalize left border (leader process)
        if ( rank == 0 )
        {
            for ( int64_t j = 0; j != world_grid->t.size; ++j )
            {
                node_grid[j * grid_width] = exp( j * world_grid->t.step);
            }
        }
    }


    double time_start = MPI_Wtime();

    MPI_Request request;
    for ( int64_t j = 0; j != world_grid->t.size - 1; ++j )
    {
        if ( rank != 0 )
        {
            MPI_Isend( &node_grid[j * grid_width], 
                       1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request);

            NODE_PRINT_MSG_V( "Node %d: ISend( &node_grid[%ld] = %lf)\n", 
                              rank, j * grid_width, node_grid[j * grid_width]);
        }

        if ( rank != n_nodes - 1 )
        {
            MPI_Isend( &node_grid[(j + 1) * grid_width - 1], 
                       1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request);

            NODE_PRINT_MSG_V( "Node %d: ISend( &node_grid[%ld] = %lf)\n", 
                              rank, (j + 1) * grid_width - 1, node_grid[(j + 1) * grid_width - 1]);
        }

        double virtual_right;
        if ( rank != n_nodes - 1 )
        {
            MPI_Recv( &virtual_right, 1, MPI_DOUBLE, rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

            NODE_PRINT_MSG_V( "Node %d: Recv( virtual_right = %lf)\n", 
                              rank, virtual_right);
        }

        double virtual_left;
        if ( rank != 0 )
        {
            MPI_Recv( &virtual_left, 1, MPI_DOUBLE, rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, NULL);

            NODE_PRINT_MSG_V( "Node %d: Recv( virtual_left = %lf)\n", 
                              rank, virtual_left);
        }

        // Calculate all middle points via "Central differencing scheme"
        double left, right, f;
        for ( int64_t i = 0; i != grid_width; ++i )
        {
            if ( i == 0 )
            {
                if ( rank == 0 )
                {
                    // Skip first iteration due to boundary conditions.
                    continue;
                }

                left = virtual_left;
            } else
            {
                left = node_grid[j * grid_width + i - 1];
            }

            f = Func( i * world_grid->x.step, j * world_grid->t.step);

            if ( i == grid_width - 1 )
            {
                if ( rank == n_nodes - 1 )
                {
                    // Go to "Explicit left corner scheme".
                    break;
                }

                right = virtual_right;
            } else
            {
                right = node_grid[j * grid_width + i + 1];
            }

            node_grid[(j + 1) * grid_width + i]
                = (f - (right - left) / (2 * world_grid->x.step)) * world_grid->t.step + (left + right) / 2;
        }

        // Calculate last point in line by "Explicit left corner scheme"
        if ( rank == n_nodes - 1 )
        {
            double curr = node_grid[j * grid_width + to - from - 1];
            node_grid[(j + 1) * grid_width + to - from - 1]
                = (f - (curr - left) / world_grid->x.step) * world_grid->t.step + curr;
        }
    }

    if ( rank == 0 )
    {
        NODE_PRINT_MSG( "Node %d: Finished\n", rank);
        Dump_grid( grid_width, world_grid->t.size, node_grid);
        Dump_grid( world_grid->x.size, world_grid->t.size, node_grid);

        for ( int i = 1; i != n_nodes; ++i )
        {
            int count;
            MPI_Status status;
            MPI_Probe( i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            MPI_Get_count( &status, MPI_DOUBLE, &count);

            void* buf = node_grid + i * block_size * world_grid->t.size;
            MPI_Recv( buf, count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            NODE_PRINT_MSG( "Node %d: Recv( node_grid[size = %d] from Node %d)\n", 
                              rank, count, i);
            int64_t size = block_size;
            if ( i == n_nodes - 1 )
            {
                size += world_grid->x.size % n_nodes;
            }
        }
    } else
    {
        NODE_PRINT_MSG( "Node %d: Finished\n", rank);
        Dump_grid( grid_width, world_grid->t.size, node_grid);
        MPI_Ssend( node_grid, grid_alloc_count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        free( node_grid);
    }

    if ( rank == 0 )
    {
        double time_end = MPI_Wtime();
        NODE_PRINT_MSG( "Elapsed time: %lf\n", time_end - time_start);
        
        NODE_PRINT_MSG( "Start saving data...\n");

#ifdef DUMP_GRID
        Dump_grid( world_grid->x.size, world_grid->t.size, node_grid);
        FILE* file = fopen( "cont-sol.txt", "w");
        for ( int64_t j = 0; j != world_grid->t.size; ++j )
        {
            const int64_t shift = block_size * world_grid->t.size;
            for ( int n = 0; n != n_nodes; ++n )
            {
                int64_t size = block_size;
                if ( n == n_nodes - 1 )
                {
                    size += world_grid->x.size % n_nodes;
                }

                for ( int i = 0; i != size; ++i )
                {
                    fprintf( file, "%lf ", node_grid[shift * n + j * size + i]);
                }
            }
            fprintf( file, "\n");
        }
#endif

        NODE_PRINT_MSG( "Finish saving data. Cleaning up...\n");
        free( node_grid);
    }

    return 0;
}

int 
main( int argc, 
      char *argv[])
{
    MPI_Init( &argc , &argv);
    MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    int n_nodes;
    MPI_Comm_size( MPI_COMM_WORLD, &n_nodes);

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    GridConfig world_grid;

    world_grid.x.step = 1e-4;
    world_grid.x.size = 50000; // 100000;
    world_grid.t.step = 1e-4;
    world_grid.t.size = 20000; //10000;

    Start_job( rank, n_nodes, &world_grid);

    MPI_Finalize();
    return 0;
}
