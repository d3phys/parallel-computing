#include <mpi.h>
#include <stdio.h>
#include <string.h>

#ifndef NDEBUG
#define NODE_PRINT_MSG( fmt, ...) printf( fmt, ##__VA_ARGS__)
#else
#define NODE_PRINT_MSG( fmt, ...) (void)
#endif

#ifndef ELEM_COUNT
#define ELEM_COUNT (10'000'000)
#endif

#ifndef ELEM_SIZE
#define ELEM_SIZE (sizeof( char))
#endif

const int kElem_count = int{ ELEM_COUNT};
const size_t kElem_size = size_t{ ELEM_SIZE};

int Start_job( int rank)
{
    NODE_PRINT_MSG( "Node %d: Started...\n", rank);

    char one_elem_buffer[kElem_size] = {0};

    if ( rank == 0 )
    {
        double start_time = MPI_Wtime();

        for ( int i = 0; i != kElem_count; ++i )
        {
            MPI_Ssend( &one_elem_buffer, 1, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        }

        double end_time = MPI_Wtime();

        NODE_PRINT_MSG( "Elapsed time: %lf\n", end_time - start_time);

    } else if ( rank == 1 )
    {
        MPI_Status status;
        for ( int i = 0; i != kElem_count; ++i )
        {
            MPI_Recv( &one_elem_buffer, 1, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
    }

    return 0;
}

int main( int argc, 
          char *argv[])
{
    MPI_Init( &argc , &argv);
    MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    Start_job( rank);

    MPI_Finalize();
    return 0;
}
