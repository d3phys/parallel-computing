#include <mpi.h>
#include <gmp.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <stdlib.h>
#include <assert.h>

#ifndef NDEBUG
#define NODE_PRINT_MSG( fmt, ...) printf( fmt, ##__VA_ARGS__)
#else
#define NODE_PRINT_MSG( fmt, ...) (void)
#endif

#define Fatal_error( fmt, ...) \
do { printf( fmt, ##__VA_ARGS__); assert( 0 ); } while ( 0 )

class Buffer
{
public:
    Buffer( size_t size = 0): size_{ 0}, data_{ 0} { resize( size); }
    ~Buffer() { free( data_); }

    size_t size() const { return size_; }
    void resize( size_t size)
    {
        if ( size_ < size )
        {
            size_t new_size = size * 2;
            void* new_data = realloc( data_, new_size);
            if ( new_data == nullptr )
            {
                assert( 0 );
                return;
            }

            size_ = new_size;
            data_ = new_data;
        }
    }

    const void* data() const { return data_; }
          void* data()       { return data_; }

private:
    size_t size_;
    void* data_;
};

static void
Lorenz_calc( uint64_t from,
             uint64_t to,
             mpq_t sum,
             int rank)
{
    mpq_t tmp;
    mpq_init( tmp);

    for ( uint64_t i = from; i != to; ++i )
    {
        mpq_set_si( tmp, (i % 2 == 0) ? 1 : -1, 2 * i + 1);
        mpq_add( sum, sum, tmp);
    }

    {
        double res = mpq_get_d( sum);
        // gmp_printf("Node %d: part sum = %Qd ( %lf)\n", rank, sum, 4 * res);
        gmp_printf("Node %d: part sum = %lf\n", rank, 4 * res);
    }

    mpq_clear( tmp);
}

static void
Recv_mpz( Buffer *buf,
          int src_rank,
          mpz_t number)
{
    MPI_Status status;

    MPI_Probe( src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    int n_bytes;
    MPI_Get_count( &status, MPI_BYTE, &n_bytes);
    
    if ( buf->size() < (size_t)n_bytes )
    {
        buf->resize( n_bytes);
    }

    MPI_Recv( buf->data(), n_bytes, MPI_BYTE, src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    mpz_import( number, n_bytes, 1, sizeof( char), 0, 0, buf->data());
}

static void
Send_mpz( int dst_rank,
          mpz_t number)
{
    size_t count;
    void* buf = mpz_export( nullptr, &count, 1, sizeof( char), 0, 0, number);

    MPI_Ssend( buf, count, MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD);
    mpz_import( number, count, 1, sizeof( char), 0, 0, buf);

    free( buf);
}

static int 
Start_job( int rank,
           int n_nodes,
           int n_iters)
{
    NODE_PRINT_MSG( "Node %d: Started...\n", rank);

    double time_start = MPI_Wtime();

    uint64_t from = n_iters / n_nodes * rank;
    uint64_t to   = n_iters / n_nodes * (rank + 1);

    if ( rank == n_nodes - 1 )
    {
        to += n_iters % n_nodes;
    }

    NODE_PRINT_MSG( "Node %d: Got interval: [%lu, %lu)\n", rank, from, to);
    mpq_t part_sum;
    mpq_init( part_sum);
    Lorenz_calc( from, to, part_sum, rank);

    if ( rank == 0 )
    {
        Buffer buf;

        for ( int i = 1; i != n_nodes; ++i )
        {
            mpq_t tmp;
            mpq_init( tmp);
            Recv_mpz( &buf, i, mpq_numref( tmp));
            Recv_mpz( &buf, i, mpq_denref( tmp));

            //
            // GMP Docs: 5.14 Integer Import and Export:
            // There is no sign taken from the data, rop will simply be a positive integer. 
            // An application can handle any sign itself, and apply it for instance with mpz_neg. 
            //
            if ( n_iters / n_nodes * i % 2 != 0 )
            {
                mpz_neg( mpq_numref( tmp), mpq_numref( tmp));
            }

            mpq_add( part_sum, part_sum, tmp);
            mpq_clear( tmp);
        }

        mpz_mul_ui( mpq_numref( part_sum), mpq_numref( part_sum), 4u);

        double time_end = MPI_Wtime();

        mpf_t result;
        mpf_init2( result, 256);
        mpf_set_q( result, part_sum);

//        {
//            double res = mpq_get_d( part_sum);
//            mpq_out_str( stdout, 10, part_sum);
//            printf(" ( %lf)\n", res);
//        }

        mpf_out_str( stdout, 10, 100, result);
        printf("\n");

        printf( "Time elapsed: %lf s\n", time_end - time_start);

    } else
    {
        Send_mpz( 0, mpq_numref( part_sum));
        Send_mpz( 0, mpq_denref( part_sum));
    }

    mpq_clear( part_sum);
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

    int n_iters = 100;
    if ( argc == 2 )
    {
         n_iters = atoi( argv[1]);
    }

    Start_job( rank, n_nodes, n_iters);

    MPI_Finalize();
    return 0;
}
