#include <cstdio>
#include <iterator>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <omp.h>

// Default benchmark settings
constexpr int kSeed = 1;
constexpr int kIterationsCount = 10;
constexpr int kArraySize = 10'000'000;

template < class RandomIt >
void QuickSort( RandomIt first,
                RandomIt last)
{
    if ( first == std::prev( last) )
    {
        return;
    }

    auto pivot = (*first);

    RandomIt front = first;
    RandomIt back  = std::prev( last);

    for ( ;; )
    {
        while ( *front < pivot )
        {
            ++front;
        }

        while ( *back > pivot )
        {
            --back;
        }

        if ( front >= back )
        {
            break;
        }

        std::iter_swap( front, back);
        ++front;
        --back;
    }

    #pragma omp task
    {
        QuickSort( first, std::next( back));
    }

    #pragma omp task
    {
        QuickSort( std::next( back), last);
    }

    // #pragma omp taskwait
}

template < class RandomIt >
void QuickSortParallel( RandomIt first,
                        RandomIt last)
{
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            QuickSort( first, last);
        }
    }
}

template < typename T >
void Dump( const std::vector<T>& array)
{
    auto dump = []( auto elem) { std::cout << elem << " "; };
    std::for_each( array.cbegin(), array.cend(), dump);
    std::cout << std::endl;
}

int main()
{
    // Setup benchmark settings
    int n_iters   = kIterationsCount;
    int seed      = kSeed;
    int n_elems   = kArraySize;
    int n_threads = 1;

    const char*   n_iters_env = std::getenv( "N_ITERS");
    const char*      seed_env = std::getenv( "SEED");
    const char*   n_elems_env = std::getenv( "N_ELEMS");
    const char* n_threads_env = std::getenv( "OMP_NUM_THREADS");

    if (   n_iters_env ) { n_iters   = std::atoi(   n_iters_env); }
    if (      seed_env ) { seed      = std::atoi(      seed_env); }
    if (   n_elems_env ) { n_elems   = std::atoi(   n_elems_env); }
    if ( n_threads_env ) { n_threads = std::atoi( n_threads_env); }

    std::srand( seed);
    std::vector<int> array( n_elems);

    // Dump( array);

    std::chrono::duration<double> diff;

    for ( int i = 0; i != n_iters; ++i )
    {
        std::generate( array.begin(), array.end(), []() { return std::rand() % 100000; });

        const auto start = std::chrono::high_resolution_clock::now();
        QuickSortParallel( array.begin(), array.end());
        const auto end = std::chrono::high_resolution_clock::now();

        diff += end - start;
    }

    // Dump( array);

    std::cout << "Seed:              " << seed         <<  "\n";
    std::cout << "Iterations number: " << n_iters      <<  "\n";
    std::cout << "Elements number:   " << n_elems      <<  "\n";
    std::cout << "Threads:           " << n_threads    <<  "\n";
    std::cout << "Elapsed time:      " << diff.count() << "s\n";

    return 0;
}

