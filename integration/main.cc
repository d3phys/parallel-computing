#include <cstdio>
#include <vector>
#include <cmath>
#include <mutex>
#include <cassert>
#include <thread>
#include <array>
#include <iostream>
#include <sstream>

using Float_t = double;

class Task_t
{
//public:
//    Float_t start() const { return start_; }
//    Float_t end()   const { return end_; }
//    Float_t end()   const { return end_; }
//
//private:
public:
    //bool isTerminating() { return };

public:
    Float_t start;
    Float_t end;
    Float_t fn_start;
    Float_t fn_end;
    Float_t area;
};

Float_t 
Func( Float_t x)
{
    return std::sin( x) / x;
}

#define dbg_Print( fmt__, ...)                                          \
do                                                                      \
{                                                                       \
    std::ostringstream oss;                                             \
    oss << std::this_thread::get_id();                                  \
    fprintf( stderr, "[%s] " fmt__, oss.str().c_str(), ##__VA_ARGS__);  \
} while ( 0 )

//#define dbg_Print( fmt__, ...) ;

const Float_t kEpsilon = 1e-7;
const size_t kMaxLocalStackSize  = 1;
const size_t kMaxGlobalStackSize = 1000;

std::mutex gStackMutex;
std::mutex gHasTaskMutex;
std::vector<Task_t> gStack;

std::mutex gResultMutex;
Float_t gResult = 0;
size_t gActiveThreads = 0;

const size_t kThreadsCount = 2;

Float_t
IntegrateTrap()
{
    Task_t task;

    dbg_Print( "Hello!");

    for ( ;; )
    {
        // Get one interval from global stack
        {
            // Wait for a new task
            gHasTaskMutex.lock();

            std::lock_guard<std::mutex> scoped_lock{ gStackMutex};

            // If we have task, there must be task in global stack
            assert( !gStack.empty() );

            {
                task = gStack.back();
                gStack.pop_back();
            }

            if ( !gStack.empty() )
            {
                gHasTaskMutex.unlock();
            }

            if ( task.start > task.end) // It is terminating task
            {
                // Terminating interval. Stop...
                break;
            }

            gActiveThreads++;
        }

        // Integrate one interval using local stack
        {
            Float_t result = 0;
            std::vector<Task_t> stack;

            for ( ;; )
            {
                Float_t mid = (task.start + task.end) / 2;
                Float_t fn_mid = Func( mid);

                Float_t area_start_mid = (fn_mid + task.fn_start) * (mid - task.start) / 2;
                Float_t area_mid_end   = (task.fn_end + fn_mid) * (task.end - mid) / 2;
                
                Float_t new_area = area_start_mid + area_mid_end;

                if ( std::abs( new_area - task.area) >= kEpsilon * std::abs( new_area) )
                {
                    stack.push_back( Task_t{ task.start, mid, task.fn_start, fn_mid, area_start_mid});
                    task.start = mid;
                    task.fn_start = fn_mid;
                    task.area = area_mid_end;
                } else
                {
                    result += new_area;

                    if ( stack.empty() )
                    {
                        // No tasks left in local stack. Stop...
                        dbg_Print( "No tasks in stack!!!\n");
                        break;
                    }

                    {
                        task = stack.back();
                        stack.pop_back();
                    }
                }

                // Move tasks to the global stack
                if ( stack.size() >= kMaxLocalStackSize )
                {
                    std::lock_guard<std::mutex> scoped_lock{ gStackMutex};

                    if ( gStack.empty() )
                    {
                        dbg_Print( "Sending to global!!!\n");
                        size_t n_elements = std::min( stack.size(), kMaxGlobalStackSize);
                        auto from = stack.end() - n_elements;
                        auto to   = stack.end();
                        gStack.insert( gStack.end(), make_move_iterator( from), make_move_iterator( to));
                        stack.erase( from, to);

                        gHasTaskMutex.unlock();
                    }
                }
            }

            // Add partial sum
            {
                std::lock_guard<std::mutex> scoped_lock{ gResultMutex};
                gResult += result;
            }
        }

        {
            std::lock_guard<std::mutex> scoped_lock{ gStackMutex};
            
            gActiveThreads--;

            if ( gActiveThreads == 0 
                 && gStack.empty() )
            {
                // Fill stack with terminating intervals
                for ( size_t i = 0; i != kThreadsCount; ++i )
                {
                    gStack.push_back( Task_t{ 1, 0, 0, 0, 0});
                }

                gHasTaskMutex.unlock();
            }
        }
    }

    return gResult;
}

int 
main()
{
    std::printf( "Hello world!\n");

    //std::array<std::thread, 4> threads = {};
    //

    Float_t start    = 1;
    Float_t end      = 2;
    Float_t fn_start = Func( start);
    Float_t fn_end   = Func( end);
    Float_t area     = (fn_end + fn_start) * (end - start) / 2;

    gStack.push_back( Task_t{ start, end, fn_start, fn_end, area});

    std::thread t1{ IntegrateTrap};
    std::thread t2{ IntegrateTrap};

    t1.join();
    t2.join();

    std::printf( "Integral = %.20lf\n", gResult);

    return 0;
}


