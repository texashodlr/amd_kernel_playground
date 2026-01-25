// Threads;
/*
    Creating:
        pthreads --> API
        C++ Thread APIs
        MPI
    Advantage over processes
        Memory sharing
        lower overhead
    
    Context = Thread
        From a CPU perspective
    Thread Pool
        Maintained by the OS
        CPU Scheduler picks one thread to execute per core
            Several CPU cores for a thread pool
        Context
            Store the context back to the memory
            Fetch another context from the memory
    pThread -> POSIX Thread
*/

#include <pthread.h>
#include <stdio.h>

void *print_message(void *args){
    for(int i = 0; i < 10; i++){
        printf("Hello World!\n");
    }
    return NULL;
}

int main(){
    pthread_t thread1, thread2;

    //Pointer to thread var (&thread) and then the function
    //These two threads are running print_message in parallel before running at the return line
    // Then rejoining at the pthread_join line
    pthread_create(&thread1, NULL,
    print_message, NULL);
    pthread_create(&thread2, NULL,
    print_message, NULL);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    return 0;
}

//Execute with: gcc main.c -lpthread