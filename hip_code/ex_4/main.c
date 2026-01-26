// Monte Carlo Code
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>

// Serial implementation of MC
double monte_carlo_pi_serial(int n) {
    double x, y;
    double pi = 0;

    for (int i = 0; i < n; i++){
        x = (double)rand() / RAND_MAX * 2 - 1;
        y = (double)rand() / RAND_MAX * 2 - 1;
        if (x*x + y*y <= 1){
            pi++;
        }
    }
    return 4 * pi / n;
}

// Each thread ends up calculating it's own PI values
struct monte_carlo_pi_args{
    int n;
    double pi;
};

void *monte_carlo_pi_thread(void *data){
    struct monte_carlo_pi_args *args = (struct monte_carlo_pi_args *)data;
    args->pi = monte_carlo_pi_serial(args->n);
    return NULL;
}

double monte_carlo_pi_parallel(int n){
    int n_threads = 4;
    pthread_t threads[n_threads];
    struct monte_carlo_pi_args args[n_threads];

    //Creating threads
    for (int i = 0; i < n_threads; i++){
        args[i].n = n / n_threads;
        pthread_create(&threads[i], NULL, monte_carlo_pi_thread, &args[i]);
    }

    // Wait for threads to finish
    for (int i=0;i<n_threads;i++){
        pthread_join(threads[i], NULL);
    }

    //Calculate PI
    double pi = 0;
    for (int i = 0; i < n_threads; i++){
        pi += args[i].pi;
    }

    pi /= n_threads;
    return pi;
}

int main(){
    printf("Starting monte carlo parallel run!");
    printf("Value of PI: %f\n", monte_carlo_pi_parallel(12));
    return 0;
}