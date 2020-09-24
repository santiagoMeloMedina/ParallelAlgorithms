
#include<iostream>
#include<vector>
#include<sys/time.h>
#include<stdio.h>
#include<unistd.h>
#include<pthread.h>
using namespace std;

#define THREADS 4

pthread_barrier_t barrier;
vector<pthread_t> avaliable;
vector<pthread_t> working;

vector<vector<int>> dist;

int min(int a, int b) {
  return (a < b) ? a : b;
}

void *floyd(void *arg) {
  long thread = (long)arg;
  int start = thread*dist.size()/THREADS;
  int end = ((thread+1)*dist.size()/THREADS) - 1;
  //printf("u: %d, start: %d, end: %d, dist: %d\n", thread, start, end, dist.size());
  pthread_barrier_wait(&barrier);
  for (int k = 0; k < dist.size(); k++) {
    for (int i = start; i <= end; i++) {
      printf("i: %d", i);
      for (int j = 0; j < dist.size(); j++) {
        //printf("u: %d, k: %d, i: %d, j: %d, dist: %d\n", thread, k, i, j, dist.size());
        //printf("u: %d, i-k: %d, k-j: %d, i-j: %d\n", thread, dist[i][k], dist[k][j], dist[i][j]);
        //dist[i][j] = min(dist[i][k]+dist[k][j], dist[i][j]);
      }
    }
    pthread_barrier_wait(&barrier);
  }
  pthread_exit(NULL);
}

void parallel() {
  pthread_attr_t attr;
  avaliable = vector<pthread_t>(THREADS, pthread_t());
  working = vector<pthread_t>(THREADS, pthread_t());
  pthread_barrier_init(&barrier, NULL, THREADS);
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  for (int n = 0; n < THREADS; n++) {
    pthread_t tmp = avaliable.back();
    avaliable.pop_back();
    working.push_back(tmp);
    pthread_create(&tmp, &attr, floyd, (void*)n);
  }
  for (int n = 0; n < THREADS; n++) {
    pthread_t tmp = working.back();
    pthread_join(tmp, NULL);
    working.pop_back();
  }
  pthread_attr_destroy(&attr);
  pthread_barrier_destroy(&barrier);
  //pthread_exit(NULL);
}

void print(vector<vector<int>> &matrix) {
  printf("\n");
  for (vector<int> n : matrix) {
    for (int m : n) {
      printf("%d ", m);
    }
    printf("\n");
  }
}

int main() {
  dist = vector<vector<int>>(4, vector<int>{1,3,2,5});
  //print(dist);
  parallel();
  //print(dist);
  return 0;
}
