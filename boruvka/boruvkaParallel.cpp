#include<iostream>
#include<vector>
#include "pthread.h"
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
using namespace std;

vector<int> parents;
vector<int> ranking;
int components, vertices, totalWeight;

struct Edge {
  int u;
  int v;
  int w;

  Edge() {
    u = -1; v = -1; w = -1;
  }

  Edge(int uu, int vv, int ww) {
    u = uu; v = vv; w = ww;
  }

  bool isNew() {
    return u == -1 && v == -1 && w == -1;
  }

  void set(int uu, int vv, int ww) {
    u = uu;
    v = vv;
    w = ww;
  }
};

struct Adj {
  int v, w;

  Adj() {
    w = -1; v = -1;
  }

  Adj(int vv, int ww) {
    v = vv; w = ww;
  }

  void set(int vv, int ww) {
    v = vv;
    w = ww;
  }

};

vector<Edge> G;
vector<vector<Adj>> GA;
vector<Edge> best;

vector<pthread_t> avaliable;

int find(int n) {
  if (parents[n] == n)
    return n;
  return find(parents[n]);
}

void unite(int u, int v) {
  int uparent = find(u);
  int vparent = find(v);
  if (ranking[uparent] < ranking[vparent]) {
    parents[uparent] = vparent;
  } else if (ranking[uparent] > ranking[vparent]) {
    parents[vparent] = uparent;
  } else {
    parents[vparent] = uparent;
    ranking[uparent]++;
  }
}

vector<vector<Adj>> edgeToAdjacent() {
  vector<vector<Adj>> result(vertices, vector<Adj>(0, Adj()));
  for (Edge e: G) {
    result[e.u].push_back(Adj(e.v,e.w));
    result[e.v].push_back(Adj(e.u,e.w));
  }
  return result;
}

void initializeAlgorithm() {
  parents.clear();
  ranking.clear();
  best.clear();
  totalWeight = 0;
  GA = edgeToAdjacent();
  for (int n = 0; n < vertices; n++) {
    parents.push_back(n);
    ranking.push_back(0);
    best.push_back(Edge());
  }
}

void search(int u) {
  int uparent, vparent;
  for (int n = 0; n < GA[u].size(); n++) {
    int v = GA[u][n].v, w = GA[u][n].w;
    uparent = find(u);
    vparent = find(v);

    if (uparent != vparent) {
      if (best[uparent].isNew() || best[uparent].w > w)
        best[uparent].set(u, v, w);
    }
  }
  avaliable.push_back(pthread_t());
}

void compose() {
  int uparent, vparent;
  for (int n = 0; n < vertices; n++) {
    if (!best[n].isNew()) {
      int u = best[n].u, v = best[n].v, w = best[n].w;
      uparent = find(u);
      vparent = find(v);
      if (uparent != vparent) {
        unite(uparent, vparent);
        totalWeight += w;
        components--;
      }
    }
  }
}

void* parallelSearch(void* prm) {
  long n = (long)prm;
  search(n);
}

int algorithm() {

  initializeAlgorithm();
  avaliable = vector<pthread_t>(GA.size(), pthread_t());

  while (components > 1) {

    struct timeval pbegin, pend;
    double timeParallel = 0.0;

    // gettimeofday is used to calculate the time spent in microseconds
    gettimeofday(&pbegin, NULL);

    for (int u = 0; u < GA.size(); u++) {
      pthread_t tmp = avaliable.back();
      avaliable.pop_back();
      pthread_create(&tmp, NULL, parallelSearch, (void*)u);
      pthread_detach(tmp);
    }

    gettimeofday(&pend, NULL);
    timeParallel = 1000000*(pend.tv_sec-pbegin.tv_sec)+(pend.tv_usec-pbegin.tv_usec);

    printf("---\nParallel time: %f\n", (timeParallel/1000000));

    compose();

    best.clear();
  }
  cout << "Weight is " << totalWeight << "\n";
}

int main() {
  cin >> vertices;
  components = vertices;
  int u, v, w;
  while (cin >> u >> v >> w){
    G.push_back(Edge(u, v, w));
  }
  algorithm();
  return 0;
}
