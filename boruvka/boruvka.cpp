#include<iostream>
#include<vector>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
using namespace std;

vector<int> parents;
vector<int> ranking;
int components, vertices;

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

vector<Edge> G;
vector<Edge> best;

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

int algorithm() {
  parents.clear();
  ranking.clear();
  best.clear();
  int uparent, vparent, totalWeight = 0;
  for (int n = 0; n < vertices; n++) {
    parents.push_back(n);
    ranking.push_back(0);
    best.push_back(Edge());
  }
  while (components > 1) {
    struct timeval pbegin, pend;
    double timeParallel = 0.0;

    // gettimeofday is used to calculate the time spent in microseconds
    gettimeofday(&pbegin, NULL);
    for (int n = 0; n < G.size(); n++) {
      int u = G[n].u, v = G[n].v, w = G[n].w;
      uparent = find(u);
      vparent = find(v);

      if (uparent != vparent) {
        if (best[uparent].isNew() || best[uparent].w > w)
          best[uparent].set(u, v, w);
        if (best[vparent].isNew() || best[vparent].w > w)
          best[vparent].set(u, v, w);
      }
    }
    gettimeofday(&pend, NULL);
    timeParallel = 1000000*(pend.tv_sec-pbegin.tv_sec)+(pend.tv_usec-pbegin.tv_usec);

    printf("---\nSecuential time: %f\n", (timeParallel/1000000));

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

    best.clear();
  }
  cout << "Minimum weight: " << totalWeight << endl;
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
