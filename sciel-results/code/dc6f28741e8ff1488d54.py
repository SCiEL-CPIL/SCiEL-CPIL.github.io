<<<<<<< SEARCH
import time
import random
import numpy as np

DEADLINE = 130.0  # safety margin before 150s kill

def _score(edges, n):
    # build adjacency
    adj = [[False]*n for _ in range(n)]
    deg = [0]*n
    for u,v in edges:
        adj[u][v]=adj[v][u]=True
        deg[u]+=1; deg[v]+=1
    best = 0
    for i in range(n):
        for j in range(i+1,n):
            if adj[i][j]:
                for k in range(j+1,n):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i]+deg[j]+deg[k]
                        if s>best: best=s
    return best
=======
import time
import random
import numpy as np

DEADLINE = 130.0  # safety margin before 150s kill

def _score(edges, n):
    # build adjacency using bitmasks for speed
    adj = [0] * n
    deg = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    best = 0
    # For each edge (i,j), find common neighbors k > j to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                # common neighbors of i and j, with k > j
                common = adj[i] & adj[j] & (~((1 << (j + 1)) - 1))
                while common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > best:
                        best = s
                    common ^= k_bit
    return best
>>>>>>> REPLACE

