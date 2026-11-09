import numpy as np
import time

def priority(adjacency):
       start_time = time.time()
       deadline = start_time + 100.0 # Safety margin
       
       adj = adjacency.astype(float)
       n = adj.shape[0]
       
       # Current state
       remaining = np.ones(n, dtype=bool)
       # Current degrees
       deg = adj.sum(axis=1).astype(int)
       
       scores = np.zeros(n)
       current_score = n
       
       for _ in range(n):
           if time.time() > deadline:
               break
           
           if not remaining.any():
               break
           
           # Find vertex with minimum degree among remaining
           # Use a large number for non-remaining
           masked_deg = np.where(remaining, deg, n + 1)
           v = np.argmin(masked_deg)
           
           if not remaining[v]:
               break
           
           # Select v
           scores[v] = current_score
           current_score -= 1
           
           # Block v and its neighbors
           # Neighbors are where adj[v] == 1
           neighbors = np.where(adj[v] > 0.5)[0]
           
           # Update remaining
           remaining[v] = False
           for u in neighbors:
               if remaining[u]:
                   remaining[u] = False
                   # Update degrees of u's neighbors?
                   # For Dynamic Lowest Degree, we need accurate degrees of remaining nodes.
                   # When we remove u, the degree of u's neighbors decreases.
                   
           # We need to update degrees for all remaining nodes whose neighbors were removed.
           # Removed nodes: v and neighbors.
           removed = np.zeros(n, dtype=bool)
           removed[v] = True
           removed[neighbors] = True
           
           # For each remaining node w, deg[w] = sum(adj[w, ~removed])
           # This is O(N^2) per step -> O(N^3) total.
           # N=200, 200^3 = 8,000,000. Fast.
           
           for w in range(n):
               if remaining[w]:
                   deg[w] = int(adj[w, ~removed].sum())
       
       return scores
