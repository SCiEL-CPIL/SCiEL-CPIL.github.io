import itertools
import numpy as np

def construct():
       D = 11
       # Generate all 4-subsets of range(11)
       supports = list(itertools.combinations(range(D), 4))
       n_sup = len(supports)
       
       # Build conflict graph: edge if intersection size == 3
       # adjacency list
       adj = [set() for _ in range(n_sup)]
       for i in range(n_sup):
           s_i = set(supports[i])
           for j in range(i+1, n_sup):
               s_j = set(supports[j])
               if len(s_i.intersection(s_j)) == 3:
                   adj[i].add(j)
                   adj[j].add(i)
                   
       # Greedy independent set with random restarts
       best_set = []
       for _ in range(100): # 100 restarts
           remaining = set(range(n_sup))
           current_set = []
           while remaining:
               # Pick vertex with min degree in remaining subgraph
               min_deg = float('inf')
               best_v = None
               for v in remaining:
                   deg = len(adj[v].intersection(remaining))
                   if deg < min_deg:
                       min_deg = deg
                       best_v = v
                       if deg == 0: break
               current_set.append(best_v)
               # Remove best_v and its neighbors
               to_remove = {best_v}.union(adj[best_v].intersection(remaining))
               remaining.difference_update(to_remove)
           
           if len(current_set) > len(best_set):
               best_set = current_set
               
       # Construct vectors
       vectors = []
       for idx in best_set:
           sup = supports[idx]
           # Generate all 16 sign patterns for these 4 indices
           for signs in itertools.product([1, -1], repeat=4):
               v = [0] * D
               for k, s in enumerate(sup):
                   v[s] = signs[k]
               vectors.append(v)
               
       return np.array(vectors, dtype=np.int64)
