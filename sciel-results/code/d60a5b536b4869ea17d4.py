import numpy as np
import itertools
import time

D = 11

def construct():
       start_time = time.time()
       
       # 1. Generate D11 Minimal Vectors
       # Type 1: Weight 1, +/- 2
       # Type 2: Weight 4, +/- 1, even sum
       
       vectors = []
       
       # Type 1
       for i in range(D):
           for s in [1, -1]:
               v = [0]*D
               v[i] = 2*s
               vectors.append(v)
               
       # Type 2
       for pos in itertools.combinations(range(D), 4):
           # Generate all sign combos with even number of -1s
           # There are 8 such combos
           signs = []
           for mask in range(16):
               # count bits set in mask
               if bin(mask).count('1') % 2 == 0:
                   signs.append([1 if (mask & (1<<j)) == 0 else -1 for j in range(4)])
           
           for s in signs:
               v = [0]*D
               for p, val in zip(pos, s):
                   v[p] = val
               vectors.append(v)
               
       V = np.array(vectors, dtype=np.int64)
       N = len(V)
       
       # 2. Compute Conflict Matrix
       # dot = V @ V.T
       # conflict = dot > 2
       dots = V @ V.T
       conflicts = dots > 2
       np.fill_diagonal(conflicts, False)
       
       # 3. Find Independent Set
       # Use Greedy Min-Degree
       in_set = np.zeros(N, dtype=bool)
       alive = np.ones(N, dtype=bool)
       
       # Precompute degrees
       degrees = conflicts.sum(axis=1).astype(np.int32)
       
       while np.any(alive):
           # Find min degree among alive
           alive_idx = np.where(alive)[0]
           if len(alive_idx) == 0: break
           
           min_deg = np.min(degrees[alive_idx])
           cands = alive_idx[degrees[alive_idx] == min_deg]
           # Pick one (random or first)
           chosen = cands[0]
           
           in_set[chosen] = True
           
           # Remove chosen and its neighbors from alive
           nbrs = np.where(conflicts[chosen])[0]
           to_remove = np.unique(np.concatenate(([chosen], nbrs)))
           alive[to_remove] = False
           
           # Update degrees of remaining alive nodes
           rem = np.where(alive)[0]
           if len(rem) > 0:
               # Degree of u decreases by number of its neighbors in to_remove
               # to_remove includes chosen and its neighbors
               # We just need to subtract 1 for each neighbor in to_remove
               # Actually, we removed 'to_remove' from the graph.
               # For any u in rem, its new degree is old degree - (number of neighbors in to_remove)
               # Since to_remove is independent set of removed nodes? No.
               # to_remove = {chosen} U N(chosen).
               # For u in rem, u is not adjacent to chosen (otherwise u would be in N(chosen) and removed).
               # So u is not adjacent to chosen.
               # u might be adjacent to some nodes in N(chosen).
               # Let's just recompute or update carefully.
               # Simpler: degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
               degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
               
       result = V[in_set]
       return result
