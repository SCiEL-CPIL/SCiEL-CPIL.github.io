import numpy as np
import itertools
import time

D = 11
WEIGHT = 6

def construct():
       start_time = time.time()
       deadline = start_time + 140.0

       # 1. Generate vectors
       vectors = []
       # Map vector index to its data for quick access if needed, but we just need indices
       # We need to map (indices, signs) to vector index for bucketing?
       # No, we just iterate vectors and generate their 4-subsets.
       
       vec_list = []
       for indices in itertools.combinations(range(D), WEIGHT):
           for signs in itertools.product([1, -1], repeat=WEIGHT):
               v = [0] * D
               for idx, s in zip(indices, signs):
                   v[idx] = s
               vec_list.append(v)
               
       N = len(vec_list)
       
       # 2. Build Buckets
       # Key: (tuple of 4 indices, tuple of 4 signs)
       # Value: list of vector indices
       buckets = {}
       
       for i, v in enumerate(vec_list):
           # Get non-zero indices and signs
           nz_indices = []
           nz_signs = []
           for idx in range(D):
               if v[idx] != 0:
                   nz_indices.append(idx)
                   nz_signs.append(v[idx])
           
           # Generate all 4-subsets
           # nz_indices has length 6
           for sub_idx in itertools.combinations(range(WEIGHT), 4):
               # Extract indices and signs
               sub_indices = tuple(nz_indices[j] for j in sub_idx)
               sub_signs = tuple(nz_signs[j] for j in sub_idx)
               key = (sub_indices, sub_signs)
               if key not in buckets:
                   buckets[key] = []
               buckets[key].append(i)
               
       # 3. Build Adjacency
       adj = [set() for _ in range(N)]
       
       for key, members in buckets.items():
           m = len(members)
           if m > 1:
               # All pairs conflict
               for i in range(m):
                   u = members[i]
                   for j in range(i+1, m):
                       v = members[j]
                       adj[u].add(v)
                       adj[v].add(u)
                       
       if time.time() > deadline - 20:
           # Fallback: Return a small valid set?
           # Just return the first 10? They might conflict.
           # Better: Return empty or 1 vector?
           # Score 0 is bad.
           # Let's just return a single vector.
           return np.array(vec_list[:1], dtype=np.int64)

       # 4. Greedy MIS (Min-Degree)
       # Calculate degrees
       degrees = [len(adj[i]) for i in range(N)]
       # Sort indices by degree
       sorted_nodes = sorted(range(N), key=lambda x: degrees[x])
       
       excluded = [False] * N
       mis_indices = []
       
       for u in sorted_nodes:
           if not excluded[u]:
               mis_indices.append(u)
               # Exclude neighbors
               for v in adj[u]:
                   excluded[v] = True
                   
       # 5. Local Improvement
       # Check if any excluded node can be added
       # A node x can be added if no neighbor of x is in mis_indices.
       # We need to know which nodes are in MIS.
       in_mis = [False] * N
       for u in mis_indices:
           in_mis[u] = True
           
       improved = True
       while improved and time.time() < deadline - 5:
           improved = False
           # Iterate through all nodes? Or just excluded ones?
           # We can iterate through all nodes not in MIS.
           for x in range(N):
               if in_mis[x]:
                   continue
               # Check if x can be added
               can_add = True
               for y in adj[x]:
                   if in_mis[y]:
                       can_add = False
                       break
               if can_add:
                   # Add x
                   in_mis[x] = True
                   mis_indices.append(x)
                   improved = True
                   
       # 6. Construct Result
       result_vectors = [vec_list[i] for i in mis_indices]
       return np.array(result_vectors, dtype=np.int64)
