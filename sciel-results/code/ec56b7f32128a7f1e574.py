import time
import random
import itertools

def construct():
       start_time = time.time()
       deadline = 300.0
       
       # 1. Construct P(37)
       q = 37
       residues = {(x * x) % q for x in range(1, q)}
       n_base = q
       adj = [0] * n_base
       for i in range(n_base):
           for j in range(i + 1, n_base):
               if (j - i) % q in residues:
                   adj[i] |= (1 << j)
                   adj[j] |= (1 << i)
                   
       # 2. Add 5 new vertices
       n_new = 5
       n = n_base + n_new
       # Initialize new vertices as isolated
       for i in range(n_base, n):
           adj.append(0)
           
       # 3. Randomize connections for new vertices
       # We will use a simple local search
       
       def find_clique_or_independent():
           # Returns a tuple (type, vertices) where type is 'clique' or 'independent'
           # Or None if not found
           
           # Check for K5
           # Iterate combinations
           # Optimization: Use bitsets
           
           # We can iterate over all 5-subsets
           # This is expensive in pure Python loops.
           # Better: Iterate over K4s and extend.
           
           # Let's stick to a simpler check for now, might be slow.
           # \binom{42}{5} = 850668.
           # 850k iterations. In Python, ~0.5s per million simple ops.
           # So ~0.4s per full scan.
           # We can do ~700 scans in 300s.
           
           # Let's optimize the check.
           
           # Find K5:
           # For each i, j, k, l, check if there is m > l such that m is connected to i,j,k,l.
           
           # Let's implement a recursive backtracking to find K5 or I5.
           
           def has_clique(size, vertices):
               if size == 0: return True
               if not vertices: return False
               # Pick a vertex
               v = vertices[0]
               rest = vertices[1:]
               # Neighbors of v in rest
               neigh = [u for u in rest if (adj[v] >> u) & 1]
               if has_clique(size - 1, neigh):
                   return True
               # Non-neighbors of v in rest
               non_neigh = [u for u in rest if not ((adj[v] >> u) & 1)]
               if has_clique(size - 1, non_neigh):
                   return True
               return False
               
           # This is for finding ANY clique of size 5.
           # But we need to find it to break it.
           
           # Let's just do a full scan with bitsets for speed.
           
           # K5 check
           for i in range(n):
               for j in range(i+1, n):
                   if not ((adj[i] >> j) & 1): continue
                   for k in range(j+1, n):
                       if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1): continue
                       for l in range(k+1, n):
                           if not ((adj[i] >> l) & 1) or not ((adj[j] >> l) & 1) or not ((adj[k] >> l) & 1): continue
                           # Check if there is m > l connected to i,j,k,l
                           mask = adj[i] & adj[j] & adj[k] & adj[l]
                           # We need m > l
                           mask &= ~((1 << (l+1)) - 1)
                           if mask:
                               m = mask.bit_length() - 1
                               return ('clique', (i, j, k, l, m))
           
           # I5 check
           # Similar, but check for non-edges
           # Or just run the same on the complement?
           # Complement adjacency: ~adj[i] & ((1<<n)-1)
           # But we have to exclude self-loops.
           
           for i in range(n):
               for j in range(i+1, n):
                   if ((adj[i] >> j) & 1): continue # Must be non-edge
                   for k in range(j+1, n):
                       if ((adj[i] >> k) & 1) or ((adj[j] >> k) & 1): continue
                       for l in range(k+1, n):
                           if ((adj[i] >> l) & 1) or ((adj[j] >> l) & 1) or ((adj[k] >> l) & 1): continue
                           # Check if there is m > l non-connected to i,j,k,l
                           # Non-neighbors of i, j, k, l
                           # mask = (~adj[i]) & (~adj[j]) & (~adj[k]) & (~adj[l])
                           # But we need to be careful with bits beyond n
                           mask = (~adj[i]) & (~adj[j]) & (~adj[k]) & (~adj[l])
                           mask &= ~((1 << (l+1)) - 1)
                           # Also remove self bits? No, m > l so m != i,j,k,l
                           # But we need to ensure m is a valid vertex
                           mask &= ((1 << n) - 1)
                           if mask:
                               m = mask.bit_length() - 1
                               return ('independent', (i, j, k, l, m))
           
           return None

       # Initial randomization
       # Connect new vertices to old ones and each other
       for i in range(n_base, n):
           for j in range(n):
               if i == j: continue
               if random.random() < 0.5:
                   adj[i] |= (1 << j)
                   adj[j] |= (1 << i)
                   
       # Local Search
       best_n = n_base
       best_edges = [(i, j) for i in range(n_base) for j in range(i+1, n_base) if (adj[i] >> j) & 1]
       
       while time.time() < deadline:
           result = find_clique_or_independent()
           if result is None:
               # Found a valid graph
               edges = []
               for i in range(n):
                   for j in range(i+1, n):
                       if (adj[i] >> j) & 1:
                           edges.append((i, j))
               return n, edges
           
           # Break the found set
           typ, verts = result
           # Pick a random edge to flip
           # If clique, flip an edge inside the clique
           # If independent, flip a non-edge inside the independent set
           
           if typ == 'clique':
               # Pick two vertices from verts
               v1, v2 = random.sample(verts, 2)
               # Flip edge (v1, v2)
               if (adj[v1] >> v2) & 1:
                   adj[v1] ^= (1 << v2)
                   adj[v2] ^= (1 << v1)
               else:
                   # Should not happen for clique
                   adj[v1] |= (1 << v2)
                   adj[v2] |= (1 << v1)
           else:
               v1, v2 = random.sample(verts, 2)
               if not ((adj[v1] >> v2) & 1):
                   adj[v1] |= (1 << v2)
                   adj[v2] |= (1 << v1)
               else:
                   adj[v1] ^= (1 << v2)
                   adj[v2] ^= (1 << v1)
                   
       # Return best found (seed)
       return n_base, best_edges
