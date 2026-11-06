   import random
   import time
   
   def construct():
       n = 16
       
       # 1. Generate a base 4-chromatic triangle-free graph
       # We use the Grötzsch graph (11 vertices) and add 5 vertices
       # connected to an independent set to keep it 4-chromatic?
       # Actually, just adding isolated vertices keeps it 4-chromatic.
       # But we want a dense base to reduce search space.
       
       # Let's use the current program's greedy output as the base.
       # We can replicate the greedy logic quickly to get a base.
       
       def get_base_graph():
           edges = []
           # C5
           for i in range(5):
               edges.append([i, (i + 1) % 5])
           # Mycielski
           for i in range(5):
               u = i
               v = (i + 1) % 5
               edges.append([u, 5 + v])
               edges.append([v, 5 + u])
           for i in range(5):
               edges.append([5 + i, 10])
           
           adj = {i: set() for i in range(n)}
           edge_set = set()
           for u, v in edges:
               adj[u].add(v)
               adj[v].add(u)
               edge_set.add(frozenset((u, v)))
               
           # Greedy add 11-15
           for v in range(11, 16):
               # Find max independent set in adj[0..v-1]
               # Simple greedy
               candidates = list(range(v))
               random.shuffle(candidates)
               iset = set()
               for u in candidates:
                   if not (adj[u] & iset):
                       iset.add(u)
               for u in iset:
                   edges.append([v, u])
                   adj[v].add(u)
                   adj[u].add(v)
                   edge_set.add(frozenset((v, u)))
           return edges
       
       base_edges = get_base_graph()
       
       # 2. Exact Search to maximize edges in triangle-free supergraph
       # Candidate edges are all pairs not in base_edges
       all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
       base_set = set(frozenset(e) for e in base_edges)
       candidates = [e for e in all_edges if frozenset(e) not in base_set]
       
       # Sort candidates to improve pruning?
       # Maybe sort by degree sum?
       
       adj = {i: set() for i in range(n)}
       for u, v in base_edges:
           adj[u].add(v)
           adj[v].add(u)
           
       current_edges = len(base_edges)
       best_edges = current_edges
       best_adj = [set(s) for s in adj.values()] # Deep copy
       
       # Common neighbors count for each pair (u, v)
       # cn[u][v] = number of w such that (u,w) and (v,w) in graph
       cn = [[0]*n for _ in range(n)]
       for u in range(n):
           for v in adj[u]:
               for w in adj[u]:
                   if v != w:
                       # w is a common neighbor of u and v?
                       # No, if (u,v) is not an edge, we care about cn[u][v].
                       # Let's compute cn for all pairs.
                       pass
       
       # Recompute cn from scratch for base
       for u in range(n):
           for v in range(u+1, n):
               if (u, v) not in base_set and frozenset((v,u)) not in base_set: # Just check adj
                   if v not in adj[u]:
                       common = adj[u] & adj[v]
                       cn[u][v] = len(common)
                       cn[v][u] = len(common)
       
       def backtrack(idx):
           nonlocal best_edges, best_adj
           
           if idx == len(candidates):
               if current_edges > best_edges:
                   best_edges = current_edges
                   best_adj = [set(s) for s in adj.values()]
               return
           
           # Pruning: even if we add all remaining, can we beat best?
           if current_edges + (len(candidates) - idx) <= best_edges:
               return
           
           u, v = candidates[idx]
           
           # Option 1: Add edge (u, v)
           if cn[u][v] == 0:
               # Add edge
               adj[u].add(v)
               adj[v].add(u)
               
               # Update cn
               # For any neighbor x of u, cn[x][v] increases by 1 (because u is now a common neighbor of x and v)
               # For any neighbor y of v, cn[u][y] increases by 1
               # And cn[x][y] for x in N(u), y in N(v) increases by 2?
               # No, if we add (u,v), then for any x in N(u) and y in N(v):
               # u is a common neighbor of x and v.
               # v is a common neighbor of u and y.
               # And u, v are NOT common neighbors of x, y unless x-u-v-y path?
               # Let's be precise.
               # cn[a][b] = |N(a) \cap N(b)|.
               # New N(u) = N(u) \cup {v}.
               # New N(v) = N(v) \cup {u}.
               
               # Update cn[x][v] for x in N(u):
               # N(x) is unchanged. N(v) gained u.
               # So if u in N(x), then cn[x][v] increases by 1.
               # But x is in N(u), so u in N(x) is true.
               # So for all x in N(u) (old N(u)), cn[x][v] += 1.
               
               # Similarly, for all y in N(v) (old N(v)), cn[u][y] += 1.
               
               # What about cn[x][y] for x in N(u), y in N(v)?
               # N(x) unchanged, N(y) unchanged.
               # But u is in N(x) and u is in N(v)? No, u is in N(x) and v is in N(y).
               # Is u in N(y)? No. Is v in N(x)? No.
               # So cn[x][y] is unchanged?
               # Wait. cn[x][y] = |N(x) \cap N(y)|.
               # N(x) and N(y) did not change. So cn[x][y] is unchanged.
               
               # However, we must also update cn for pairs involving u and v themselves?
               # cn[u][v] becomes 0 (since we added the edge, it's no longer a candidate, but logically the count of common neighbors is what matters for *other* edges).
               
               # Let's just update the counts carefully.
               old_N_u = list(adj[u] - {v}) # Before adding v, N(u) didn't have v
               old_N_v = list(adj[v] - {u})
               
               for x in old_N_u:
                   cn[x][v] += 1
                   cn[v][x] += 1
                   
               for y in old_N_v:
                   cn[u][y] += 1
                   cn[y][u] += 1
                   
               # Recurse
               current_edges += 1
               backtrack(idx + 1)
               current_edges -= 1
               
               # Undo
               for x in old_N_u:
                   cn[x][v] -= 1
                   cn[v][x] -= 1
                   
               for y in old_N_v:
                   cn[u][y] -= 1
                   cn[y][u] -= 1
                   
               adj[u].remove(v)
               adj[v].remove(u)
           
           # Option 2: Skip edge (u, v)
           backtrack(idx + 1)
           
       backtrack(0)
       
       # Convert best_adj back to edge list
       result_edges = []
       for i in range(n):
           for j in best_adj[i]:
               if i < j:
                   result_edges.append([i, j])
       return result_edges
   