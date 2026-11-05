import time
import random
import itertools

def construct():
       start_time = time.time()
       timeout = 110
       
       def is_triangle_free(adj, n):
           for u in range(n):
               for v in adj[u]:
                   if u < v:
                       for w in adj[u]:
                           if v < w and w in adj[v]:
                               return False
           return True
           
       def max_independent_set_size(adj, n, limit=10):
           # Branch and bound
           # Returns size of max independent set
           # If >= limit, return limit
           pass
           
       # Try n from 39 down to 27
       for n in range(39, 26, -1):
           if time.time() - start_time > timeout - 5:
               break
           
           # Randomized Local Search
           best_alpha = 100
           best_edges = None
           
           for trial in range(50):
               if time.time() - start_time > timeout - 5:
                   break
                   
               # Initialize
               adj = [set() for _ in range(n)]
               # Start with empty graph? Or random?
               # Random graph with p=0.5
               for u in range(n):
                   for v in range(u+1, n):
                       if random.random() < 0.5:
                           adj[u].add(v)
                           adj[v].add(u)
                           
               # Remove triangles
               changed = True
               while changed:
                   changed = False
                   for u in range(n):
                       for v in list(adj[u]):
                           if u < v:
                               for w in list(adj[u]):
                                   if v < w and w in adj[v]:
                                       # Remove edge (v,w)
                                       adj[v].remove(w)
                                       adj[w].remove(v)
                                       changed = True
                                       break
                                   if w == v: continue
                       if changed: break
                   if changed: break
                   
               # Check alpha
               # Simple greedy upper bound?
               # If greedy finds >= 10, skip
               # Else, try to add edges to reduce alpha?
               # For now, just check if it's valid
               # Use a quick check
               if is_triangle_free(adj, n):
                   # Estimate alpha
                   # If alpha < 10, valid
                   # Use a simple B&B for small n
                   alpha = estimate_alpha(adj, n)
                   if alpha < 10:
                       best_alpha = alpha
                       best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
                       
           if best_edges is not None:
               return {'n': n, 'edges': best_edges}
               
       # Fallback to seed
       n = 26
       edges = []
       for i in range(n):
           for j in range(i + 1, n):
               if (j - i) % n % 3 == 1:
                   edges.append([i, j])
       return {'n': n, 'edges': edges}
