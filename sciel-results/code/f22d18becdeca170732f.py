import time
import sys
import random

def construct():
       n = 15
       start_time = time.time()
       deadline = 80.0 # Safety margin
       
       # Initial Graph: K_7,8
       # A: 0-6, B: 7-14
       edges = set()
       for u in range(7):
           for v in range(7, 15):
               edges.add((u, v))
               
       def get_degree_set(edges):
           deg = [0]*n
           for u,v in edges:
               deg[u]+=1
               deg[v]+=1
           return deg
       
       def is_valid(edges):
           deg = get_degree_set(edges)
           adj = [set() for _ in range(n)]
           for u,v in edges:
               adj[u].add(v)
               adj[v].add(u)
               
           # Check all paths s-u-v-t
           # s != t
           # deg[s] != deg[t]
           
           # Optimization: Group vertices by degree.
           # For each degree group, check if any pair has a path of length 3.
           
           from collections import defaultdict
           deg_groups = defaultdict(list)
           for i in range(n):
               deg_groups[deg[i]].append(i)
               
           for d, verts in deg_groups.items():
               if len(verts) < 2:
                   continue
               # Check all pairs in verts
               for i in range(len(verts)):
                   s = verts[i]
                   for j in range(i+1, len(verts)):
                       t = verts[j]
                       # Check if path s-u-v-t exists
                       # u in adj[s], v in adj[u] \cap adj[t]
                       # v != s, v != t (guaranteed if simple path and s!=t?
                       # If v==s, path s-u-s-t. Not simple if s!=t? 
                       # Simple path: distinct vertices.
                       # So v != s and v != t.
                       
                       found = False
                       for u in adj[s]:
                           if u == t: continue # u != t for simple path? 
                           # Path s-u-v-t. u must be distinct from s,t,v.
                           # If u==t, path s-t-v-t. Not simple.
                           for v in adj[u]:
                               if v == s or v == t: continue
                               if v in adj[t]:
                                   # Path s-u-v-t exists
                                   found = True
                                   break
                           if found: break
                       if found:
                           return False
           return True

       # Greedy Addition
       candidates = []
       for u in range(n):
           for v in range(u+1, n):
               if (u,v) not in edges:
                   candidates.append((u,v))
                   
       improved = True
       while improved and time.time() < deadline:
           improved = False
           random.shuffle(candidates)
           for e in candidates[:]:
               if time.time() > deadline: break
               u, v = e
               if (u,v) in edges: continue
               
               # Try adding
               edges.add((u,v))
               if is_valid(edges):
                   candidates.remove(e)
                   improved = True
               else:
                   edges.remove((u,v))
                   
       # Convert to list of lists
       final_edges = [[int(u), int(v)] for u,v in edges]
       return final_edges
