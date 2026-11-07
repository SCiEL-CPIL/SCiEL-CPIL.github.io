   import time
   
   def construct():
       n = 24
       # Base edges
       edges = [(i, i+1) for i in range(n-1)]
       adj = [[False]*n for _ in range(n)]
       for u, v in edges:
           adj[u][v] = True
           adj[v][u] = True
           
       def get_diameter():
           # BFS from each node
           max_d = 0
           for s in range(n):
               dist = [-1]*n
               dist[s] = 0
               q = [s]
               for u in q:
                   for v in range(n):
                       if adj[u][v] and dist[v] == -1:
                           dist[v] = dist[u] + 1
                           q.append(v)
                           if dist[v] > max_d:
                               max_d = dist[v]
           return max_d
           
       def is_triangle_free(u, v):
           if adj[u][v]: return False
           for w in range(n):
               if adj[u][w] and adj[v][w]:
                   return False
           return True
           
       start_time = time.time()
       deadline = 50.0 # 10s margin
       
       added = []
       while get_diameter() > 4:
           if time.time() > deadline:
               break
               
           # Find farthest pair
           max_dist = 0
           u_star, v_star = -1, -1
           for s in range(n):
               dist = [-1]*n
               dist[s] = 0
               q = [s]
               for u in q:
                   for v in range(n):
                       if adj[u][v] and dist[v] == -1:
                           dist[v] = dist[u] + 1
                           q.append(v)
                           if dist[v] > max_dist:
                               max_dist = dist[v]
                               u_star, v_star = s, v
                               
           # Try to add edge between u_star and v_star?
           # Or try all pairs?
           # Greedy: try to add edge that reduces diameter.
           # Simple heuristic: add edge between u_star and v_star if valid.
           if is_triangle_free(u_star, v_star):
               adj[u_star][v_star] = True
               adj[v_star][u_star] = True
               added.append([u_star, v_star])
           else:
               # Find another valid edge that helps
               # Try neighbors
               found = False
               for du in range(-2, 3):
                   for dv in range(-2, 3):
                       nu = u_star + du
                       nv = v_star + dv
                       if 0 <= nu < n and 0 <= nv < n and nu != nv:
                           if is_triangle_free(nu, nv):
                               # Check if it reduces diameter
                               # Just add it for now
                               adj[nu][nv] = True
                               adj[nv][nu] = True
                               added.append([nu, nv])
                               found = True
                               break
                       if found: break
                   if found: break
               if not found:
                   break # Stuck
                   
       return added
   