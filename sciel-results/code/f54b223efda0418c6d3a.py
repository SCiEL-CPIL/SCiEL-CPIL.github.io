   import time
   import random
   import itertools

   def _minconflicts_50(deadline):
       N = 50
       # C[i][j] = color
       C = [[0]*N for _ in range(N)]
       # Initialize randomly
       for i in range(N):
           for j in range(i+1, N):
               c = random.randint(0, 3)
               C[i][j] = C[j][i] = c
               
       def is_mono(i, j, k):
           return C[i][j] == C[j][k] == C[i][k]
           
       def count_mono():
           cnt = 0
           for i in range(N):
               for j in range(i+1, N):
                   for k in range(j+1, N):
                       if is_mono(i, j, k):
                           cnt += 1
           return cnt
           
       # Optimization: keep a list of bad triangles?
       # Or just random sampling.
       
       # We will use a simple hill-climbing with random restarts.
       best_n = 0
       best_colors = None
       
       while time.time() < deadline - 5: # Leave 5s for verification
           # Restart with random init
           for i in range(N):
               for j in range(i+1, N):
                   c = random.randint(0, 3)
                   C[i][j] = C[j][i] = c
           
           # MinConflicts
           for step in range(10000): # Max steps per restart
               if time.time() > deadline - 5:
                   break
               
               # Find a bad triangle
               # Random sampling is faster than full scan if few bad triangles
               # But initially many.
               # Let's do a few random checks.
               bad = None
               for _ in range(20):
                   i, j, k = random.sample(range(N), 3)
                   if C[i][j] == C[j][k] == C[i][k]:
                       bad = (i, j, k)
                       break
               
               if not bad:
                   # Check if actually zero?
                   # Full scan is expensive (19600 checks).
                   # If 20 random samples found none, it's likely 0 or very few.
                   # Let's do a full scan to be sure before returning.
                   # But full scan is O(N^3).
                   # We can just return and let the outer verifier check.
                   return C
               
               i, j, k = bad
               # Try to fix by changing one edge
               # Pick random edge from the triangle
               edge_idx = random.randint(0, 2)
               if edge_idx == 0:
                   u, v = i, j
                   w = k
               elif edge_idx == 1:
                   u, v = j, k
                   w = i
               else:
                   u, v = i, k
                   w = j
                   
               old_c = C[u][v]
               best_new_c = old_c
               best_cost = float('inf')
               
               # Evaluate new colors
               for new_c in range(4):
                   if new_c == old_c: continue
                   # Calculate cost change
                   # A triangle (u, v, w') is mono if C[u][v']==C[v][w']==C[u][w']
                   # Changing C[u][v] affects triangles (u, v, w') for all w' != u, v.
                   # We only care about triangles that BECOME mono or STOP being mono.
                   # Actually, we can just compute the number of mono triangles involving (u,v) with new color.
                   # Let's just count how many w' make (u, v, w') mono with new_c.
                   cnt_new = 0
                   for w_prime in range(N):
                       if w_prime == u or w_prime == v: continue
                       if C[u][w_prime] == new_c and C[v][w_prime] == new_c:
                           cnt_new += 1
                           
                   # We want to minimize cnt_new.
                   # But we also remove the current mono triangles of old_c.
                   # The "cost" is the total mono triangles.
                   # Changing (u,v) to new_c:
                   # - Removes mono triangles of old_c involving (u,v).
                   # - Adds mono triangles of new_c involving (u,v).
                   # Note: Triangles not involving (u,v) are unchanged.
                   
                   # So we just need to minimize cnt_new.
                   if cnt_new < best_cost:
                       best_cost = cnt_new
                       best_new_c = new_c
                       
               # If best_cost is still high, maybe we should keep old?
               # MinConflicts usually just picks the best available.
               # If old_c was part of a mono triangle, and new_c creates fewer, it's an improvement.
               # What if new_c creates MORE? We should stick to old or pick random?
               # Standard MinConflicts: pick the value that minimizes conflicts.
               # If old_c has 0 conflicts (which it doesn't, we picked a bad triangle), 
               # but we are changing it because it WAS in a bad triangle.
               # Actually, the cost function is global.
               # Local change: Delta = (New Mono involving uv) - (Old Mono involving uv).
               # We want Delta < 0.
               
               # Let's calculate Old Mono involving (u,v).
               cnt_old = 0
               for w_prime in range(N):
                   if w_prime == u or w_prime == v: continue
                   if C[u][w_prime] == old_c and C[v][w_prime] == old_c:
                       cnt_old += 1
                       
               # We want to minimize (cnt_new - cnt_old).
               # If best_cost (cnt_new) < cnt_old, it's a good move.
               # If not, maybe try another edge or restart?
               # For now, if cnt_new < cnt_old, apply.
               # Else, apply anyway? No, that increases conflicts.
               # Let's only apply if it reduces conflicts.
               # If it doesn't, we might be stuck in a local minimum.
               # We can allow some "bad" moves (simulated annealing) or just restart.
               
               if best_cost < cnt_old:
                   C[u][v] = best_new_c
                   C[v][u] = best_new_c
               else:
                   # If no improvement found, try a random move? Or break?
                   # Let's try a random move to escape local min.
                   if random.random() < 0.1:
                       C[u][v] = random.randint(0, 3)
                       C[v][u] = C[u][v]
           
       return None

   def _verify(n, colors):
       # Same as before
       pass

   def construct():
       deadline = time.time() + 140.0
       # Try 50
       C = _minconflicts_50(deadline)
       if C:
           # Convert to list
           colors = []
           for i in range(50):
               for j in range(i+1, 50):
                   colors.append(C[i][j])
           if _verify(50, colors):
               return {'n': 50, 'colors': colors}
       
       # Fallback 32
       return _seed32()
   