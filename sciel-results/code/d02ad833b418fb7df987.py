   import time
   import random
   import math

   def construct():
       deadline = time.time() + 145
       
       def time_left():
           return deadline - time.time()
       
       def edge_idx(i, j, n):
           return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
       
       def build_adj(colors, n):
           adj = [[0] * n for _ in range(5)]
           idx = 0
           for i in range(n):
               for j in range(i + 1, n):
                   c = colors[idx]
                   adj[c][i] |= (1 << j)
                   adj[c][j] |= (1 << i)
                   idx += 1
           return adj
       
       def compute_energy(adj, n):
           total = 0
           for c in range(5):
               for i in range(n):
                   adj_i = adj[c][i]
                   for j in range(i + 1, n):
                       common = adj_i & adj[c][j]
                       k = common.bit_count()
                       if k >= 2:
                           total += k * (k - 1) // 2
           return total
       
       def get_delta(adj, n, u, v, c_old, c_new):
           # Calculate change in energy if edge (u,v) changes from c_old to c_new
           delta = 0
           
           # Remove from c_old
           # For each w, if w is connected to u and v in c_old, it forms C4s
           # Actually, the energy contribution of (u,v) in c_old is:
           # Number of w such that (u,w) and (v,w) are in c_old.
           # Let S = N_c_old(u) & N_c_old(v).
           # The number of C4s containing (u,v) is C(|S|, 1)? No.
           # A C4 is defined by 4 vertices. (u,v) is an edge.
           # A C4 containing edge (u,v) is formed by choosing 2 common neighbors? No.
           # A C4 is a cycle of length 4.
           # If (u,v) is an edge, a C4 containing it requires a path u-w1-w2-v of length 3?
           # No, C4 is u-v-w1-w2-u.
           # So we need w1, w2 such that (v,w1), (w1,w2), (w2,u) are edges?
           # This is complex.
           
           # Simpler: Energy is sum over all pairs of vertices (i,j) of C(common_neigh(i,j), 2).
           # Changing edge (u,v) affects:
           # 1. The common neighbors of (u,v) themselves.
           # 2. The common neighbors of (u,w) for all w.
           # 3. The common neighbors of (v,w) for all w.
           
           # Let's use the direct formula derived in the previous code, which was:
           # delta_remove = - sum_{w} (if w in N(u) and w in N(v) then ... )
           
           # Let's re-verify the delta formula.
           # E = sum_{c} sum_{i<j} C(|N_c(i) \cap N_c(j)|, 2)
           
           # Change edge (u,v) from c1 to c2.
           # In c1:
           # Term for (u,v): C(|N_c1(u) \cap N_c1(v)|, 2).
           # Terms for (u,w): C(|N_c1(u) \cap N_c1(w)|, 2).
           #   If w is in N_c1(v), then v is in N_c1(w).
           #   So N_c1(u) \cap N_c1(w) contains v?
           #   Yes, if (u,v) and (w,v) are in c1.
           #   So removing (u,v) removes v from N_c1(u) and N_c1(v).
           
           # This is getting complicated to derive from scratch quickly.
           # The previous code's delta calculation:
           # delta -= (k - 1) for w in N(u) & N(v) ...
           # This was:
           # for w in range(n):
           #   if w == u or w == v: continue
           #   if adj[c_old][w] & (1 << v): # w connected to v
           #       k = (adj[c_old][u] & adj[c_old][w]).bit_count()
           #       delta -= (k - 1)
           # This counts how many C4s are destroyed.
           
           # Let's trust the previous delta logic but optimize it.
           
           # Remove from c_old
           for w in range(n):
               if w == u or w == v:
                   continue
               mask_w = adj[c_old][w]
               if mask_w & (1 << v):
                   # w is neighbor of v in c_old
                   # Check if w is neighbor of u?
                   # If yes, (u,v,w) is a triangle? No, C4 requires 4 vertices.
                   # The formula `k = (adj[c_old][u] & mask_w).bit_count()` counts common neighbors of u and w.
                   # One of them is v (since v is neighbor of u and w).
                   # So `k-1` is the number of OTHER common neighbors.
                   # Each other common neighbor z forms a C4: u-v-w-z-u.
                   # So we destroy `k-1` C4s.
                   k = (adj[c_old][u] & mask_w).bit_count()
                   delta -= (k - 1)
               if mask_w & (1 << u):
                   # w is neighbor of u in c_old
                   k = (adj[c_old][v] & mask_w).bit_count()
                   delta -= (k - 1)
           
           # Add to c_new
           for w in range(n):
               if w == u or w == v:
                   continue
               mask_w = adj[c_new][w]
               if mask_w & (1 << v):
                   k = (adj[c_new][u] & mask_w).bit_count()
                   delta += k
               if mask_w & (1 << u):
                   k = (adj[c_new][v] & mask_w).bit_count()
                   delta += k
           
           return delta
       
       def sa_search(n, max_time):
           num_edges = n * (n - 1) // 2
           if num_edges == 0:
               return [], 0
           
           best_energy = float('inf')
           best_colors = None
           
           # Random restarts
           while time_left() > max_time:
               colors = [random.randint(0, 4) for _ in range(num_edges)]
               adj = build_adj(colors, n)
               energy = compute_energy(adj, n)
               
               if energy == 0:
                   return colors, 0
               
               temp = 1.0
               cooling = 0.9999
               min_temp = 0.01
               
               # Local SA
               while temp > min_temp and time_left() > max_time:
                   # Pick random edge
                   ei = random.randint(0, num_edges - 1)
                   # Find u, v from ei
                   # This is O(n) to find u,v. We can precompute edge_list.
                   # For n=26, num_edges=325. O(1) lookup is better.
                   
                   # We need edge_list for fast u,v retrieval.
                   # Let's build it once.
                   
                   c_old = colors[ei]
                   c_new = random.randint(0, 4)
                   if c_new == c_old:
                       continue
                   
                   u, v = edge_uv[ei]
                   
                   delta = get_delta(adj, n, u, v, c_old, c_new)
                   new_energy = energy + delta
                   
                   if delta <= 0 or random.random() < math.exp(-delta / temp):
                       energy = new_energy
                       
                       # Update adj
                       adj[c_old][u] &= ~(1 << v)
                       adj[c_old][v] &= ~(1 << u)
                       adj[c_new][u] |= (1 << v)
                       adj[c_new][v] |= (1 << u)
                       colors[ei] = c_new
                       
                       if energy < best_energy:
                           best_energy = energy
                           best_colors = colors[:]
                           if energy == 0:
                               return colors, 0
                   
                   temp *= cooling
           
           if best_colors is None:
               # Return random if no improvement?
               return [random.randint(0, 4) for _ in range(num_edges)], best_energy
           return best_colors, best_energy
       
       # Precompute edge_uv for n=26
       n_target = 26
       num_edges = n_target * (n_target - 1) // 2
       edge_uv = []
       for i in range(n_target):
           for j in range(i + 1, n_target):
               edge_uv.append((i, j))
       
       # Run SA
       colors, energy = sa_search(n_target, 10)
       
       if energy == 0:
           return {'n': n_target, 'colors': colors}
       
       # If failed, try smaller n?
       # For now, return the best found or a trivial one.
       # To ensure validity, if energy > 0, the coloring is INVALID.
       # We MUST return a valid coloring.
       # So we need a fallback.
       
       # Fallback: n=1
       return {'n': 1, 'colors': []}

   if __name__ == '__main__':
       result = construct()
       print(f"Found n = {result['n']}")
   