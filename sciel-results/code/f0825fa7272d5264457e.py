import random
import time

def construct():
         n = 40
         start_time = time.time()
         best_edges = 0
         best_graph = None
         
         # Bitset adjacency
         # adj[i] is an integer where bit j is 1 if i-j is an edge
         
         def creates_c6(u, v, adj):
             # Check if adding (u,v) creates a C6
             # Need path u-a-b-c-d-v of length 4 in current graph
             # a in N(u), d in N(v)
             # b in N(a), c in N(b)
             # c connected to d
             # All distinct and not u,v
             
             # Pre-compute N(u) and N(v) as bitsets
             nu = adj[u]
             nv = adj[v]
             
             # Mask out u and v
             mask_u = ~(1 << u)
             mask_v = ~(1 << v)
             mask_uv = mask_u & mask_v
             
             # a must be in N(u) \ {v}
             # d must be in N(v) \ {u}
             
             # We iterate over possible a
             a_candidates = nu & mask_v # N(u) without v
             # We iterate over possible d
             d_candidates = nv & mask_u # N(v) without u
             
             # For each a, we look for a path of length 3 from a to any d in d_candidates
             # Path a-b-c-d
             # b in N(a) \ {u}
             # c in N(b) \ {a}
             # d in N(c) & d_candidates
             # Also b,c must not be u or v? 
             # If b=u, path is u-a-u... not simple.
             # So b,c must be in mask_uv.
             
             # Optimization:
             # For each a, compute N(a) & mask_uv (excluding u, v)
             # Then for each b in that set, compute N(b) & mask_uv
             # Then check if N(c) intersects d_candidates.
             
             # To speed up, we can pre-compute "2-step neighbors" or similar?
             # Or just iterate.
             
             # Iterate a
             a_bit = 1
             while a_candidates:
                 # Get lowest set bit
                 lsb = a_candidates & -a_candidates
                 a = lsb.bit_length() - 1
                 a_candidates ^= lsb
                 
                 # b candidates: N(a) & mask_uv
                 b_candidates = adj[a] & mask_uv
                 
                 # Iterate b
                 while b_candidates:
                     lsb_b = b_candidates & -b_candidates
                     b = lsb_b.bit_length() - 1
                     b_candidates ^= lsb_b
                     
                     # c candidates: N(b) & mask_uv
                     c_candidates = adj[b] & mask_uv
                     
                     # For each c, check if N(c) intersects d_candidates
                     # We can check if (N(c) & d_candidates) != 0
                     # But we must ensure c is distinct from a?
                     # b_candidates excluded a? No, adj[b] might contain a.
                     # We need simple path.
                     # If c == a, path u-a-b-a-d-v? No, a-b-a is backtrack.
                     # So c must not be a.
                     # Also c must not be u,v (handled by mask_uv).
                     # Also c must not be b? N(b) doesn't contain b (simple graph).
                     
                     # So c_candidates should exclude a.
                     # c_candidates = adj[b] & mask_uv & ~(1 << a)
                     
                     c_candidates = adj[b] & mask_uv & ~(1 << a)
                     
                     while c_candidates:
                         lsb_c = c_candidates & -c_candidates
                         c = lsb_c.bit_length() - 1
                         c_candidates ^= lsb_c
                         
                         # Check if N(c) intersects d_candidates
                         if adj[c] & d_candidates:
                             return True
             
             return False
         
         # Greedy Randomized Construction
         def run_greedy():
             adj = [0] * n
             edges = set()
             
             # Initialize with some random edges to break symmetry?
             # Or start empty.
             
             # Candidate edges
             candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
             random.shuffle(candidates)
             
             # Try to add edges
             for u, v in candidates:
                 if adj[u] & (1 << v):
                     continue
                 if not creates_c6(u, v, adj):
                     adj[u] |= (1 << v)
                     adj[v] |= (1 << u)
                     edges.add((u, v))
             
             return len(edges), adj, edges
         
         # Run multiple times
         best_count = 0
         best_adj = None
         
         # Time limit 120s. Leave some margin.
         end_time = start_time + 110
         
         while time.time() < end_time:
             count, adj, edges = run_greedy()
             if count > best_count:
                 best_count = count
                 best_adj = adj
                 
                 # Optional: Local search to improve?
                 # Try swapping edges or adding more?
                 # The greedy might stop because remaining candidates create C6.
                 # We can try to remove an edge and re-add?
                 # Or just restart.
         
         # Convert best_adj to list of edges
         result = []
         for u in range(n):
             for v in range(u + 1, n):
                 if best_adj[u] & (1 << v):
                     result.append([u, v])
                     
         return result
