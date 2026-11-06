import time
import sys

def construct():
       start_time = time.time()
       deadline = start_time + 110 # Safety margin for 120s limit
       
       # 1. Construct Grötzsch graph (11 vertices)
       # Vertices 0-4: C5
       # Vertices 5-9: Copies
       # Vertex 10: Root
       adj = [0] * 11
       # C5
       for i in range(5):
           u = i
           v = (i + 1) % 5
           adj[u] |= 1 << v
           adj[v] |= 1 << u
       
       # Copies
       for i in range(5):
           u = i
           v1 = (i + 1) % 5
           v2 = (i - 1) % 5
           w = 5 + i
           adj[w] |= (1 << v1) | (1 << v2)
           adj[v1] |= 1 << w
           adj[v2] |= 1 << w
           
       # Root
       for i in range(5):
           w = 5 + i
           adj[10] |= 1 << w
           adj[w] |= 1 << 10
           
       # Verify Grötzsch is 4-chromatic (it is)
       
       # 2. Search for optimal extensions to 16 vertices
       # State: adj list (list of ints), num_vertices
       # We will use a list of states to explore
       
       # Helper: Check 3-colorability using backtracking with bitmasks
       def is_3_colorable(adj_list, n):
           # adj_list is list of ints for 0..n-1
           colors = [-1] * n
           # Order vertices by degree descending for faster pruning
           vertices = list(range(n))
           vertices.sort(key=lambda x: bin(adj_list[x]).count('1'), reverse=True)
           
           def backtrack(idx):
               if idx == len(vertices):
                   return True
               v = vertices[idx]
               # Try colors 0, 1, 2
               for c in range(3):
                   # Check if color c is available for v
                   # i.e., no neighbor has color c
                   valid = True
                   mask = adj_list[v]
                   # Check neighbors
                   for u in range(n):
                       if (mask >> u) & 1:
                           if colors[u] == c:
                               valid = False
                               break
                   if valid:
                       colors[v] = c
                       if backtrack(idx + 1):
                           return True
                       colors[v] = -1
               return False
           
           return backtrack(0)
           
       # Helper: Get independent sets of adj_list
       # We only need maximal independent sets? Or all?
       # To maximize edges, we prefer large independent sets.
       # But we need to explore to ensure we don't miss a path to a better solution.
       
       # Optimization: 
       # Instead of full backtracking on 3-colorability every time,
       # we can use the fact that adding a vertex connected to an independent set
       # might preserve 4-chromaticity.
       
       # Let's do a beam search or limited depth-first search.
       # Given 16 vertices, the number of independent sets is manageable.
       
       best_edges = 20 # Grötzsch
       best_adj = adj[:]
       
       # Current states: list of (adj_list, num_vertices)
       # We start with 1 state.
       states = [(adj, 11)]
       
       for target_v in range(11, 16):
           next_states = []
           # Generate independent sets for each state?
           # This might be too slow if we do it for all states.
           # Let's prioritize states with more edges.
           
           # Sort states by current edge count descending
           def count_edges(adj_list, n):
               return sum(bin(adj_list[i]).count('1') for i in range(n)) // 2
               
           states.sort(key=lambda s: count_edges(s[0], s[1]), reverse=True)
           
           # Limit number of states to keep (Beam Width)
           # If we keep too many, it explodes.
           # Let's try keeping top 1000 states?
           # Or just top 10?
           # Let's try top 50.
           
           current_states_to_process = states[:50]
           
           for adj_list, n in current_states_to_process:
               # Find all independent sets in adj_list
               # An independent set is a set of vertices with no edges between them.
               # We can find them by checking subsets.
               # n is at most 15. 2^15 = 32768.
               # For each subset, check if it is independent.
               
               # Optimization: 
               # We only care about independent sets that result in a 4-chromatic graph.
               
               # Iterate through all subsets of 0..n-1
               # We can do this efficiently.
               # But 32768 * 50 states * 5 steps = 8 million checks.
               # Each check involves 3-colorability (expensive).
               
               # We need to speed up 3-colorability or prune subsets.
               
               # Pruning: If a subset S is not independent, skip.
               # If S is independent, add vertex n connected to S.
               # Check 3-colorability.
               
               # To speed up 3-colorability, we can use a faster algorithm or caching.
               # Or, we can observe that if the graph is 4-chromatic, 
               # it contains a Grötzsch subgraph.
               
               # Let's just run the backtracking.
               # With n=16, backtracking is usually very fast unless it's near the threshold.
               
               # Generate independent sets
               # We can iterate masks from 0 to 2^n - 1
               # But we want large ones first.
               
               # Let's collect valid independent sets
               # And sort them by size descending.
               
               # To avoid generating all 2^n, we can use a recursive generator for independent sets.
               
               def gen_independent_sets(adj, n):
                   # Returns list of masks
                   # Simple method: iterate all masks
                   # For n=15, 32768 is small.
                   res = []
                   for mask in range(1 << n):
                       # Check if mask is independent
                       # A mask is independent if for every edge (u,v), not both u and v are in mask.
                       # Equivalently, (adj[u] & mask) == 0 for all u in mask? No.
                       # (adj[u] & mask) == 0 means u has no neighbors in mask.
                       # So for all u in mask, (adj[u] & mask) == 0 is sufficient?
                       # Yes, if u is in mask, it cannot have any neighbor in mask.
                       independent = True
                       m = mask
                       while m:
                           lsb = m & -m
                           u = lsb.bit_length() - 1
                           if adj[u] & mask:
                               independent = False
                               break
                           m -= lsb
                       if independent:
                           res.append(mask)
                   return res
               
               # This is O(2^n * n). For n=15, 32768 * 15 = 500k ops.
               # For 50 states, 25M ops.
               # Plus 3-colorability checks.
               
               # This might be too slow in Python if done naively.
               
               # Alternative:
               # We know the Grötzsch graph is 4-chromatic.
               # Adding a vertex connected to an independent set I.
               # If the graph remains 4-chromatic, we keep it.
               
               # Let's limit the independent sets to those of size >= some threshold?
               # No, we need to maximize edges.
               
               # Let's try to optimize the 3-colorability check.
               # Use a simple SAT solver logic?
               
               # Given the time constraint, I will implement a search that keeps the top K states
               # and uses a fast 3-colorability check.
               
               # Fast 3-colorability:
               # Backtracking with MRV (Minimum Remaining Values) and Forward Checking.
               
               # Let's try a smaller beam width, e.g., 10.
               # And limit the number of independent sets checked per state.
               
               # Actually, for n=11 (Grötzsch), number of independent sets is small.
               # As n grows, it grows.
               
               # Let's just do it and see if it fits in time.
               # If not, the fallback is the Grötzsch graph (20 edges).
               
               # To be safe, I will wrap this in a try-except or time check.
               
               # Refinement:
               # We can pre-calculate independent sets for the Grötzsch graph.
               # Then for the next step, we recalculate.
               
               # Let's use a global time check.
               if time.time() > deadline:
                   break
                   
               indep_sets = gen_independent_sets(adj_list, n)
               # Sort by size descending
               indep_sets.sort(key=lambda x: bin(x).count('1'), reverse=True)
               
               # Only consider top X independent sets?
               # If we take the largest, we maximize edges immediately.
               # But it might lead to a dead end (3-colorable).
               # So we should try several.
               
               # Let's try top 10 independent sets.
               for mask in indep_sets[:10]:
                   # Check time
                   if time.time() > deadline:
                       break
                       
                   # Create new adj_list
                   new_adj = adj_list[:] + [0]
                   # Connect new vertex n to mask
                   new_mask = mask
                   new_adj[n] = mask
                   # Update neighbors
                   m = mask
                   while m:
                       lsb = m & -m
                       u = lsb.bit_length() - 1
                       new_adj[u] |= (1 << n)
                       m -= lsb
                       
                   # Check 3-colorability
                   if not is_3_colorable(new_adj, n + 1):
                       # It is 4-chromatic (or higher, which is fine)
                       # Add to next states
                       next_states.append((new_adj, n + 1))
                       
               # Add all current states to next states as well? 
               # No, we must add the vertex.
               # But what if we skip adding edges? 
               # We want to maximize edges, so we should add at least one edge?
               # Actually, we can add 0 edges (isolated vertex).
               # Mask 0 is an independent set.
               # So it is included in indep_sets.
               
           # Deduplicate next_states?
           # Many states might be identical.
           # We can use a set of tuples?
           # Or just keep the top K by edge count.
           
           if not next_states:
               # If we found no valid extensions, we are stuck?
               # This shouldn't happen if we include mask 0 (isolated vertex),
               # which preserves 4-chromaticity (adding isolated vertex doesn't change chromatic number).
               # So mask 0 should always be valid.
               pass
               
           states = next_states
           # Sort states by edge count
           states.sort(key=lambda s: count_edges(s[0], s[1]), reverse=True)
           # Keep top 20 states to prevent explosion
           states = states[:20]
           
       # After loop, we have states for 16 vertices
       # Pick the one with max edges
       if states:
           best_state = max(states, key=lambda s: count_edges(s[0], s[1]))
           best_adj = best_state[0]
       
       # Convert best_adj to edge list
       edges = []
       n = 16
       for i in range(n):
           for j in range(i + 1, n):
               if (best_adj[i] >> j) & 1:
                   edges.append([i, j])
                   
       return edges

   # Wait, the above code has a bug in gen_independent_sets:
   # It checks `if adj[u] & mask`. 
   # If u is in mask, and adj[u] has any bit set in mask, it's not independent.
   # This is correct.
   
   # However, `gen_independent_sets` iterates ALL 2^n masks.
   # For n=15, this is 32768.
   # For each mask, it iterates bits.
   # This is fast.
   
   # The bottleneck is `is_3_colorable`.
   # Backtracking on 16 vertices is usually very fast.
   
   # One issue: `states` might contain duplicates.
   # We can use a set of strings? No, too slow.
   # We can just keep the top K.
   
   # Another issue: The Grötzsch graph construction in the code above:
   # `adj` is a list of 11 ints.
   # The logic for Grötzsch seems correct.
   
   # Let's double check the Grötzsch construction.
   # Vertices 0-4: C5.
   # Vertices 5-9: Copies.
   # Vertex 10: Root.
   
   # Edges:
   # C5: (0,1), (1,2), (2,3), (3,4), (4,0)
   # Copies:
   # w_i (5+i) connected to v_{i+1} and v_{i-1}.
   # w_0 (5) to v_1 (1) and v_4 (4).
   # w_1 (6) to v_2 (2) and v_0 (0).
   # ...
   # Root 10 connected to w_0..w_4.
   
   # This is the standard Mycielski of C5.
   
   # The search should find a graph with > 20 edges.
   
   # To ensure we don't timeout, I will add a hard stop.
   
   # One optimization: 
   # If we find a graph with 34 edges, we can stop early?
   # We don't know the max.
   # But if we find 34, it's likely optimal or near optimal.
   
   # Let's set a target. If we find 30+ edges, we can relax the beam width?
   # No, keep it simple.
   
   # Final check on the return format:
   # list of edges [u, v]
   
   # The code looks solid.
   
   # One potential issue: `is_3_colorable` might be slow for some instances.
   # I will add a simple cache or a faster check?
   # For 16 vertices, it's fine.
   
   # Let's write the complete program.
