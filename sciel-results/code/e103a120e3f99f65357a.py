import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def _solve_n(n):
        if time.time() > deadline:
            return None
            
        # adj[c][i] is a bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(4)]
        # color[i][j] stores the color of edge (i, j)
        color = [[-1] * n for _ in range(n)]
        
        # Initialize random coloring
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                color[i][j] = c
                color[j][i] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
        # Helper to get neighbors of i in color c as a list
        def get_neighbors(i, c):
            mask = adj[c][i]
            neighbors = []
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                neighbors.append(j)
                mask ^= lsb
            return neighbors

        # Count C6s containing edge (u, v) in color c, assuming (u, v) is NOT in adj[c] yet
        # Or if it is, we calculate the contribution of paths that would close with (u,v)
        # We want to count the number of paths v-w1-w2-w3-w4-u in G_c \ {(u,v)}
        def count_c6_contribution(u, v, c):
            # We need to count paths of length 4 from v to u in G_c, 
            # such that internal vertices are distinct and not u,v.
            # Path: v - w1 - w2 - w3 - w4 - u
            # Edges: (v,w1), (w1,w2), (w2,w3), (w3,w4), (w4,u)
            
            count = 0
            # w1 must be neighbor of v in color c, w1 != u
            # w4 must be neighbor of u in color c, w4 != v
            
            # Optimize by iterating smaller set?
            # Neighbors are sparse.
            mask_v = adj[c][v] & ~(1 << u)
            mask_u = adj[c][u] & ~(1 << v)
            
            w1_list = []
            m = mask_v
            while m:
                lsb = m & -m
                w1 = lsb.bit_length() - 1
                w1_list.append(w1)
                m ^= lsb
                
            w4_list = []
            m = mask_u
            while m:
                lsb = m & -m
                w4 = lsb.bit_length() - 1
                w4_list.append(w4)
                m ^= lsb
                
            for w1 in w1_list:
                for w4 in w4_list:
                    if w1 == w4:
                        continue
                    # Need path w1 - w2 - w3 - w4 of length 2 edges (3 vertices w1,w2,w3,w4? No, 2 edges is w1-w2-w3? No.)
                    # Path length 4 edges: v-w1 (1), w1-w2 (2), w2-w3 (3), w3-w4 (4), w4-u (5).
                    # So we need a path of length 2 edges between w1 and w4: w1-w2-w3-w4?
                    # No, w1 to w4 has 3 edges in the path w1-w2-w3-w4.
                    # Wait.
                    # Total edges: (v,w1), (w1,w2), (w2,w3), (w3,w4), (w4,u).
                    # Subpath from w1 to w4 has 3 edges: w1-w2, w2-w3, w3-w4.
                    # So we need to count paths of length 3 from w1 to w4 in G_c \ {u,v}.
                    
                    # This is getting expensive.
                    # Let's stick to a simpler heuristic for SA, and use exact check at the end.
                    # Heuristic: Count monochromatic C4s or K3,3.
                    # Or just use the exact count but only for the full graph occasionally.
                    pass
            return 0 # Placeholder, we will use a different cost function

        # Let's use a simpler cost function for SA: Number of monochromatic C4s.
        # If no monochromatic C4, then no C6? No.
        # But minimizing C4s is a good proxy.
        # Actually, let's just use the number of "bad" edges or something simple.
        
        # Better: Use the exact count but precompute all C6s for n<=17?
        # For n=17, 742k C6s.
        # We can represent each C6 as a tuple of 6 edges.
        # And maintain a counter for each C6? No, too much memory/time.
        
        # Let's go with the incremental exact count for C6s.
        # The path length 3 from w1 to w4 can be counted by:
        # Sum over w2 in N(w1) \ {u,v,w4} of |N(w2) & N(w4) \ {u,v,w1}|
        
        def count_c6_exact(u, v, c):
            count = 0
            mask_v = adj[c][v] & ~(1 << u)
            mask_u = adj[c][u] & ~(1 << v)
            
            w1_list = []
            m = mask_v
            while m:
                lsb = m & -m
                w1 = lsb.bit_length() - 1
                w1_list.append(w1)
                m ^= lsb
                
            w4_list = []
            m = mask_u
            while m:
                lsb = m & -m
                w4 = lsb.bit_length() - 1
                w4_list.append(w4)
                m ^= lsb
                
            for w1 in w1_list:
                for w4 in w4_list:
                    if w1 == w4:
                        continue
                    # Count paths w1-w2-w3-w4 of length 3 edges
                    # w2 in N(w1) \ {u,v,w4}
                    # w3 in N(w4) \ {u,v,w1}
                    # Edge (w2, w3) in G_c
                    # w2 != w3
                    
                    mask_w1 = adj[c][w1] & ~(1 << u) & ~(1 << v) & ~(1 << w4)
                    mask_w4 = adj[c][w4] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
                    
                    # For each w2 in mask_w1, check if any w3 in mask_w4 is connected to w2
                    m2 = mask_w1
                    while m2:
                        lsb2 = m2 & -m2
                        w2 = lsb2.bit_length() - 1
                        # Check intersection of N(w2) and mask_w4
                        # N(w2) in color c
                        common = adj[c][w2] & mask_w4
                        # Ensure w2 != w3 (handled by mask_w4 excluding w1, but w2 could be in mask_w4? No, mask_w4 excludes w1, but w2 is not w1.
                        # However, w2 could be w4? No, mask_w1 excludes w4.
                        # So w2 and w3 are distinct from u,v,w1,w4.
                        # Are w2 and w3 distinct from each other?
                        # If w2 is in mask_w4, then w2 is a neighbor of w4.
                        # If w2 is in N(w2)? No, no self loops.
                        # So we just count bits in common.
                        count += common.bit_count()
                        m2 ^= lsb2
                        
            return count

        # Initial total conflicts
        total_conflicts = 0
        # This is O(E * N^2) which is too slow for initialization.
        # We can estimate or just start SA with a random state and let it find the cost?
        # No, SA needs the cost.
        
        # Alternative: Start with a known valid coloring for n=16 or 17?
        # The problem states best known is 17.
        # We can try to construct one from scratch.
        
        # To speed up initialization, we can use a heuristic initialization.
        # Or, we can just run SA and calculate the full cost once at the beginning using a faster method?
        # For n=17, 742k C6s.
        # We can iterate all 6-subsets and check all 60 cycles.
        # 12376 * 60 = 742k checks.
        # Each check is 6 edge lookups. ~4.5M operations.
        # In Python, this might take ~1-2 seconds. Acceptable.
        
        def calculate_total_conflicts():
            total = 0
            # Precompute all 6-subsets
            vertices = range(n)
            for subset in combinations(vertices, 6):
                # Check all 60 cycles
                # We can just check if any cycle is monochromatic
                # To speed up, we can check colors of edges
                colors_in_subset = {}
                for i in range(6):
                    for j in range(i+1, 6):
                        u = subset[i]
                        v = subset[j]
                        c = color[u][v]
                        colors_in_subset[(i,j)] = c
                        
                # 60 cycles. We can hardcode or generate.
                # Generating 60 permutations of 1..5 is fast.
                # But doing it 12376 times is slow.
                # Precompute the list of edge index pairs for each of the 60 cycles.
                pass
            return total

        # This is getting too complex for the time limit.
        # Let's use a simpler heuristic: Minimize the number of monochromatic C4s.
        # A C4 is a 4-cycle.
        # Counting C4s is faster.
        # If we get 0 C4s, we are close.
        
        # Let's try a different approach: 
        # Use the known valid coloring for n=17 if we can hardcode it? 
        # No, we must generate it.
        
        # I will implement the SA with the exact C6 count using the incremental method, 
        # but I will initialize the total conflict count by summing count_c6_exact for all edges.
        # This is O(E * N^2) = (17*16/2) * 17^2 ~ 136 * 289 ~ 39k operations. Very fast!
        
        total_conflicts = 0
        for i in range(n):
            for j in range(i+1, n):
                c = color[i][j]
                # Temporarily remove edge to count contribution?
                # No, count_c6_exact(u, v, c) counts paths that *would* form a C6 if (u,v) was added.
                # Since (u,v) IS in color c, the number of C6s containing (u,v) is exactly 
                # the number of paths of length 4 from v to u in G_c \ {(u,v)}.
                # So we can just sum count_c6_exact(i, j, c) for all edges.
                # But count_c6_exact uses adj[c] which INCLUDES (i,j).
                # We must exclude (i,j) from the neighbor lists.
                # My count_c6_exact already does `& ~(1 << u)` and `& ~(1 << v)`.
                # So it correctly ignores the edge (u,v) itself in the path search.
                # So total_conflicts = sum(count_c6_exact(i, j, c) for all edges (i,j)) / 2?
                # Each C6 has 6 edges. Each edge is counted once for that C6.
                # So summing over all edges gives the total number of C6s.
                
                total_conflicts += count_c6_exact(i, j, c)
                
        # SA Loop
        T = 1.0
        best_conflicts = total_conflicts
        best_color = [row[:] for row in color]
        
        steps = 0
        max_steps = 200000
        
        while steps < max_steps and time.time() < deadline - 5:
            # Pick random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_c = color[i][j]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            # Calculate delta
            # Remove (i,j) from old_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            
            # Count C6s lost (those that contained (i,j) and were monochromatic in old_c)
            # This is count_c6_exact(i, j, old_c) calculated on the graph WITHOUT (i,j).
            # But count_c6_exact excludes (i,j) from neighbor lists.
            # So we just call it.
            lost = count_c6_exact(i, j, old_c)
            
            # Add (i,j) to new_c
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
            
            # Count C6s gained
            gained = count_c6_exact(i, j, new_c)
            
            delta = gained - lost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                color[i][j] = new_c
                color[j][i] = new_c
                total_conflicts += delta
                if total_conflicts < best_conflicts:
                    best_conflicts = total_conflicts
                    best_color = [row[:] for row in color]
                    if total_conflicts == 0:
                        # Found valid coloring
                        return best_color
            else:
                # Reject move
                adj[new_c][i] &= ~(1 << j)
                adj[new_c][j] &= ~(1 << i)
                adj[old_c][i] |= (1 << j)
                adj[old_c][j] |= (1 << i)
                
            T *= 0.999
            if T < 0.01:
                T = 0.01
            steps += 1
            
        # Return best found
        if best_conflicts == 0:
            return best_color
        return None

    # Try to solve for n=17, then 16, etc.
    for n_target in range(17, 1, -1):
        if time.time() > deadline - 10:
            break
        colors = _solve_n(n_target)
        if colors is not None:
            # Flatten colors
            flat_colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    flat_colors.append(colors[i][j])
            return {'n': n_target, 'colors': flat_colors}
            
    # Fallback: Return a small valid coloring
    # n=1 is always valid
    return {'n': 1, 'colors': []}

# EVOLVE-BLOCK-END
