import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute K4s
    k4s = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        # Edge indices for (v0,v1), (v0,v2), (v0,v3), (v1,v2), (v1,v3), (v2,v3)
        # We need a fast way to get edge index from (u,v)
        # (u,v) index: 
        # Number of edges starting at < u: sum_{k=0}^{u-1} (n-1-k)
        # Edges starting at u: (u, u+1) ... (u, v) is the (v-u-1)-th one
        # Let's precompute a dictionary for speed, it's small (120 entries)
        edge_idx_map = {e: i for i, e in enumerate(edges)}
        
        e_indices = [
            edge_idx_map[(v0, v1)],
            edge_idx_map[(v0, v2)],
            edge_idx_map[(v0, v3)],
            edge_idx_map[(v1, v2)],
            edge_idx_map[(v1, v3)],
            edge_idx_map[(v2, v3)]
        ]
        k4s.append(e_indices)
    
    k4s = np.array(k4s, dtype=np.int32)
    num_k4s = len(k4s)
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Convert to numpy arrays for faster iteration if needed, but lists are fine for 91 items
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]

    def count_violations(col):
        """
        col: numpy array of shape (120,)
        Returns number of K4s with < 5 distinct colors.
        """
        # col[k4s] gives shape (1820, 6)
        k4_colors = col[k4s]
        # To count distinct elements in each row efficiently:
        # Sort and count changes? Or use a set per row (slow in python loop).
        # Since 1820 is small, a vectorized approach might be tricky without loops.
        # Let's try a vectorized trick:
        # For each row, check if any color is missing?
        # Better: For each K4, the number of distinct colors is 6 - (number of pairs of edges in the K4 that share a color).
        # This is not quite right if 3 edges share a color.
        # 
        # Simplest vectorized distinct count for small rows:
        # It's hard to do in pure numpy without loops.
        # Let's use a python loop over the 1820 K4s, but it's fast enough?
        # 1820 iterations is very fast.
        
        # Alternative: 
        # distinct_count = np.apply_along_axis(lambda x: len(set(x)), 1, k4_colors)
        # This might be slow.
        
        # Let's stick to a slightly optimized python loop or just accept the 1820 loop.
        # 1820 * (cost of set of 6 ints) is negligible.
        
        violations = 0
        for i in range(num_k4s):
            # k4_colors[i] is a numpy array of 6 ints
            # Converting to set is fast
            if len(set(k4_colors[i].tolist())) < 5:
                violations += 1
        return violations

    def is_valid(col):
        return count_violations(col) == 0

    # Local Search Helper
    def try_move(col, e_idx, new_color, k4_colors_cache, k4_distinct_cache):
        """
        Tries to change col[e_idx] to new_color.
        Updates k4_colors_cache and k4_distinct_cache for affected K4s.
        Returns (delta_violations, affected_k4s, old_color)
        """
        old_color = col[e_idx]
        if old_color == new_color:
            return 0, [], old_color
        
        col[e_idx] = new_color
        
        affected_k4s = edge_to_k4s_np[e_idx]
        delta_viol = 0
        
        for k4_idx in affected_k4s:
            # Get the 6 edge indices for this K4
            k4_edges = k4s[k4_idx]
            
            # Update the cache for this K4
            # We need to recalculate the distinct count for this K4
            # Old distinct count is in k4_distinct_cache[k4_idx]
            old_distinct = k4_distinct_cache[k4_idx]
            
            # Get new colors
            new_colors = col[k4_edges]
            new_distinct = len(set(new_colors.tolist()))
            
            # Update cache
            k4_colors_cache[k4_idx] = new_colors
            k4_distinct_cache[k4_idx] = new_distinct
            
            old_is_valid = old_distinct >= 5
            new_is_valid = new_distinct >= 5
            
            if old_is_valid and not new_is_valid:
                delta_viol += 1
            elif not old_is_valid and new_is_valid:
                delta_viol -= 1
        
        return delta_viol, affected_k4s, old_color

    def initialize_caches(col):
        k4_colors_cache = np.zeros((num_k4s, 6), dtype=np.int32)
        k4_distinct_cache = np.zeros(num_k4s, dtype=np.int32)
        
        for i in range(num_k4s):
            k4_edges = k4s[i]
            colors = col[k4_edges]
            k4_colors_cache[i] = colors
            k4_distinct_cache[i] = len(set(colors.tolist()))
            
        return k4_colors_cache, k4_distinct_cache

    # Main Search Loop
    best_col = None
    best_colors_used = 100
    
    # Try to find a valid solution with C colors for C in [13, 14, 15]
    # We start with 15 because it's easier to find a valid solution.
    # Then we try to reduce.
    
    # Strategy:
    # 1. Find a valid solution with 15 colors (or fewer if lucky).
    # 2. Try to eliminate colors to get down to 13.
    
    # Step 1: Find a valid solution
    # Start with random 15 colors
    current_colors = 15
    col = np.random.randint(0, current_colors, size=num_edges)
    k4_colors_cache, k4_distinct_cache = initialize_caches(col)
    current_violations = np.sum(k4_distinct_cache < 5)
    
    if current_violations == 0:
        best_col = col.copy()
        best_colors_used = len(set(col.tolist()))
    
    # Simulated Annealing to reduce violations to 0
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline - 10.0 and current_violations > 0:
        e_idx = random.randint(0, num_edges - 1)
        new_color = random.randint(0, current_colors - 1)
        
        delta_viol, affected, old_color = try_move(col, e_idx, new_color, k4_colors_cache, k4_distinct_cache)
        
        if delta_viol <= 0:
            current_violations += delta_viol
            # Accept
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < np.exp(-delta_viol / T):
                current_violations += delta_viol
            else:
                # Reject: revert
                col[e_idx] = old_color
                # Revert caches
                for k4_idx in affected:
                    k4_edges = k4s[k4_idx]
                    colors = col[k4_edges]
                    k4_colors_cache[k4_idx] = colors
                    k4_distinct_cache[k4_idx] = len(set(colors.tolist()))
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    if current_violations == 0:
        # We have a valid solution
        # Try to reduce the number of colors
        colors_in_use = set(col.tolist())
        
        while len(colors_in_use) > 13 and time.time() < deadline - 5.0:
            # Pick a color to eliminate
            # Pick the one with fewest edges? Or random?
            color_counts = np.bincount(col, minlength=current_colors)
            # Find colors with count > 0
            used_colors = [c for c in range(current_colors) if color_counts[c] > 0]
            
            if not used_colors:
                break
                
            # Try to eliminate a random used color
            target_color = random.choice(used_colors)
            edges_to_change = np.where(col == target_color)[0]
            
            success = True
            for e_idx in edges_to_change:
                # Try to change this edge to a different color
                old_color = col[e_idx]
                found_new_color = False
                for new_color in range(current_colors):
                    if new_color == old_color:
                        continue
                    
                    delta_viol, affected, _ = try_move(col, e_idx, new_color, k4_colors_cache, k4_distinct_cache)
                    
                    if delta_viol <= 0:
                        # Accept
                        current_violations += delta_viol
                        found_new_color = True
                        break
                    else:
                        # Reject: revert
                        col[e_idx] = old_color
                        for k4_idx in affected:
                            k4_edges = k4s[k4_idx]
                            colors = col[k4_edges]
                            k4_colors_cache[k4_idx] = colors
                            k4_distinct_cache[k4_idx] = len(set(colors.tolist()))
                
                if not found_new_color:
                    success = False
                    break
            
            if success:
                # We eliminated target_color
                # Update colors_in_use
                colors_in_use.discard(target_color)
                # Update best
                best_col = col.copy()
                best_colors_used = len(colors_in_use)
            else:
                # Failed to eliminate this color, try another?
                # Or restart the elimination process?
                # Let's just break and try a different random restart if we have time
                pass
                
    # If we didn't find a valid solution, or didn't reduce colors, 
    # we might want to do a few more random restarts of the SA phase
    # But we already have a best_col if we found any valid one.
    
    # If best_col is None, we need a fallback.
    # The seed (i+j)%17 is valid.
    if best_col is None:
        # Generate seed
        seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        return seed_col
    
    return best_col.tolist()

# EVOLVE-BLOCK-END
