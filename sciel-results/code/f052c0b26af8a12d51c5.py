import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    # Generate all edges in lexicographic order: (i, j) for 0 <= i < j < 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s (sets of 4 vertices) and their edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        # Map vertex pair to edge index
        # Edge (i,j) has index: sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Precompute a lookup dictionary for speed
        edge_idx_map = {e: idx for idx, e in enumerate(edges)}
        
        e_indices = [
            edge_idx_map[(v0, v1)],
            edge_idx_map[(v0, v2)],
            edge_idx_map[(v0, v3)],
            edge_idx_map[(v1, v2)],
            edge_idx_map[(v1, v3)],
            edge_idx_map[(v2, v3)],
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # 1820
    
    # Precompute which K4s contain each edge for fast delta updates
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Convert k4s to numpy array for fast indexing
    k4s_np = np.array(k4s, dtype=np.int32)
    
    def count_violations(col):
        """
        col: numpy array of shape (120,)
        Returns number of K4s with < 5 distinct colors.
        """
        violations = 0
        for i in range(num_k4s):
            # Get the 6 colors for this K4
            # col[k4s_np[i]] is a numpy array of 6 ints
            # Convert to set to count distinct
            if len(set(col[k4s_np[i]].tolist())) < 5:
                violations += 1
        return violations

    def get_initialization(method, target_colors):
        """
        Generate an initial coloring.
        method: 'random' or 'algebraic'
        """
        if method == 'random':
            return np.random.randint(0, target_colors, size=num_edges)
        elif method == 'algebraic':
            # Construct coloring based on GF(4)^2 differences
            # Vertices are elements of GF(4)^2.
            # We map vertices 0..15 to vectors in GF(4)^2.
            # GF(4) = {0, 1, w, w^2} where w^2 + w + 1 = 0.
            # We can represent GF(4) elements as 2-bit integers with custom addition/multiplication.
            # But for simplicity, let's just use the standard difference in Z_16? No, that's not GF(4).
            # Let's stick to a simpler algebraic-like initialization that is structurally sound but valid.
            # Actually, the prompt's algebraic idea failed for K4s in affine planes.
            # Let's use a "structured random" initialization:
            # Assign colors based on (i + j) % target_colors, then shuffle.
            base_col = [(i + j) % target_colors for i in range(n) for j in range(i + 1, n)]
            return np.array(base_col, dtype=np.int32)
        else:
            return np.random.randint(0, target_colors, size=num_edges)

    best_result = None
    best_color_count = float('inf')
    
    # Fallback: The seed solution (i+j)%17 is valid and uses 17 colors.
    # We want to beat this.
    seed_col = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    if count_violations(seed_col) == 0:
        best_result = seed_col.tolist()
        best_color_count = 17

    # Try to find solutions with fewer colors: 13, 14, 15
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 15.0:
            break
            
        # Try different initialization strategies
        init_strategies = ['random', 'algebraic']
        
        for init_method in init_strategies:
            if time.time() > deadline - 10.0:
                break
                
            # Multiple restarts for this target and init
            max_restarts = 20 if target_colors == 13 else 10
            for restart in range(max_restarts):
                if time.time() > deadline - 5.0:
                    break
                    
                col = get_initialization(init_method, target_colors)
                current_violations = count_violations(col)
                
                if current_violations == 0:
                    # Found a valid solution
                    colors_used = len(set(col.tolist()))
                    if colors_used < best_color_count:
                        best_color_count = colors_used
                        best_result = col.tolist()
                    break # Break restarts for this init method if found? No, might find better color count?
                    # Actually, if we target 13, we use at most 13. If valid, it's great.
                    # If we target 14, we use at most 14.
                    # So finding one with 13 is best.
                    continue
                
                # Simulated Annealing / Local Search
                # We want to minimize violations.
                T = 1.0
                T_min = 0.01
                cooling = 0.9995
                max_iters = 50000
                
                for it in range(max_iters):
                    if time.time() > deadline - 2.0:
                        break
                        
                    # Pick a random edge
                    e_idx = random.randint(0, num_edges - 1)
                    old_color = col[e_idx]
                    
                    # Pick a new random color
                    new_color = random.randint(0, target_colors - 1)
                    if new_color == old_color:
                        continue
                    
                    # Calculate delta violations
                    # Only K4s containing e_idx are affected
                    affected_k4s = edge_to_k4s[e_idx]
                    delta_viol = 0
                    
                    # Temporarily apply change
                    col[e_idx] = new_color
                    
                    for k4_idx in affected_k4s:
                        # Get colors for this K4
                        k4_colors = col[k4s_np[k4_idx]]
                        distinct_count = len(set(k4_colors.tolist()))
                        
                        # We need to know if this K4 was valid before and after.
                        # Since we don't store the "before" state for each K4 in a cache easily without complex bookkeeping,
                        # and num_affected is small (~90), we can just re-evaluate the "before" state by reverting?
                        # No, that's expensive.
                        # Better: Store current distinct counts in a cache array.
                        
                    # Let's refactor to use a cache for distinct counts to make delta calculation O(1) per affected K4.
                    # This requires maintaining `k4_distinct` array.
                    
                    # Revert for now if we don't have cache, but let's implement cache properly.
                    pass # Placeholder, will rewrite loop with cache

                # (Re-implementing the search loop with proper caching below)
                
                # --- Proper Search Loop with Caching ---
                # Initialize cache
                k4_distinct = np.zeros(num_k4s, dtype=np.int32)
                for i in range(num_k4s):
                    k4_colors = col[k4s_np[i]]
                    k4_distinct[i] = len(set(k4_colors.tolist()))
                
                current_violations = np.sum(k4_distinct < 5)
                
                T = 1.0
                cooling = 0.9995
                T_min = 0.01
                max_iters = 20000
                
                for it in range(max_iters):
                    if time.time() > deadline - 2.0:
                        break
                        
                    e_idx = random.randint(0, num_edges - 1)
                    old_color = col[e_idx]
                    new_color = random.randint(0, target_colors - 1)
                    if new_color == old_color:
                        continue
                    
                    # Apply change
                    col[e_idx] = new_color
                    
                    delta_viol = 0
                    affected_k4s = edge_to_k4s[e_idx]
                    
                    for k4_idx in affected_k4s:
                        old_distinct = k4_distinct[k4_idx]
                        # Recalculate new distinct count
                        k4_colors = col[k4s_np[k4_idx]]
                        new_distinct = len(set(k4_colors.tolist()))
                        
                        # Update cache
                        k4_distinct[k4_idx] = new_distinct
                        
                        # Calculate delta in violation status
                        old_is_violation = (old_distinct < 5)
                        new_is_violation = (new_distinct < 5)
                        
                        if old_is_violation and not new_is_violation:
                            delta_viol -= 1
                        elif not old_is_violation and new_is_violation:
                            delta_viol += 1
                    
                    # Acceptance criterion
                    if delta_viol <= 0:
                        current_violations += delta_viol
                        # Accept
                    else:
                        # Accept with probability exp(-delta / T)
                        if random.random() < np.exp(-delta_viol / T):
                            current_violations += delta_viol
                            # Accept
                        else:
                            # Reject: Revert
                            col[e_idx] = old_color
                            # Revert cache for affected K4s
                            for k4_idx in affected_k4s:
                                k4_colors = col[k4s_np[k4_idx]]
                                k4_distinct[k4_idx] = len(set(k4_colors.tolist()))
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    
                    if current_violations == 0:
                        break
                
                if current_violations == 0:
                    colors_used = len(set(col.tolist()))
                    if colors_used < best_color_count:
                        best_color_count = colors_used
                        best_result = col.tolist()
                    # If we found a valid solution with target_colors, 
                    # we can break out of restarts for this target_colors 
                    # because we can't do better than target_colors with this palette.
                    # However, if target_colors is 14, we might find a solution using only 13 colors?
                    # No, we are restricting colors to 0..target_colors-1.
                    # So if we find a valid solution with target_colors=14, it uses <= 14 colors.
                    # If we find one with 13, it's better.
                    # So we should continue searching for 13 if we haven't found it yet.
                    # But if we are in the loop for target_colors=13, and we found it, we can stop.
                    if target_colors == 13:
                        break # Found optimal (lower bound)

    if best_result is not None:
        return best_result
    
    # Fallback
    return seed_col.tolist()
