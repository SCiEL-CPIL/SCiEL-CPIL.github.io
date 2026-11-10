"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys
import math

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute edge indices
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            idx_map[(i,j)] = idx
            idx += 1
            
    # Precompute K4 structure
    k4_list = list(itertools.combinations(range(n), 4))
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    
    for k4_idx, v in enumerate(k4_list):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i,j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(k4_idx)
        k4_edges.append(e_indices)
        
    def count_violations(colors):
        """Count number of K4s with < 5 distinct colours."""
        violations = 0
        for e_indices in k4_edges:
            col_set = set()
            for ei in e_indices:
                col_set.add(colors[ei])
            if len(col_set) < 5:
                violations += 1
        return violations
        
    def num_colours_used(colors):
        return len(set(colors))
        
    # Base solution: (i+j) mod 17
    base_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Verify base
    if count_violations(base_colors) != 0:
        raise ValueError("Base solution invalid")
        
    # State
    best_colors = base_colors.copy()
    best_num = num_colours_used(base_colors) # 17
    
    # Local Search / SA Function
    def local_search(colors, target_colors=None, max_time=5.0, mode="repair"):
        """
        mode='repair': Minimize violations.
        mode='optimize': Minimize violations * 1000 + num_colors.
        """
        cur = colors.copy()
        cur_viol = count_violations(cur)
        cur_num = num_colours_used(cur)
        
        best_local = cur.copy()
        best_local_viol = cur_viol
        best_local_num = cur_num
        
        temp = 1.0
        cooling = 0.99
        time_limit = time.time() + max_time
        
        # Precompute affected K4s for each edge for fast updates? 
        # For simplicity in this short run, we recalculate violations for affected K4s only.
        
        while time.time() < time_limit:
            # Pick a random edge
            ei = random.randint(0, num_edges - 1)
            old_col = cur[ei]
            
            # Pick a new color
            if mode == "repair":
                # Try colors from 0..16 or current palette
                if random.random() < 0.5:
                    new_col = random.randint(0, 16)
                else:
                    new_col = random.choice(list(set(cur)))
            else:
                # Try to merge: pick a color that is NOT old_col, preferably one that is frequent
                # Or just random
                if random.random() < 0.7:
                    # Pick from existing colors
                    existing = list(set(cur))
                    if old_col in existing:
                        existing.remove(old_col)
                    if existing:
                        new_col = random.choice(existing)
                    else:
                        new_col = old_col
                else:
                    new_col = random.randint(0, 16)
                    
            if new_col == old_col:
                continue
                
            # Apply change
            cur[ei] = new_col
            
            # Calculate delta violations
            # Only K4s containing ei are affected
            delta_viol = 0
            for k4i in edge_to_k4[ei]:
                e_indices = k4_edges[k4i]
                # Check old state
                old_set = set()
                for ej in e_indices:
                    if ej == ei:
                        old_set.add(old_col)
                    else:
                        old_set.add(cur[ej])
                old_viol = 1 if len(old_set) < 5 else 0
                
                # Check new state
                new_set = set()
                for ej in e_indices:
                    new_set.add(cur[ej])
                new_viol = 1 if len(new_set) < 5 else 0
                
                delta_viol += new_viol - old_viol
                
            new_viol = cur_viol + delta_viol
            new_num = num_colours_used(cur)
            
            # Cost function
            if mode == "repair":
                old_cost = cur_viol
                new_cost = new_viol
            else:
                old_cost = cur_viol * 1000 + cur_num
                new_cost = new_viol * 1000 + new_num
                
            delta_cost = new_cost - old_cost
            
            if delta_cost <= 0:
                # Accept
                cur_viol = new_viol
                cur_num = new_num
                if (mode == "repair" and cur_viol < best_local_viol) or \
                   (mode == "optimize" and (cur_viol * 1000 + cur_num) < (best_local_viol * 1000 + best_local_num)):
                    best_local = cur.copy()
                    best_local_viol = cur_viol
                    best_local_num = cur_num
            else:
                # Metropolis
                if random.random() < math.exp(-delta_cost / temp):
                    cur_viol = new_viol
                    cur_num = new_num
                else:
                    # Revert
                    cur[ei] = old_col
                    
            temp *= cooling
            
        return best_local, best_local_viol, best_local_num

    # Strategy: Merge colors from base 17 down to 15
    # We will try to merge pairs of colors.
    # To be efficient, we will try to merge colors that are "similar" or just random pairs.
    
    current_colors = base_colors.copy()
    current_viol = 0
    current_num = 17
    
    # We want to reach 15.
    # We will attempt to reduce the number of colors by 1, then by another 1.
    
    for target_num in [16, 15]:
        if time.time() > deadline - 20:
            break
            
        success = False
        # Try multiple random merge attempts
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
                
            # Pick two colors to merge
            # Prefer colors that are less frequent? Or just random.
            color_counts = {}
            for c in current_colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            used_colors = list(color_counts.keys())
            
            if len(used_colors) < 2:
                break
                
            # Pick two distinct colors
            c1, c2 = random.sample(used_colors, 2)
            
            # Merge c2 into c1
            merged_colors = current_colors.copy()
            for i in range(num_edges):
                if merged_colors[i] == c2:
                    merged_colors[i] = c1
                    
            merged_viol = count_violations(merged_colors)
            
            if merged_viol == 0:
                # Perfect merge
                current_colors = merged_colors
                current_viol = 0
                current_num = num_colours_used(current_colors)
                success = True
                break
            else:
                # Try to repair
                # Run SA to fix violations
                repaired, rep_viol, rep_num = local_search(merged_colors, max_time=2.0, mode="repair")
                if rep_viol == 0:
                    current_colors = repaired
                    current_viol = 0
                    current_num = num_colours_used(current_colors)
                    success = True
                    break
                    
        if not success:
            # If we failed to reduce to target_num, try a different global strategy?
            # Or just break and return best so far.
            pass

    # Now we have a valid coloring with ~15-17 colors.
    # Try to optimize further (15 -> 14 -> 13)
    if time.time() < deadline - 10:
        opt_colors, opt_viol, opt_num = local_search(current_colors, max_time=deadline - 15, mode="optimize")
        if opt_viol == 0 and opt_num < current_num:
            current_colors = opt_colors
            current_num = opt_num

    # Final check
    if count_violations(current_colors) != 0:
        # Fallback to base
        current_colors = base_colors
        current_num = 17

    # Renormalize
    col_map = {}
    new_colors = []
    next_col = 0
    for c in current_colors:
        if c not in col_map:
            col_map[c] = next_col
            next_col += 1
        new_colors.append(col_map[c])
        
    return new_colors

# EVOLVE-BLOCK-END
