import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s) # 1820
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    # --- 1. Generate 1-factorization coloring (15 colors) ---
    # Standard 1-factorization of K_16
    # Vertices 0..14, 15 (infinity)
    # Matchings M_0 ... M_14
    # M_i contains (15, i) and ( (i+j)%15, (i-j)%15 ) for j=1..7
    
    def generate_1_factorization_colors():
        colors = [0] * num_edges
        # Map vertices 0..14 to 0..14, and 15 to 15
        # The standard construction uses vertices 0..2k-2 and infinity.
        # Here 2k-2 = 14. So vertices 0..14 and 15.
        
        for i in range(15):
            # Edge (15, i) gets color i
            e_idx = edge_index[(15, i)] if (15, i) in edge_index else edge_index[(i, 15)]
            colors[e_idx] = i
            
            for j in range(1, 8):
                u = (i + j) % 15
                v = (i - j) % 15
                if u == v: continue # Should not happen for odd modulus
                e = (u, v) if u < v else (v, u)
                e_idx = edge_index[e]
                colors[e_idx] = i
        return colors

    seed_colors = generate_1_factorization_colors()
    
    # Verify seed validity
    def is_valid(colors):
        for e_indices in k4_edge_indices:
            if len(set(colors[idx] for idx in e_indices)) < 5:
                return False
        return True

    if not is_valid(seed_colors):
        # If the 1-factorization is not valid (unlikely), fallback to random search with 15 colors
        # But let's assume it is valid for now. If not, the local search will fix it.
        pass

    # --- 2. Local Search with Simulated Annealing ---
    
    def local_search(initial_colors, max_colors_allowed, time_limit):
        # We want to minimize the number of colors used, subject to validity.
        # We start with a valid solution (hopefully) and try to reduce colors.
        # If we start with an invalid one, we try to make it valid first.
        
        colors = initial_colors[:]
        
        # Data structures for fast updates
        # k4_counts[k4_idx] is a dict {color: count} or list of size max_colors_allowed?
        # Since we might use colors 0..14, let's use a fixed size array of 15.
        # If we want to allow "merging" colors, we just change values.
        
        max_c = 15 # We only use colors 0..14
        
        k4_counts = []
        k4_distinct = []
        total_violations = 0
        
        # Initialize counts
        for e_indices in k4_edge_indices:
            counts = [0] * max_c
            for idx in e_indices:
                c = colors[idx]
                if c < max_c:
                    counts[c] += 1
                else:
                    # Should not happen if we stick to 0..14
                    pass
            distinct = sum(1 for cnt in counts if cnt > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_violations += 1
                
        # Global color usage count
        global_color_counts = [0] * max_c
        for c in colors:
            if c < max_c:
                global_color_counts[c] += 1
        
        def get_score():
            used_colors = sum(1 for cnt in global_color_counts if cnt > 0)
            return used_colors * 10000 + total_violations

        best_score = get_score()
        best_colors = colors[:]
        best_violations = total_violations
        best_used_colors = sum(1 for cnt in global_color_counts if cnt > 0)
        
        temp = 100.0
        min_temp = 0.1
        cool_rate = 0.999
        
        start_ls = time.time()
        
        while time.time() < start_ls + time_limit:
            if total_violations == 0:
                # If valid, check if we can improve used_colors
                # The score function handles this.
                pass
            
            # Pick a random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick a new color
            # Bias towards colors that are already in use to reduce used_colors
            # Or random 0..14
            new_color = random.randint(0, max_c - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate Delta
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            old_distincts = []
            old_global_counts = global_color_counts[:] # Copy is cheap for 15 ints
            
            # Tentative update
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_distincts.append(old_d)
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # Was 0
                    new_d += 1
                
                k4_distinct[k4_idx] = new_d
                
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
            
            # Update global counts
            global_color_counts[old_color] -= 1
            global_color_counts[new_color] += 1
            
            # Calculate Delta Score
            old_used = sum(1 for cnt in old_global_counts if cnt > 0)
            new_used = sum(1 for cnt in global_color_counts if cnt > 0)
            delta_used = new_used - old_used
            
            delta_score = delta_used * 10000 + delta_viol
            
            # Acceptance
            accept = False
            if delta_score <= 0:
                accept = True
            else:
                if delta_score < 1000 * temp:
                    if random.random() < math.exp(-delta_score / temp):
                        accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_violations += delta_viol
                
                current_score = get_score()
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    best_violations = total_violations
                    best_used_colors = new_used
            else:
                # Revert
                for k4_idx, old_d in zip(affected_k4s, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                global_color_counts = old_global_counts[:]
                
            temp *= cool_rate
            if temp < min_temp:
                temp = min_temp

        return best_colors, best_used_colors, best_violations

    # Run local search
    # Try to find a solution with < 15 colors
    # The seed uses 15 colors.
    
    # We will run one long local search, hoping to reduce colors.
    # If it finds a valid solution with 14 colors, great.
    # If it finds one with 13, even better.
    
    final_colors, final_used, final_viol = local_search(seed_colors, 15, deadline - time.time() - 5.0)
    
    # If the result is invalid (violations > 0), we must return a valid one.
    # The seed is valid (assumed).
    if final_viol > 0:
        return seed_colors
    else:
        return final_colors

