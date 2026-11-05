"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s and their edge indices
    k4s = []
    edge_idx_map = {e: i for i, e in enumerate(edges)}
    for quad in combinations(range(n), 4):
        v0, v1, v2, v3 = quad
        e_indices = [
            edge_idx_map[(v0, v1)],
            edge_idx_map[(v0, v2)],
            edge_idx_map[(v0, v3)],
            edge_idx_map[(v1, v2)],
            edge_idx_map[(v1, v3)],
            edge_idx_map[(v2, v3)]
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # 1820
    
    # Precompute which K4s contain each edge
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4s[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)
    
    # Fallback solution: (i+j) % 17
    seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    def count_colors(col):
        return len(set(col))
    
    def is_valid(col):
        for k in range(num_k4s):
            if len(set(col[e] for e in k4s[k])) < 5:
                return False
        return True

    best_result = None
    best_color_count = float('inf')
    
    # Check seed
    if is_valid(seed_col):
        best_result = seed_col
        best_color_count = count_colors(seed_col)
        
    # GF(4) construction for initial coloring
    # Represent elements of GF(4) as pairs (a, b) where a, b in {0, 1}
    # Addition: XOR
    # Multiplication: (a1, b1) * (a2, b2) = (a1*a2 + b1*b2, a1*b2 + b1*a2) mod 2
    # Actually, let's use a simpler representation.
    # GF(4) = {0, 1, alpha, alpha^2} where alpha^2 = alpha + 1
    # Let's map: 0->0, 1->1, alpha->2, alpha^2->3
    # Addition table (XOR-like but with carries in GF(2)[x]/(x^2+x+1)):
    # Actually, for GF(4), addition is just XOR if we represent elements as bits.
    # 0 = 00, 1 = 01, alpha = 10, alpha^2 = 11
    # Addition is bitwise XOR.
    # Multiplication:
    # 0*x = 0
    # 1*x = x
    # alpha * 1 = alpha
    # alpha * alpha = alpha^2
    # alpha * alpha^2 = alpha^3 = 1 (since alpha^3 = alpha * alpha^2 = alpha(alpha+1) = alpha^2 + alpha = (alpha+1)+alpha = 1)
    # alpha^2 * alpha^2 = alpha^4 = alpha (since alpha^3=1, alpha^4=alpha)
    
    # Let's define multiplication table for indices 0,1,2,3
    # 0: 0,0,0,0
    # 1: 0,1,2,3
    # 2 (alpha): 0,2,3,1
    # 3 (alpha^2): 0,3,1,2
    mult_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    # Assign vertices to GF(4)^2
    # There are 16 points in GF(4)^2.
    # Map vertex i to point (i // 4, i % 4) in GF(4)^2
    # Color of edge (u, v) is the slope of the line connecting them, or something similar.
    # Slope m = (y2 - y1) / (x2 - x1)
    # If x2 == x1, slope is infinity.
    
    # Let's try: color(u, v) = slope * (x_u + x_v) or something?
    # The idea is that for any 4 points, the slopes should be distinct enough.
    
    # Actually, let's just use the difference coloring as a strong initial guess,
    # and then use local search to try to reduce colors or fix violations.
    
    # Difference coloring: color(i, j) = (i - j) mod 15 (for i != j)
    # But we have 16 vertices.
    # Let's use color(i, j) = (i + j) % 16? No, that's not a proper difference.
    
    # Let's stick to the local search approach but optimize it significantly.
    # Key optimization: Use NumPy for the distinct color count?
    # No, 1820 K4s is small. Python loops are okay if we minimize overhead.
    
    # Let's try a different local search: Simulated Annealing with a good initial state.
    # Initial state: Random assignment of 13 colors.
    
    MAX_COLORS = 20
    
    def fast_local_search(initial_col, max_colors, max_iters, time_limit):
        col = initial_col[:]
        # k4_color_counts[k][c] = count of color c in k4 k
        # To save memory and time, we can store the distinct count and the counts.
        # But updating distinct count requires knowing if a count goes to/from 0.
        
        # Let's use a dictionary for counts per K4? No, lists are faster.
        k4_color_counts = [[0] * MAX_COLORS for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        current_violations = 0
        
        # Initialize state
        for k in range(num_k4s):
            counts = [0] * MAX_COLORS
            for e in k4s[k]:
                c = col[e]
                if c < MAX_COLORS:
                    counts[c] += 1
            k4_color_counts[k] = counts
            d = sum(1 for x in counts if x > 0)
            k4_distinct[k] = d
            if d < 5:
                current_violations += 1
                
        if current_violations == 0:
            return col, count_colors(col)
            
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
                
            e_idx = random.randint(0, num_edges - 1)
            old_color = col[e_idx]
            new_color = random.randint(0, max_colors - 1)
            
            if old_color == new_color:
                continue
            
            if old_color >= MAX_COLORS or new_color >= MAX_COLORS:
                continue
                
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Apply change
            col[e_idx] = new_color
            
            for k in affected_k4s:
                counts = k4_color_counts[k]
                old_d = k4_distinct[k]
                
                # Update old color
                counts[old_color] -= 1
                # Update new color
                counts[new_color] += 1
                
                # Recalculate distinct count for this k4
                new_d = sum(1 for x in counts if x > 0)
                k4_distinct[k] = new_d
                
                if old_d < 5 and new_d >= 5:
                    delta_viol -= 1
                elif old_d >= 5 and new_d < 5:
                    delta_viol += 1
                    
            if delta_viol <= 0:
                current_violations += delta_viol
            else:
                if random.random() < np.exp(-delta_viol / T):
                    current_violations += delta_viol
                else:
                    # Revert
                    col[e_idx] = old_color
                    for k in affected_k4s:
                        counts = k4_color_counts[k]
                        
                        # Revert counts
                        counts[new_color] -= 1
                        counts[old_color] += 1
                        
                        # Recalculate distinct
                        new_d = sum(1 for x in counts if x > 0)
                        k4_distinct[k] = new_d
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
            if current_violations == 0:
                return col, count_colors(col)
                
        return None, float('inf')

    # Try to find better solutions
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 20.0:
            break
            
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline - 10.0:
                break
                
            # Random initialization
            col = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            
            # Run fast local search
            result_col, c_count = fast_local_search(col, target_colors, 5000, deadline - 5.0)
            
            if result_col is not None:
                if c_count < best_color_count:
                    best_color_count = c_count
                    best_result = result_col
                if c_count <= 13:
                    return best_result

    # Polish: Try to reduce best_result if it has > 13 colors
    if best_result is not None and best_color_count > 13:
        # Try to eliminate colors from best_result
        # We can use the fast local search with a constraint that we try to reduce color count?
        # Or just try to recolor edges to reduce the set of colors used.
        
        # A simple approach: Try to recolor edges to use fewer colors.
        # This is tricky. Let's just return the best found.
        pass

    if best_result is not None:
        return best_result
    
    return seed_col
# EVOLVE-BLOCK-END
