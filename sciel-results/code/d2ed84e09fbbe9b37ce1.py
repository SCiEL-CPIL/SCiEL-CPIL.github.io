import itertools
import time
import numpy as np
import random

def construct():
    deadline = time.time() + 135
    n = 16
    num_edges = n * (n - 1) // 2
    num_k4 = 1820
    
    # Precompute K4s and their edge indices
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    k4_edges = []
    k4s = list(itertools.combinations(range(n), 4))
    for a, b, c, d in k4s:
        edges = [
            edge_to_idx[(a, b)],
            edge_to_idx[(a, c)],
            edge_to_idx[(a, d)],
            edge_to_idx[(b, c)],
            edge_to_idx[(b, d)],
            edge_to_idx[(c, d)]
        ]
        k4_edges.append(edges)
        
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e_idx in edges:
            edge_to_k4s[e_idx].append(k4_idx)

    def is_valid_fast(coloring):
        for edges in k4_edges:
            c = [coloring[e] for e in edges]
            if len(set(c)) < 5:
                return False
        return True

    def solve_with_max_colors(max_c):
        # Simulated Annealing with distinct count tracking
        edge_colors = [random.randint(0, max_c - 1) for _ in range(num_edges)]
        
        # k4_distinct[k] = number of distinct colors in k-th K4
        k4_distinct = [0] * num_k4
        # k4_color_counts[k] = list of counts for each color in k-th K4
        k4_color_counts = [ [0]*max_c for _ in range(num_k4) ]
        
        # Initialize
        penalty = 0
        for k4_idx, edges in enumerate(k4_edges):
            distinct = 0
            for e_idx in edges:
                c = edge_colors[e_idx]
                if k4_color_counts[k4_idx][c] == 0:
                    distinct += 1
                k4_color_counts[k4_idx][c] += 1
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                penalty += 1
                
        best_penalty = penalty
        best_edge_colors = edge_colors.copy()
        
        T = 5.0
        cool_rate = 0.9995
        start_time = time.time()
        
        while time.time() < deadline - 10 and penalty > 0:
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_c = edge_colors[e_idx]
            new_c = random.randint(0, max_c - 1)
            
            if old_c == new_c:
                continue
                
            delta_penalty = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            for k4_idx in affected_k4s:
                # Update distinct count
                # Remove old_c
                count_old = k4_color_counts[k4_idx][old_c]
                if count_old == 1:
                    k4_distinct[k4_idx] -= 1
                k4_color_counts[k4_idx][old_c] -= 1
                
                # Add new_c
                count_new = k4_color_counts[k4_idx][new_c]
                if count_new == 0:
                    k4_distinct[k4_idx] += 1
                k4_color_counts[k4_idx][new_c] += 1
                
                # Calculate delta in penalty for this K4
                old_distinct = k4_distinct[k4_idx]
                # We need to know if it was < 5 before and is >= 5 now, or vice versa.
                # But we modified k4_distinct already.
                # Let's store old_distinct_before.
                
            # The above block modifies state in place. We need to be careful with delta calculation.
            # Let's revert to calculating delta based on old/new distinct values.
            
            # Let's redo the delta calculation properly.
            # 1. Calculate old penalty contribution of affected K4s
            # 2. Apply change to counts
            # 3. Calculate new penalty contribution
            # 4. Delta = New - Old
            # 5. If reject, revert counts
            
            # This is safer.
            
            # We already modified counts in the loop above? No, the code above is just a sketch.
            # Let's implement the safe version.
            
            # 1. Calculate old penalty for affected K4s
            old_pen_sub = 0
            for k4_idx in affected_k4s:
                if k4_distinct[k4_idx] < 5:
                    old_pen_sub += 1
                    
            # 2. Apply change to counts
            for k4_idx in affected_k4s:
                # Remove old_c
                count_old = k4_color_counts[k4_idx][old_c]
                if count_old == 1:
                    k4_distinct[k4_idx] -= 1
                k4_color_counts[k4_idx][old_c] -= 1
                
                # Add new_c
                count_new = k4_color_counts[k4_idx][new_c]
                if count_new == 0:
                    k4_distinct[k4_idx] += 1
                k4_color_counts[k4_idx][new_c] += 1
                
            # 3. Calculate new penalty for affected K4s
            new_pen_sub = 0
            for k4_idx in affected_k4s:
                if k4_distinct[k4_idx] < 5:
                    new_pen_sub += 1
                    
            delta = new_pen_sub - old_pen_sub
            penalty += delta
            
            if delta > 0:
                if random.random() > np.exp(-delta / T):
                    # Reject: Revert
                    edge_colors[e_idx] = old_c
                    for k4_idx in affected_k4s:
                        # Revert counts
                        # Remove new_c
                        count_new = k4_color_counts[k4_idx][new_c]
                        if count_new == 1:
                            k4_distinct[k4_idx] -= 1
                        k4_color_counts[k4_idx][new_c] -= 1
                        
                        # Add old_c
                        count_old = k4_color_counts[k4_idx][old_c]
                        if count_old == 0:
                            k4_distinct[k4_idx] += 1
                        k4_color_counts[k4_idx][old_c] += 1
                    penalty -= delta
            else:
                # Accept
                edge_colors[e_idx] = new_c
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_edge_colors = edge_colors.copy()
                    
            T *= cool_rate
            if T < 0.1:
                T = 0.1
                
        if penalty == 0:
            return best_edge_colors
        return None

    # Try to find a 13-coloring
    result = solve_with_max_colors(13)
    if result is not None:
        return result
        
    # If 13 fails, try 14
    if time.time() < deadline - 5:
        result = solve_with_max_colors(14)
        if result is not None:
            return result
            
    # Fallback
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 17)
    return colors

