import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n = 16
    # Edge list in lexicographic order
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    m = len(edges)  # 120 edges
    
    # Map (u,v) to edge index
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
    num_k4 = len(k4s)
    
    # For each edge, which K4s does it appear in?
    edge_k4s = [[] for _ in range(m)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
    
    # Helper: count distinct colors in a K4 given a colors array
    def count_distinct_in_k4(k4_idx, colors):
        e_list = k4s[k4_idx]
        cset = set()
        for ei in e_list:
            cset.add(colors[ei])
        return len(cset)
    
    def is_valid_k4(k4_idx, colors):
        return count_distinct_in_k4(k4_idx, colors) >= 5
    
    def total_penalty(colors):
        """Sum over all K4s of max(0, 5 - distinct_colors)"""
        pen = 0
        for k4_idx in range(num_k4):
            dc = count_distinct_in_k4(k4_idx, colors)
            if dc < 5:
                pen += (5 - dc)
        return pen
    
    def num_distinct(colors):
        return len(set(colors))
    
    # Try simple algebraic constructions first
    best_colors = None
    best_count = float('inf')
    
    # (i+j) mod p for various p
    for p in range(13, 20):
        colors = [(i + j) % p for (i, j) in edges]
        pen = total_penalty(colors)
        if pen == 0:
            nc = num_distinct(colors)
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break
    
    # (i*j + i + j) mod p
    for p in range(13, 20):
        colors = [(i * j + i + j) % p for (i, j) in edges]
        pen = total_penalty(colors)
        if pen == 0:
            nc = num_distinct(colors)
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break
    
    # (i^2 + j^2) mod p
    for p in range(13, 20):
        colors = [(i * i + j * j) % p for (i, j) in edges]
        pen = total_penalty(colors)
        if pen == 0:
            nc = num_distinct(colors)
            if nc < best_count:
                best_count = nc
                best_colors = colors[:]
            if nc <= 13:
                break
    
    # (i+j) mod p with offset
    for p in range(13, 20):
        for off in range(p):
            colors = [((i + j + off) % p) for (i, j) in edges]
            pen = total_penalty(colors)
            if pen == 0:
                nc = num_distinct(colors)
                if nc < best_count:
                    best_count = nc
                    best_colors = colors[:]
                if nc <= 13:
                    break
        if best_count <= 13:
            break
    
    if best_colors is not None and best_count <= 13:
        unique = sorted(set(best_colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in best_colors]
    
    # Use simulated annealing with penalty
    # Start from a random assignment or from a good algebraic seed
    if best_colors is not None:
        colors = best_colors[:]
    else:
        # Start from (i+j) % 17 which is known to be valid
        colors = [(i + j) % 17 for (i, j) in edges]
    
    current_penalty = total_penalty(colors)
    current_count = num_distinct(colors)
    best_colors = colors[:]
    best_count = current_count
    best_penalty = current_penalty
    
    # Simulated annealing
    max_k = 15  # allow up to 15 colors in the palette during search
    T = 10.0
    cooling = 0.9997
    min_T = 0.005
    
    # Track best valid solution
    step_count = 0
    no_improve = 0
    improve_threshold = 5000
    
    while time.time() < deadline - 5.0:
        e_idx = random.randint(0, m - 1)
        old_c = colors[e_idx]
        new_c = random.randint(0, max_k - 1)
        if new_c == old_c:
            continue
        
        # Compute penalty change efficiently
        old_pen_local = 0
        new_pen_local = 0
        for k4_idx in edge_k4s[e_idx]:
            e_list = k4s[k4_idx]
            # Count before
            cset_old = set()
            for ei in e_list:
                if ei == e_idx:
                    cset_old.add(old_c)
                else:
                    cset_old.add(colors[ei])
            dc_old = len(cset_old)
            old_pen_local += (5 - dc_old) if dc_old < 5 else 0
            
            # Count after
            cset_new = set()
            for ei in e_list:
                if ei == e_idx:
                    cset_new.add(new_c)
                else:
                    cset_new.add(colors[ei])
            dc_new = len(cset_new)
            new_pen_local += (5 - dc_new) if dc_new < 5 else 0
        
        delta_pen = new_pen_local - old_pen_local
        colors[e_idx] = new_c
        current_penalty += delta_pen
        
        # Also track color count change
        # If we changed from old_c to new_c, the set of colors might change
        # We'll just recompute periodically
        
        step_count += 1
        
        if delta_pen <= 0:
            # Accept
            if current_penalty < best_penalty or (current_penalty == best_penalty and num_distinct(colors) < best_count):
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                # Check if we have a valid solution
                if current_penalty == 0:
                    nc = num_distinct(colors)
                    if nc < best_count:
                        best_count = nc
                        best_colors = colors[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        else:
            # Accept with probability
            if random.random() < math.exp(-delta_pen / T):
                no_improve += 1
            else:
                colors[e_idx] = old_c
                current_penalty -= delta_pen
                no_improve += 1
        
        T *= cooling
        
        # If no improvement for a long time and we have a valid solution, 
        # try to do a focused reduction
        if no_improve > improve_threshold and best_penalty == 0:
            # Try to reduce colors in best_colors
            no_improve = 0
            improve_threshold = int(improve_threshold * 1.5)
            if best_count > 13:
                # Attempt local search to reduce colors
                bc = best_colors[:]
                for _ in range(2000):
                    if time.time() > deadline - 5.0:
                        break
                    e_idx2 = random.randint(0, m - 1)
                    old_c2 = bc[e_idx2]
                    # Try to change to a color already in use (not old_c2)
                    used = set(bc)
                    if len(used) <= 13:
                        break
                    candidates = [c for c in used if c != old_c2]
                    if not candidates:
                        continue
                    new_c2 = random.choice(candidates)
                    # Check validity
                    valid = True
                    for k4_idx in edge_k4s[e_idx2]:
                        e_list = k4s[k4_idx]
                        cset = set(bc[ei] for ei in e_list)
                        cset.discard(old_c2)
                        cset.add(new_c2)
                        if len(cset) < 5:
                            valid = False
                            break
                    if valid:
                        bc[e_idx2] = new_c2
                nc2 = num_distinct(bc)
                if nc2 < best_count:
                    best_count = nc2
                    best_colors = bc[:]
    
    # Final polish: try to reduce colors in best solution
    if best_penalty == 0 and best_count > 13:
        bc = best_colors[:]
        # Iteratively try to recolor edges to eliminate one color at a time
        for target_colors_to_remove in range(best_count - 1, 12, -1):
            if time.time() > deadline - 3.0:
                break
            if num_distinct(bc) <= 13:
                break
            # Try to eliminate the least-used color
            color_counts = {}
            for c in bc:
                color_counts[c] = color_counts.get(c, 0) + 1
            # Remove the least frequent color
            min_color = min(color_counts, key=color_counts.get)
            # Try to recolor all edges with min_color to other colors
            success = True
            for e_idx in range(m):
                if bc[e_idx] == min_color:
                    # Try other colors
                    found = False
                    for new_c in range(max_k):
                        if new_c == min_color:
                            continue
                        # Check validity
                        valid = True
                        for k4_idx in edge_k4s[e_idx]:
                            e_list = k4s[k4_idx]
                            cset = set(bc[ei] for ei in e_list)
                            cset.discard(min_color)
                            cset.add(new_c)
                            if len(cset) < 5:
                                valid = False
                                break
                        if valid:
                            bc[e_idx] = new_c
                            found = True
                            break
                    if not found:
                        success = False
                        break
            if success:
                # Verify
                if total_penalty(bc) == 0:
                    best_colors = bc[:]
                    best_count = num_distinct(bc)
    
    # Compact labels
    unique = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique)}
    result = [mapping[c] for c in best_colors]
    
    return result
