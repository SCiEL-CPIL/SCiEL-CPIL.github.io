import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    total_edges = n * (n - 1) // 2
    
    # Precompute edge indices
    edges = []
    edge_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = len(edges)
            edge_idx[(j, i)] = len(edges)
            edges.append((i, j))
            
    # Precompute K4s
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
            edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
        ]
        k4s.append(e_idx)
        
    # Map edge to list of K4 indices
    edge_k4s = [[] for _ in range(total_edges)]
    for k4_idx, e_list in enumerate(k4s):
        for ei in e_list:
            edge_k4s[ei].append(k4_idx)
            
    def get_color_count(colors):
        return len(set(colors))
        
    def check_validity(colors):
        for e_list in k4s:
            if len(set(colors[ei] for ei in e_list)) < 5:
                return False
        return True
        
    def count_violations(colors):
        count = 0
        for e_list in k4s:
            if len(set(colors[ei] for ei in e_list)) < 5:
                count += 1
        return count

    # 1. Try 1-factorization of K16 (uses 15 colors)
    # Vertices 0..14, 15 is infinity
    colors_15 = [-1] * total_edges
    for r in range(15):
        # Edge (15, r)
        idx = edge_idx[(15, r)]
        colors_15[idx] = r
        for k in range(1, 8):
            u = (r + k) % 15
            v = (r - k) % 15
            if u > v:
                u, v = v, u
            idx = edge_idx[(u, v)]
            colors_15[idx] = r
            
    if -1 in colors_15:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    # Verify 1-factorization
    if check_validity(colors_15):
        best_colors = colors_15[:]
        best_count = 15
    else:
        # Fallback to (i+j)%17
        best_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        best_count = get_color_count(best_colors)
        
    # 2. Try to reduce colors using Simulated Annealing
    # Objective: Minimize number of colors, subject to validity.
    # Penalty: violations * large_number + color_count
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    max_colors = 15
    
    current_colors = best_colors[:]
    current_violations = count_violations(current_colors)
    current_count = get_color_count(current_colors)
    
    while time.time() < deadline and T > min_T:
        e_idx = random.randint(0, total_edges - 1)
        old_c = current_colors[e_idx]
        new_c = random.randint(0, max_colors - 1)
        if new_c == old_c:
            continue
            
        # Calculate delta violations
        old_v = 0
        new_v = 0
        for k4_idx in edge_k4s[e_idx]:
            e_list = k4s[k4_idx]
            # Old state
            old_set = set(current_colors[ei] for ei in e_list)
            if len(old_set) < 5:
                old_v += 1
            # New state
            current_colors[e_idx] = new_c
            new_set = set(current_colors[ei] for ei in e_list)
            if len(new_set) < 5:
                new_v += 1
            current_colors[e_idx] = old_c
            
        delta_v = new_v - old_v
        
        # Calculate delta color count
        # This is expensive to do exactly every step, so we approximate or update carefully
        # For now, just accept based on violations, and occasionally check count
        
        current_colors[e_idx] = new_c
        
        if delta_v <= 0:
            # Accept if violations don't increase
            new_count = get_color_count(current_colors)
            if new_count < best_count:
                best_count = new_count
                best_colors = current_colors[:]
        else:
            # Accept with probability exp(-delta_v / T)
            if random.random() < np.exp(-delta_v / T):
                new_count = get_color_count(current_colors)
                if new_count < best_count:
                    best_count = new_count
                    best_colors = current_colors[:]
            else:
                current_colors[e_idx] = old_c
                
        T *= cooling
        
    # 3. Final Polish: Try to merge colors in best_colors
    # If best_count is 15, try to force it to 14
    if best_count > 14 and time.time() < deadline - 5:
        # Try to recolor edges of the most frequent color to others
        color_counts = {}
        for c in best_colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        # Sort colors by frequency
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        
        for target_c, count in sorted_colors:
            if time.time() > deadline - 5:
                break
            # Try to remove color target_c
            # For each edge with color target_c, try to change it to another color
            edges_with_c = [i for i, c in enumerate(best_colors) if c == target_c]
            random.shuffle(edges_with_c)
            
            success = True
            for ei in edges_with_c:
                # Try other colors
                other_colors = [c for c in range(best_count) if c != target_c]
                random.shuffle(other_colors)
                found = False
                for new_c in other_colors:
                    # Check validity
                    valid = True
                    for k4_idx in edge_k4s[ei]:
                        e_list = k4s[k4_idx]
                        # Simulate change
                        cset = set(best_colors[ei2] for ei2 in e_list)
                        if best_colors[ei] in cset:
                            cset.remove(best_colors[ei])
                        cset.add(new_c)
                        if len(cset) < 5:
                            valid = False
                            break
                    if valid:
                        best_colors[ei] = new_c
                        found = True
                        break
                if not found:
                    success = False
                    break
                    
            if success:
                # Update best_count
                best_count = get_color_count(best_colors)
                if best_count <= 14:
                    break
                    
    # Compact labels
    unique = sorted(set(best_colors))
    mapping = {c: i for i, c in enumerate(unique)}
    return [mapping[c] for c in best_colors]
