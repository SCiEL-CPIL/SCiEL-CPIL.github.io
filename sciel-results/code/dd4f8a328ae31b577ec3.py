import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Index mapping: (i,j) with i<j -> index
    # index = i*(2n-i-1)//2 + (j-i-1)
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
    
    # Build all K4s (1820 of them)
    from itertools import combinations
    k4s = list(combinations(range(n), 4))
    
    # Precompute edges for each K4
    k4_edges = []
    for a, b, c, d in k4s:
        edges = [
            edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
            edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
        ]
        k4_edges.append(edges)
    
    # Start with 15-color 1-factorization
    def make_initial_coloring():
        colors = [0] * total_edges
        for r in range(15):
            # Edge (15, r)
            idx = edge_idx(15, r)
            colors[idx] = r
            for k in range(1, 8):
                u = (r + k) % 15
                v = (r - k) % 15
                idx = edge_idx(u, v)
                colors[idx] = r
        return colors
    
    colors = make_initial_coloring()
    
    # Count distinct colors in each K4
    def count_violations(colors):
        """Return number of K4s with fewer than 5 distinct colors"""
        violations = 0
        for edges in k4_edges:
            c0, c1, c2, c3, c4, c5 = (colors[e] for e in edges)
            # Check if at most 4 distinct colors
            distinct = len({c0, c1, c2, c3, c4, c5})
            if distinct < 5:
                violations += 1
        return violations
    
    # Count how many times each color appears in each K4
    # For fast penalty updates when changing an edge's color
    # k4_color_count[k4_idx][color] = count of that color in this K4
    # But we need to track distinct count, so let's use a different approach
    
    # For each edge, precompute which K4s contain it
    edge_to_k4s = [[] for _ in range(total_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Build color count per K4: k4_counts[k4_idx] = dict or array of color->count
    # Since max 15 colors initially, use arrays of size 16
    max_colors = 16
    k4_color_counts = [np.zeros(max_colors, dtype=np.int8) for _ in range(len(k4s))]
    
    def build_k4_counts(colors):
        for k4_idx in range(len(k4s)):
            k4_color_counts[k4_idx] = np.zeros(max_colors, dtype=np.int8)
        for e, c in enumerate(colors):
            for k4_idx in edge_to_k4s[e]:
                k4_color_counts[k4_idx][c] += 1
    
    build_k4_counts(colors)
    
    # Get distinct count for a K4 from its color counts
    def k4_distinct(k4_idx):
        return np.count_nonzero(k4_color_counts[k4_idx])
    
    # Current violation count
    current_violations = sum(1 for k in range(len(k4s)) if k4_distinct(k) < 5)
    
    # Local search: try to reduce colors
    # Strategy: try to recolor edges to use fewer distinct colors overall
    # while maintaining the K4 constraint
    
    best_colors = colors[:]
    best_violations = current_violations
    best_num_colors = len(set(colors))
    
    # Simulated annealing / local search
    random.seed(42)
    
    # Try to reduce number of colors
    # First, find all colors actually used
    used_colors = sorted(set(colors))
    
    # Try to merge color classes
    # For each pair of colors, try to recolor all edges of color b to color a
    # and check if violations stay the same or decrease
    
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        iteration += 1
        
        # Randomly pick an edge and a new color
        e = random.randint(0, total_edges - 1)
        old_color = colors[e]
        
        # Try to recolor to a color that's already used, or a new lower color
        # Focus on reducing total distinct colors
        used = set(colors)
        if len(used) <= 13:
            # Already at 13, try to maintain
            new_color = random.choice(list(used))
        else:
            # Try to use a color that's in the lower range
            # or merge two colors
            if random.random() < 0.7:
                # Pick a random used color
                new_color = random.choice(list(used))
            else:
                # Try to introduce a new low color (0..12)
                new_color = random.randint(0, 12)
        
        if new_color == old_color:
            continue
        
        # Compute delta in violations
        # For each K4 containing edge e:
        # - old_color count decreases by 1
        # - new_color count increases by 1
        # - distinct count changes based on whether old_color becomes 0 and/or new_color was 0
        
        delta_violations = 0
        affected_k4s = edge_to_k4s[e]
        
        for k4_idx in affected_k4s:
            old_count = k4_color_counts[k4_idx][old_color]
            new_count = k4_color_counts[k4_idx][new_color]
            
            old_distinct = np.count_nonzero(k4_color_counts[k4_idx])
            
            # After change:
            new_old_count = old_count - 1
            new_new_count = new_count + 1
            
            # New distinct count
            new_distinct = old_distinct
            if new_old_count == 0:
                new_distinct -= 1
            if new_new_count == 1:  # was 0, now 1
                new_distinct += 1
            
            old_violation = 1 if old_distinct < 5 else 0
            new_violation = 1 if new_distinct < 5 else 0
            delta_violations += new_violation - old_violation
        
        # Accept if no increase in violations, or with some probability if it helps reduce colors
        old_num_colors = len(set(colors))
        
        # Temporarily apply
        colors[e] = new_color
        for k4_idx in affected_k4s:
            k4_color_counts[k4_idx][old_color] -= 1
            k4_color_counts[k4_idx][new_color] += 1
        
        new_num_colors = len(set(colors))
        
        if delta_violations <= 0 and new_num_colors <= old_num_colors:
            # Good move
            current_violations += delta_violations
            if current_violations < best_violations or (current_violations == best_violations and new_num_colors < best_num_colors):
                best_violations = current_violations
                best_num_colors = new_num_colors
                best_colors = colors[:]
        else:
            # Revert
            colors[e] = old_color
            for k4_idx in affected_k4s:
                k4_color_counts[k4_idx][old_color] += 1
                k4_color_counts[k4_idx][new_color] -= 1
    
    # Final check: try to further reduce colors by merging
    # Get best solution
    colors = best_colors[:]
    
    # Try to reduce colors in the best solution
    used = sorted(set(colors))
    for target_color in used[:]:
        # Try to recolor all edges with target_color to another color
        for other_color in used:
            if other_color == target_color:
                continue
            # Try recoloring all edges of target_color to other_color
            edges_to_recolor = [e for e, c in enumerate(colors) if c == target_color]
            if not edges_to_recolor:
                continue
            
            # Check if this would maintain validity
            test_colors = colors[:]
            for e in edges_to_recolor:
                test_colors[e] = other_color
            
            # Check violations
            valid = True
            for k4_idx, edges in enumerate(k4_edges):
                c0, c1, c2, c3, c4, c5 = (test_colors[e] for e in edges)
                if len({c0, c1, c2, c3, c4, c5}) < 5:
                    valid = False
                    break
            
            if valid:
                colors = test_colors
                used = sorted(set(colors))
                break
    
    # Final verification
    final_valid = True
    for k4_idx, edges in enumerate(k4_edges):
        c0, c1, c2, c3, c4, c5 = (colors[e] for e in edges)
        if len({c0, c1, c2, c3, c4, c5}) < 5:
            final_valid = False
            break
    
    if final_valid:
        return colors
    else:
        # Fallback to original 15-coloring
        return make_initial_coloring()
