import time
import random
import itertools

def construct():
    deadline = time.time() + 140  # safety margin
    
    n = 16
    # Precompute all K4s and their edges
    vertices = list(range(n))
    k4s = []
    k4_edges = []
    for combo in itertools.combinations(vertices, 4):
        a, b, c, d = combo
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(combo)
        k4_edges.append(edges)
    
    num_k4s = len(k4s)
    
    # Edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_list.append((i,j))
            idx += 1
    num_edges = len(edge_list)
    
    # Precompute for each edge, which K4s it belongs to
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[edge_index[e]].append(k4_idx)
    
    # Start with (i+j) mod 17 coloring
    colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
    
    # Count colors used
    def count_colors_used(c):
        return len(set(c))
    
    # Check if coloring is valid (every K4 has >= 5 distinct colors)
    def is_valid(c):
        for k4_idx in range(num_k4s):
            edges = k4_edges[k4_idx]
            k4_colors = set()
            for e in edges:
                k4_colors.add(c[edge_index[e]])
            if len(k4_colors) < 5:
                return False
        return True
    
    # Count violations (K4s with < 5 colors)
    def count_violations(c):
        violations = 0
        for k4_idx in range(num_k4s):
            edges = k4_edges[k4_idx]
            k4_colors = set()
            for e in edges:
                k4_colors.add(c[edge_index[e]])
            if len(k4_colors) < 5:
                violations += 1
        return violations
    
    # Initial check
    if is_valid(colors):
        best_colors = colors[:]
        best_count = count_colors_used(colors)
    else:
        best_colors = colors[:]
        best_count = count_colors_used(colors)
    
    # Try to reduce colors by remapping
    # First, try to find a valid coloring with fewer colors using local search
    
    # Strategy: try to merge colors
    # For each pair of colors (a, b), try to merge them and check validity
    
    def try_merge(colors_arr, color_a, color_b):
        """Try to merge color_b into color_a. Returns new coloring or None if invalid."""
        new_colors = colors_arr[:]
        for i in range(num_edges):
            if new_colors[i] == color_b:
                new_colors[i] = color_a
        if is_valid(new_colors):
            return new_colors
        return None
    
    # Greedy color merging
    current_colors = colors[:]
    current_color_count = count_colors_used(current_colors)
    
    # Try to reduce colors
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        
        colors_used = sorted(set(current_colors))
        merged = False
        for i in range(len(colors_used)):
            for j in range(i+1, len(colors_used)):
                if time.time() > deadline - 10:
                    break
                new_c = try_merge(current_colors, colors_used[i], colors_used[j])
                if new_c is not None:
                    current_colors = new_c
                    current_color_count = count_colors_used(current_colors)
                    merged = True
                    break
            if merged or time.time() > deadline - 10:
                break
        if not merged:
            break
    
    # Local search: try changing individual edge colors to reduce total colors
    def local_search_reduce_colors(colors_arr, max_iters=500):
        """Try to reduce color count by changing edge colors."""
        c = colors_arr[:]
        current_count = count_colors_used(c)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            colors_used = list(set(c))
            if len(colors_used) <= 13:
                break
            
            # Try to eliminate the least-used color
            from collections import Counter
            color_freq = Counter(c)
            # Pick a color to eliminate (least frequent, but not the most frequent)
            sorted_colors = sorted(color_freq.items(), key=lambda x: x[1])
            
            eliminated = False
            for elim_color, freq in sorted_colors[:3]:  # Try top 3 least frequent
                if freq > len(colors_used) // 2:
                    continue
                
                # Find a target color to remap to
                for target_color in colors_used:
                    if target_color == elim_color:
                        continue
                    
                    new_c = c[:]
                    for i in range(num_edges):
                        if new_c[i] == elim_color:
                            new_c[i] = target_color
                    
                    if is_valid(new_c):
                        c = new_c
                        current_count = count_colors_used(c)
                        eliminated = True
                        break
                
                if eliminated:
                    break
            
            if not eliminated:
                break
        
        return c, current_count
    
    # Apply local search
    improved_colors, improved_count = local_search_reduce_colors(current_colors)
    if improved_count < best_count:
        best_colors = improved_colors
        best_count = improved_count
    
    # Final validation
    if not is_valid(best_colors):
        # Fallback to original
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i+1, n)]
        best_count = 17
    
    return best_colors
