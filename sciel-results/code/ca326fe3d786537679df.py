import time
import random
import itertools
from collections import Counter
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all K4s and their edges
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
    
    # Precompute k4 edge indices for fast access
    k4_edge_indices = []
    for edges in k4_edges:
        k4_edge_indices.append([edge_index[e] for e in edges])
    k4_edge_indices = np.array(k4_edge_indices, dtype=np.int32)  # shape (num_k4s, 6)
    
    # For each edge, which K4s does it belong to?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[edge_index[e]].append(k4_idx)
    edge_k4s_arrays = [np.array(e, dtype=np.int32) for e in edge_k4s]
    
    # Start with (i+j) mod 17 coloring
    colors = np.array([(i + j) % 17 for i in range(n) for j in range(i+1, n)], dtype=np.int32)
    
    def count_colors_used(c):
        return len(set(c.tolist()))
    
    def count_violations(c):
        """Count K4s with fewer than 5 distinct colors. Vectorized."""
        # c has shape (num_edges,)
        # For each K4, get the 6 edge colors
        k4_colors = c[k4_edge_indices]  # shape (num_k4s, 6)
        # Count distinct colors per K4
        # A K4 has < 5 distinct colors iff it has <= 4 distinct colors
        # Check: number of distinct colors in each row
        # For 6 elements, distinct count < 5 means at least one duplicate
        # More efficient: check if any two pairs are equal (but that's complex)
        # Simple approach: for each K4, count distinct
        violations = 0
        for k4_idx in range(num_k4s):
            k4c = k4_colors[k4_idx]
            if len(set(k4c.tolist())) < 5:
                violations += 1
        return violations
    
    def is_valid(c):
        return count_violations(c) == 0
    
    # Initial check
    best_colors = colors.copy()
    best_count = count_colors_used(best_colors)
    best_violations = count_violations(best_colors)
    
    if best_violations > 0:
        # The initial coloring isn't valid, need to fix it
        # Try different mod values
        for mod in range(13, 20):
            test_colors = np.array([(i + j) % mod for i in range(n) for j in range(i+1, n)], dtype=np.int32)
            v = count_violations(test_colors)
            if v == 0:
                cc = count_colors_used(test_colors)
                if cc < best_count or (cc == best_count and v < best_violations):
                    best_colors = test_colors.copy()
                    best_count = cc
                    best_violations = 0
                break
    
    # Local search: change individual edge colors to reduce violations
    current_colors = best_colors.copy()
    current_violations = best_violations
    current_count = best_count
    
    def try_improve(c, c_violations, c_count, max_iters, time_limit):
        """Local search to reduce violations first, then colors."""
        c = c.copy()
        cv = c_violations
        cc = c_count
        best_c = c.copy()
        best_cv = cv
        best_cc = cc
        
        start_time = time.time()
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            old_color = c[e]
            
            # Try changing to a different color
            # Prefer colors that are already in use, or 0 if trying to reduce
            if cv > 0:
                # Try all colors 0..max_color+1 to fix violations
                max_col = max(c.max(), 10)
                new_colors = list(range(max_col + 1))
                random.shuffle(new_colors)
                new_colors = [nc for nc in new_colors if nc != old_color][:5]
            else:
                # Try to reduce color count: map to existing colors
                existing = list(set(c.tolist()))
                if len(existing) <= 1:
                    break
                new_colors = [nc for nc in existing if nc != old_color]
                random.shuffle(new_colors)
                new_colors = new_colors[:3]
            
            improved = False
            for nc in new_colors:
                c[e] = nc
                new_v = count_violations(c)
                new_c = count_colors_used(c)
                
                if new_v < cv or (new_v == cv and new_c < cc):
                    cv = new_v
                    cc = new_c
                    improved = True
                    if cv < best_cv or (cv == best_cv and cc < best_cc):
                        best_c = c.copy()
                        best_cv = cv
                        best_cc = cc
                    break
                else:
                    c[e] = old_color
            
            if not improved and cv == 0:
                # No improvement possible, try harder: random restarts
                if it % 1000 == 999:
                    # Occasionally try a random perturbation
                    if random.random() < 0.1:
                        e2 = random.randint(0, num_edges - 1)
                        c[e2] = random.randint(0, max(c.max(), 10))
                        cv = count_violations(c)
                        cc = count_colors_used(c)
        
        return best_c, best_cv, best_cc
    
    # Phase 1: Fix violations if any
    if current_violations > 0:
        current_colors, current_violations, current_count = try_improve(
            current_colors, current_violations, current_count, 
            max_iters=50000, time_limit=deadline - 30
        )
        if current_violations == 0:
            best_colors = current_colors.copy()
            best_count = current_count
            best_violations = 0
    
    # Phase 2: Reduce color count
    if best_violations == 0 and time.time() < deadline - 20:
        # Greedy color merging
        current_colors = best_colors.copy()
        current_count = best_count
        
        for _ in range(50):
            if time.time() > deadline - 20:
                break
            
            colors_used = sorted(set(current_colors.tolist()))
            merged = False
            for i in range(len(colors_used)):
                for j in range(i+1, len(colors_used)):
                    if time.time() > deadline - 20:
                        break
                    new_c = current_colors.copy()
                    new_c[new_c == colors_used[j]] = colors_used[i]
                    if count_violations(new_c) == 0:
                        current_colors = new_c
                        current_count = count_colors_used(current_colors)
                        merged = True
                        break
                if merged or time.time() > deadline - 20:
                    break
            if not merged:
                break
        
        if current_count < best_count:
            best_colors = current_colors.copy()
            best_count = current_count
    
    # Phase 3: Local search to reduce colors further
    if best_violations == 0 and time.time() < deadline - 10:
        improved_colors, improved_v, improved_c = try_improve(
            best_colors, 0, best_count,
            max_iters=100000, time_limit=deadline - 5
        )
        if improved_v == 0 and improved_c < best_count:
            best_colors = improved_colors.copy()
            best_count = improved_c
    
    # Final validation
    if count_violations(best_colors) != 0:
        # Fallback: try to find any valid coloring
        for mod in range(13, 25):
            test_colors = np.array([(i + j) % mod for i in range(n) for j in range(i+1, n)], dtype=np.int32)
            if count_violations(test_colors) == 0:
                best_colors = test_colors.copy()
                best_count = count_colors_used(best_colors)
                break
        else:
            # Last resort: use 16 colors with a known valid pattern
            # (i+j) % 16 might not be valid, use a different construction
            # Simply assign unique colors to ensure validity
            best_colors = np.array([i * 16 + j for i in range(n) for j in range(i+1, n)], dtype=np.int32)
            best_count = count_colors_used(best_colors)
    
    return best_colors.tolist()
