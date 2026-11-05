"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    n = 16
    # Generate all edges in lexicographic order: (i, j) for 0 <= i < j < 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute all K4s (sets of 4 vertices) and their edge indices
    from itertools import combinations
    k4s = []
    for quad in combinations(range(n), 4):
        # For a K4 with vertices v0, v1, v2, v3, the edges are:
        # (v0,v1), (v0,v2), (v0,v3), (v1,v2), (v1,v3), (v2,v3)
        v0, v1, v2, v3 = quad
        # Map vertex pair to edge index
        # Edge (i,j) has index: sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # Simpler: precompute a lookup
        edge_idx = {}
        for idx, (a, b) in enumerate(edges):
            edge_idx[(a, b)] = idx
        e_indices = [
            edge_idx[(v0, v1)],
            edge_idx[(v0, v2)],
            edge_idx[(v0, v3)],
            edge_idx[(v1, v2)],
            edge_idx[(v1, v3)],
            edge_idx[(v2, v3)],
        ]
        k4s.append(e_indices)
    
    num_k4s = len(k4s)  # C(16,4) = 1820
    
    # We want to minimize the number of distinct colors used.
    # Lower bound is 13 (from general bounds 13 <= f(16) <= 15).
    # Let's try to find a coloring with 13 colors, then 14, then 15.
    
    # Strategy: Use a randomized local search / simulated annealing approach.
    # Objective: Minimize number of distinct colors.
    # Constraint: Every K4 must have at least 5 distinct colors.
    
    # Start with the seed coloring: (i+j) % 17, which uses 17 colors.
    # But we want to reduce to 13-15.
    
    # Let's try a different approach: start with a random assignment of colors in [0, max_colors-1]
    # and use local search to fix violations while trying to reduce color count.
    
    # First, let's try to construct a valid coloring with 13 colors.
    # If we can't find one quickly, try 14, then 15.
    
    # Key insight: The constraint is that every K4 has >= 5 distinct colors.
    # Equivalently, no K4 can have <= 4 distinct colors.
    # A K4 has 6 edges. If it uses <= 4 colors, then by pigeonhole, some color appears on >= 2 edges.
    # More specifically, the "bad" configurations are K4s where the 6 edges use at most 4 colors.
    
    # Let's use a constraint-based approach with randomized restarts.
    
    def count_colors(col):
        return len(set(col))
    
    def is_valid(col):
        """Check if every K4 has at least 5 distinct colors."""
        for e_idx in k4s:
            colors_in_k4 = set(col[i] for i in e_idx)
            if len(colors_in_k4) < 5:
                return False
        return True
    
    def count_violations(col):
        """Count number of K4s with < 5 distinct colors."""
        violations = 0
        for e_idx in k4s:
            colors_in_k4 = set(col[i] for i in e_idx)
            if len(colors_in_k4) < 5:
                violations += 1
        return violations
    
    # Try different target numbers of colors
    best_result = None
    best_color_count = float('inf')
    
    # Start with the seed solution as a fallback
    seed_col = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    if is_valid(seed_col):
        best_result = seed_col
        best_color_count = count_colors(seed_col)
    
    # Try to find solutions with fewer colors
    for target_colors in [13, 14, 15]:
        if time.time() > deadline - 20.0:
            break
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Random initialization with target_colors colors
            col = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            
            # Local search: repeatedly pick a random edge, change its color,
            # accept if it doesn't increase violations and reduces color count or same.
            
            current_violations = count_violations(col)
            current_colors = count_colors(col)
            
            if current_violations == 0:
                # Found a valid solution
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col[:]
                break
            
            # Local search
            no_improve = 0
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - 10.0:
                    break
                
                # Pick a random edge
                e = random.randint(0, num_edges - 1)
                old_color = col[e]
                
                # Try a new random color
                new_color = random.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                col[e] = new_color
                new_violations = count_violations(col)
                new_colors = count_colors(col)
                
                # Accept if violations don't increase, and prefer fewer colors
                if new_violations <= current_violations:
                    if new_colors < current_colors or (new_colors == current_colors and new_violations < current_violations):
                        current_violations = new_violations
                        current_colors = new_colors
                        no_improve = 0
                    elif new_violations == current_violations and new_colors <= current_colors:
                        # Accept with some probability to escape local optima
                        if random.random() < 0.1:
                            current_violations = new_violations
                            current_colors = new_colors
                            no_improve = 0
                        else:
                            col[e] = old_color
                    else:
                        col[e] = old_color
                else:
                    col[e] = old_color
                
                if current_violations == 0:
                    break
                
                if current_violations == current_colors:  # heuristic: stuck
                    no_improve += 1
                else:
                    no_improve = 0
                
                if no_improve > 500:
                    break
            
            if current_violations == 0:
                if current_colors < best_color_count:
                    best_color_count = current_colors
                    best_result = col[:]
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the seed coloring
    return seed_col
# EVOLVE-BLOCK-END
