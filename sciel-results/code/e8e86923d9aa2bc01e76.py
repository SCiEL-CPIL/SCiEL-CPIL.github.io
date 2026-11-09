"""Program for no-isosceles set task using optimized bitset-based greedy."""
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of grid points in [0, N)^2 with no three points forming an isosceles triangle.
    
    Approach: Optimized Greedy with bitset-based squared distance tracking.
    Uses multiple orderings including randomized restarts to find a larger independent set.
    """
    N_local = N
    candidates = [(x, y) for x in range(N_local) for y in range(N_local)]
    
    best_size = 0
    best_points = []
    
    # Helper function to run greedy with a specific order
    def greedy(order):
        xs = []
        ys = []
        dist_bitsets = []
        k = 0
        
        for (px, py) in order:
            # If k is 0, just add
            if k == 0:
                xs.append(px)
                ys.append(py)
                dist_bitsets.append(0)
                k = 1
                continue
                
            valid = True
            new_bitset = 0
            dists = [0] * k
            
            # Check against existing points
            for i in range(k):
                dx = px - xs[i]
                dy = py - ys[i]
                d2 = dx * dx + dy * dy
                dists[i] = d2
                
                # Check if this distance already exists from point i to another point in S
                # This covers the case where A_i is the apex of an isosceles triangle
                if (dist_bitsets[i] >> d2) & 1:
                    valid = False
                    break
                
                new_bitset |= (1 << d2)
            
            if not valid:
                continue
            
            # Check if all distances from P to existing points are distinct
            # This covers the case where P is the apex
            # If two distances are equal, bit_count will be less than k
            if new_bitset.bit_count() != k:
                continue
            
            # All checks passed, add the point
            xs.append(px)
            ys.append(py)
            dist_bitsets.append(new_bitset)
            
            # Update existing points' distance bitsets
            for i in range(k):
                dist_bitsets[i] |= (1 << dists[i])
            
            k += 1
        
        return list(zip(xs, ys))

    # Define orderings
    # 1. Lexicographic
    order_lex = sorted(candidates, key=lambda p: (p[0], p[1]))
    # 2. Reverse Lexicographic
    order_rev_lex = sorted(candidates, key=lambda p: (-p[0], -p[1]))
    # 3. Diagonal (x+y, x)
    order_diag = sorted(candidates, key=lambda p: (p[0] + p[1], p[0]))
    # 4. Reverse Diagonal
    order_rev_diag = sorted(candidates, key=lambda p: (-(p[0] + p[1]), -p[0]))
    
    fixed_orderings = [order_lex, order_rev_lex, order_diag, order_rev_diag]
    
    # Run fixed orderings
    for order in fixed_orderings:
        pts = greedy(order)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = pts
    
    # Run randomized restarts
    # Use a fixed seed for reproducibility if needed, but random is fine for search
    # We can run a few random shuffles
    for _ in range(10):
        random.shuffle(candidates)
        pts = greedy(candidates)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = pts
    
    return best_points
# EVOLVE-BLOCK-END
