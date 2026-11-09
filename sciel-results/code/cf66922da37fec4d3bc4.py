# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n

    def compute_area(x_vec):
        # x_vec is shape (n,)
        # L is shape (n, M)
        L = x_vec[:, None] + a[:, None] * y[None, :]
        # Sort L along axis 0
        # np.sort is O(M * n log n)
        L_sorted = np.sort(L, axis=0)
        # Diffs
        # d_k = L_sorted[k+1] - L_sorted[k]
        d = np.diff(L_sorted, axis=0) # shape (n-1, M)
        # union_len = w + sum(min(d, w))
        # min(d, w) can be computed as np.minimum(d, w)
        # Sum along axis 0
        union_lens = w + np.sum(np.minimum(d, w), axis=0)
        area = np.mean(union_lens)
        return area

    # Initial guess based on Keich's construction
    # The seed program used a binary expansion based approach.
    # Let's stick to a robust initialization and then optimize.
    # The seed program's initialization:
    k = int(np.log2(n))
    positions = []
    j = np.arange(1, k + 1)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Try a few other initializations to see if we can start better
    inits = [
        x,
        np.zeros(n),
        -a * 0.5,
        -a * 0.25,
        -a * 0.75,
        np.linspace(-0.5, 0.5, n),
    ]
    
    best_x = None
    best_area = float('inf')
    
    for init in inits:
        area = compute_area(init)
        if area < best_area:
            best_area = area
            best_x = init.copy()
            
    x = best_x.copy()
    current_area = best_area

    # Coordinate Descent with Adaptive Line Search
    # The area function is convex but non-smooth.
    # We use a "shrink-wrap" line search:
    # 1. Try a step in a direction (e.g., coordinate i +/-).
    # 2. If it improves, try to go further in that direction.
    # 3. If it fails, shrink the step.
    
    # We will iterate over coordinates.
    # For each coordinate, we will try to minimize the area.
    
    max_iterations = 2000
    tol = 1e-12
    
    for it in range(max_iterations):
        if time.time() > deadline - 30.0:
            break
            
        improved = False
        
        # Shuffle coordinates to avoid bias?
        # Or just go 0..n-1.
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 30.0:
                break
                
            # Current x[i]
            x_i = x[i]
            
            # We want to find the best x[i] in a local range.
            # Let's try a ternary search in a window around x_i.
            # How wide?
            # The function is piecewise linear.
            # The "kinks" are where the ordering of L_j changes or where min(d, w) switches.
            # A reasonable window is probably related to the scale of the problem.
            # Let's start with a window of size 1.0.
            # If the optimum is outside, we expand.
            
            # Actually, let's use a more robust approach:
            # Try a few discrete steps first to get a direction, then line search.
            
            # 1. Discrete search
            best_delta = 0.0
            best_val = current_area
            
            # Try steps
            # Scale of steps:
            # Initial scale 0.1, then 0.01, then 0.001...
            # But this is slow if we do it for every coordinate every iteration.
            
            # Alternative: Ternary search on [x_i - 0.5, x_i + 0.5]
            # This is O(log(1/eps)) evaluations.
            # 40 iterations of ternary search gives precision 0.5 * (2/3)^40 ~ 1e-8.
            # This is likely sufficient.
            
            lo = x_i - 0.5
            hi = x_i + 0.5
            
            # Ternary search
            # We need to be careful: if the function is flat, ternary search might stall or move randomly.
            # If area1 == area2, we can shrink the interval.
            
            for _ in range(40):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                
                x[i] = m1
                area1 = compute_area(x)
                
                x[i] = m2
                area2 = compute_area(x)
                
                if area1 < area2:
                    hi = m2
                elif area2 < area1:
                    lo = m1
                else:
                    # Flat region? Shrink
                    lo = m1
                    hi = m2
                    
                if time.time() > deadline - 30.0:
                    break
            
            # The best is roughly in the middle of [lo, hi]
            x_new = (lo + hi) / 2.0
            
            # Evaluate the new point
            x[i] = x_new
            area_new = compute_area(x)
            
            # Also check the original point?
            # If area_new < current_area, accept.
            # If area_new >= current_area, revert?
            # Actually, ternary search should find a local min.
            # But if the global min is outside [lo, hi], we might have moved away.
            # So we should check if x_new is better than x_i.
            
            if area_new < current_area - 1e-12:
                current_area = area_new
                improved = True
            else:
                # Revert
                x[i] = x_i
                # current_area remains unchanged (it was computed with x_i)
                # Wait, we overwrote x[i] with x_new.
                # We need to restore x[i] to x_i.
                # And current_area is the area at x_i.
                
        if not improved:
            # If no coordinate improved, we might be at a local minimum (or global).
            # Try to expand the search range?
            # Or just break.
            # Given the convexity, if no coordinate improves, we are likely at the global minimum
            # (for coordinate descent on convex functions, this is a strong signal).
            break
            
    return x

# EVOLVE-BLOCK-END
