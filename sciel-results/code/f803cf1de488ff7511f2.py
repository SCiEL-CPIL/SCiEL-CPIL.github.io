"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 21

# EVOLVE-BLOCK-START

def _pack_circles(n, w, h, time_limit=10.0, seed=42):
    """
    Pack n circles in a w x h rectangle.
    Returns (success, radii, centers) where centers is (n,2) array.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # Initial placement: grid-like with some jitter
    cols = int(np.ceil(np.sqrt(n * w / h))) if h > 0 else n
    rows = int(np.ceil(n / cols))
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place in grid
    for i in range(n):
        r = i // cols
        c = i % cols
        centers[i, 0] = (c + 0.5) * w / cols
        centers[i, 1] = (r + 0.5) * h / rows
    
    # Initial radius: min of distances to walls and half min distance to others
    for i in range(n):
        min_dist = min(
            centers[i, 0], w - centers[i, 0],
            centers[i, 1], h - centers[i, 1]
        )
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, d / 2)
        radii[i] = min_dist * 0.9
    
    # Ensure non-overlap initially
    for it in range(50):
        valid = True
        for i in range(n):
            if radii[i] < 1e-10:
                valid = False
                break
            # Check walls
            if centers[i, 0] - radii[i] < -1e-9 or centers[i, 0] + radii[i] > w + 1e-9:
                valid = False
                break
            if centers[i, 1] - radii[i] < -1e-9 or centers[i, 1] + radii[i] > h + 1e-9:
                valid = False
                break
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        # Scale down radii
        radii *= 0.95
    
    if not valid:
        return False, radii, centers
    
    # Local optimization: try to increase radii
    # Use a simple iterative approach
    for outer in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Try to increase each radius slightly
        improved = False
        for i in range(n):
            # Max possible radius for circle i given current positions of others
            max_r = min(
                centers[i, 0], w - centers[i, 0],
                centers[i, 1], h - centers[i, 1]
            )
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, (d - radii[j]))
            
            if max_r > radii[i] + 1e-6:
                # Try increasing radius i
                new_r = (radii[i] + max_r) / 2
                # Check if this is feasible (it should be by construction)
                radii[i] = new_r
                improved = True
        
        if not improved:
            break
    
    # Try to move circles to create more space
    for outer in range(100):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            # Try small perturbations
            for dx in [-0.01, 0.01]:
                for dy in [-0.01, 0.01]:
                    old_c = centers[i].copy()
                    old_r = radii[i]
                    centers[i, 0] += dx
                    centers[i, 1] += dy
                    
                    # Check feasibility
                    feasible = True
                    for j in range(n):
                        if i == j:
                            continue
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < radii[i] + radii[j] - 1e-9:
                            feasible = False
                            break
                    if feasible:
                        if centers[i, 0] - radii[i] < -1e-9 or centers[i, 0] + radii[i] > w + 1e-9:
                            feasible = False
                        if centers[i, 1] - radii[i] < -1e-9 or centers[i, 1] + radii[i] > h + 1e-9:
                            feasible = False
                    
                    if feasible:
                        # Try to increase radius
                        max_r = min(
                            centers[i, 0], w - centers[i, 0],
                            centers[i, 1], h - centers[i, 1]
                        )
                        for j in range(n):
                            if i != j:
                                d = np.linalg.norm(centers[i] - centers[j])
                                max_r = min(max_r, (d - radii[j]))
                        if max_r > radii[i]:
                            radii[i] = max_r * 0.99
                            improved = True
                            break
                    else:
                        centers[i] = old_c
                        radii[i] = old_r
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return True, radii, centers


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_sum = 0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter 4: 2*(w+h) = 4, so w+h = 2
    aspect_ratios = np.linspace(0.3, 3.0, 30)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = ar / (1 + ar) * 2  # w + h = 2
        h = 2 - w
        
        success, radii, centers = _pack_circles(N, w, h, time_limit=3.0, seed=42)
        
        if success:
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_result = (w, h, radii.copy(), centers.copy())
    
    if best_result is None:
        # Fallback: simple grid
        w, h = 1.0, 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result
    
    w, h, radii, centers = best_result
    
    # Scale to fit perimeter 4
    # Current: 2*(w+h) = 4, so it's already correct
    # Build result: [x, y, r]
    result = np.column_stack([centers[:, 0], centers[:, 1], radii])
    
    return result

# EVOLVE-BLOCK-END
