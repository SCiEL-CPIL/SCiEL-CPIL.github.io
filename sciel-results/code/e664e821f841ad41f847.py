import numpy as np
import time
import random
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _score(centers, sides, tol=1e-7):
    """Score = sum of sides. Higher is better."""
    if not _is_valid(centers, sides, tol):
        return -np.inf
    return np.sum(sides)

def _local_search(centers, sides, tol=1e-7, max_iter=5000, temp=0.01):
    """
    Perform a local search (simulated annealing) to maximize the sum of sides.
    Moves: 
    1. Move a square slightly.
    2. Change the size of a square slightly (and adjust position to stay in bounds/valid).
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _score(best_centers, best_sides, tol)
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_score = best_score
    
    for i in range(max_iter):
        # Decay temperature
        temp_curr = temp * (0.999 ** i)
        
        # Pick a random square to modify
        idx = np.random.randint(n)
        
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        
        move_type = np.random.random()
        
        if move_type < 0.7:
            # Move the square
            dx = np.random.normal(0, 0.01)
            dy = np.random.normal(0, 0.01)
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
        else:
            # Change size
            # Keep center, expand/contract
            factor = np.random.normal(1.0, 0.05)
            new_sides[idx] *= factor
            if new_sides[idx] <= 0:
                new_sides[idx] = 1e-4
            
            # Adjust center to stay in bounds if possible
            s_new = new_sides[idx]
            cx, cy = new_centers[idx]
            
            # Clamp center to ensure square stays in [0, 1]
            min_cx = s_new / 2
            max_cx = 1 - s_new / 2
            min_cy = s_new / 2
            max_cy = 1 - s_new / 2
            
            if min_cx > max_cx or min_cy > max_cy:
                # Square too big
                continue
                
            new_centers[idx, 0] = np.clip(cx, min_cx, max_cx)
            new_centers[idx, 1] = np.clip(cy, min_cy, max_cy)
            
        new_score = _score(new_centers, new_sides, tol)
        
        if new_score > current_score:
            current_centers = new_centers
            current_sides = new_sides
            current_score = new_score
            
            if current_score > best_score:
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                best_score = current_score
        else:
            # Accept worse solution with probability exp((new-old)/temp)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / (temp_curr + 1e-9))
                if np.random.random() < prob:
                    current_centers = new_centers
                    current_sides = new_sides
                    current_score = new_score
    
    return best_centers, best_sides

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600  # seconds
    
    best_centers = None
    best_sides = None
    best_value = -np.inf
    
    # Configuration 1: 2 large (0.5) + 8 small (0.25)
    # Sum = 2*0.5 + 8*0.25 = 3.0
    c1 = np.zeros((10, 2))
    s1 = np.zeros(10)
    
    # Bottom row: 2 squares of side 0.5
    s1[0] = 0.5
    c1[0] = [0.25, 0.25]
    s1[1] = 0.5
    c1[1] = [0.75, 0.25]
    
    # Top strip: 8 squares of side 0.25
    # 4 columns, 2 rows
    # y ranges from 0.5 to 1.0
    # x ranges from 0 to 1.0
    idx = 2
    for row in range(2):
        for col in range(4):
            cx = col * 0.25 + 0.125
            cy = 0.5 + row * 0.25 + 0.125
            c1[idx] = [cx, cy]
            s1[idx] = 0.25
            idx += 1
            
    if _is_valid(c1, s1, tol):
        val1 = np.sum(s1)
        if val1 > best_value:
            best_value = val1
            best_centers = c1.copy()
            best_sides = s1.copy()
            
    # Configuration 2: 1 large (0.5) + 4 medium (0.25) + 5 small (0.25)
    # Sum = 0.5 + 4*0.25 + 5*0.25 = 2.75
    # This is worse than 3.0, but good for diversity
    
    # Configuration 3: Random grid
    # 3x3 grid of 1/3 squares, plus 1 more?
    # Let's just run local search from Config 1
    
    if time.time() - start_time < deadline - 10:
        # Run local search from the best configuration
        centers_ls, sides_ls = _local_search(best_centers, best_sides, tol, max_iter=20000, temp=0.05)
        val_ls = _score(centers_ls, sides_ls, tol)
        if val_ls > best_value:
            best_value = val_ls
            best_centers = centers_ls
            best_sides = sides_ls
            
    # Fallback
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
