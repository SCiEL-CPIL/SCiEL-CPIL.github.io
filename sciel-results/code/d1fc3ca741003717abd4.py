"""Improved circle packing in unit square for N=26."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_valid(circles, tol=1e-8):
    """Check if all circles are inside unit square and non-overlapping."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol or r <= 0:
            return False
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2 - tol:
                return False
    return True

def _score(circles):
    """Sum of radii."""
    return np.sum(circles[:, 2])

def _grid_initial():
    """Create a grid-based initial configuration."""
    # Try a 5x5 grid, take 25, add one more
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([x, y, r])
    # Add a 26th circle in a corner with small radius
    cells.append([0.05, 0.05, 0.02])
    return np.array(cells[:N])

def _random_initial():
    """Random initial configuration with small radii."""
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.01, 0.05)
        circles.append([x, y, r])
    return np.array(circles)

def _optimize(circles, max_iters=5000, lr=0.01, tol=1e-10):
    """Optimize circle packing using gradient-based method."""
    circles = circles.copy().astype(float)
    n = len(circles)
    
    for it in range(max_iters):
        # Compute gradients
        grad = np.zeros_like(circles)
        
        # Boundary constraints: push circles away from walls
        for i in range(n):
            x, y, r = circles[i]
            # Left wall
            if x - r < 0:
                grad[i, 0] += (x - r) * 0  # x should increase
                grad[i, 2] += (x - r) * 0  # r should decrease
                # Force: increase x or decrease r
                grad[i, 0] += 1.0
                grad[i, 2] -= 1.0
            # Right wall
            if x + r > 1:
                grad[i, 0] -= 1.0
                grad[i, 2] -= 1.0
            # Bottom wall
            if y - r < 0:
                grad[i, 1] += 1.0
                grad[i, 2] -= 1.0
            # Top wall
            if y + r > 1:
                grad[i, 1] -= 1.0
                grad[i, 2] -= 1.0
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r1 + r2
                
                if dist < min_dist and dist > 1e-12:
                    # Push apart
                    overlap = min_dist - dist
                    ux = dx / dist
                    uy = dy / dist
                    # Move i in opposite direction, j in same direction
                    grad[i, 0] -= ux * 0.5
                    grad[i, 1] -= uy * 0.5
                    grad[j, 0] += ux * 0.5
                    grad[j, 1] += uy * 0.5
                    # Also reduce radii
                    grad[i, 2] -= 0.25
                    grad[j, 2] -= 0.25
                elif dist < min_dist:
                    # Very close, random push
                    grad[i, 0] += random.uniform(-0.1, 0.1)
                    grad[i, 1] += random.uniform(-0.1, 0.1)
                    grad[j, 0] += random.uniform(-0.1, 0.1)
                    grad[j, 1] += random.uniform(-0.1, 0.1)
        
        # Objective: maximize sum of radii
        for i in range(n):
            grad[i, 2] += 1.0  # Want to increase radius
        
        # Update
        for i in range(n):
            circles[i, 0] += lr * grad[i, 0]
            circles[i, 1] += lr * grad[i, 1]
            circles[i, 2] += lr * grad[i, 2]
            
            # Clamp
            circles[i, 0] = np.clip(circles[i, 0], 0, 1)
            circles[i, 1] = np.clip(circles[i, 1], 0, 1)
            circles[i, 2] = np.clip(circles[i, 2], 0.001, 0.5)
    
    return circles

def _local_search(circles, n_restarts=10):
    """Local search with random perturbations."""
    best = circles.copy()
    best_score = _score(best)
    
    for _ in range(n_restarts):
        current = best.copy()
        for _ in range(100):
            # Perturb a random circle
            i = random.randint(0, len(current)-1)
            current[i, 0] += random.uniform(-0.05, 0.05)
            current[i, 1] += random.uniform(-0.05, 0.05)
            current[i, 2] += random.uniform(-0.02, 0.02)
            
            # Clamp
            current[i, 0] = np.clip(current[i, 0], 0, 1)
            current[i, 1] = np.clip(current[i, 1], 0, 1)
            current[i, 2] = np.clip(current[i, 2], 0.001, 0.5)
            
            if _is_valid(current):
                s = _score(current)
                if s > best_score:
                    best_score = s
                    best = current.copy()
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    # Start with grid
    best = _grid_initial()
    best_score = _score(best)
    
    # Optimize the grid
    optimized = _optimize(best, max_iters=3000, lr=0.005)
    if _is_valid(optimized):
        s = _score(optimized)
        if s > best_score:
            best = optimized
            best_score = s
    
    # Local search
    if time.time() < deadline:
        ls_result = _local_search(best, n_restarts=5)
        if _is_valid(ls_result):
            s = _score(ls_result)
            if s > best_score:
                best = ls_result
                best_score = s
    
    # Try random starts
    if time.time() < deadline:
        for _ in range(5):
            if time.time() >= deadline:
                break
            rand = _random_initial()
            opt = _optimize(rand, max_iters=2000, lr=0.005)
            if _is_valid(opt):
                s = _score(opt)
                if s > best_score:
                    best = opt
                    best_score = s
    
    # Final check
    if not _is_valid(best):
        # Fallback to grid
        best = _grid_initial()
    
    return best
# EVOLVE-BLOCK-END
