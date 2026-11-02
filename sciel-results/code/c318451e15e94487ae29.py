import numpy as np
import time

def construct():
    N = 26
    deadline = time.time() + 100  # Safety margin for 150s limit
    
    # 1. Initialization: Hexagonal packing
    # Hexagonal packing is denser than square.
    # For N=26, we can fit roughly 5x5 or 6x4 grid with adjustments.
    # Let's start with a grid and then relax.
    # A simple robust init: Grid of 5x6 (30 cells) take 26? Or 6x5?
    # Let's use a 5x5 grid (25 circles) + 1 extra? No, N is fixed.
    # Let's use a 6x5 grid (30 spots) and pick 26?
    # Better: Start with a 5x5 grid (25 circles) and add 1? 
    # Actually, let's just use a 6x5 grid layout, pick the 26 best positions (centered).
    
    cols, rows = 6, 5
    xs = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    
    # Create a grid of potential centers
    centers = []
    for x in xs:
        for y in ys:
            centers.append([x, y])
    centers = np.array(centers)
    
    # We need N=26 circles. The grid has 30.
    # Let's select the 26 most central ones to maximize min distance initially.
    # Distance from center (0.5, 0.5)
    d2c = np.sum((centers - 0.5)**2, axis=1)
    idx = np.argsort(d2c)[:N]
    pos = centers[idx].copy()
    
    # Initial radii: 
    # The minimum distance between neighbors in this grid is 1/6 or 1/5.
    # Let's start with a small radius to ensure validity.
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            min_dist = min(min_dist, d)
    # Also check boundary
    for i in range(N):
        min_dist = min(min_dist, pos[i][0], 1-pos[i][0], pos[i][1], 1-pos[i][1])
        
    r = min_dist * 0.95
    radii = np.full(N, r)
    
    def is_valid(pos, radii):
        for i in range(N):
            # Boundary
            if pos[i][0] - radii[i] < -1e-6 or pos[i][0] + radii[i] > 1 + 1e-6:
                return False
            if pos[i][1] - radii[i] < -1e-6 or pos[i][1] + radii[i] > 1 + 1e-6:
                return False
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j] - 1e-6:
                    return False
        return True

    def score(pos, radii):
        if not is_valid(pos, radii):
            return -np.inf
        return np.sum(radii)
    
    current_score = score(pos, radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = current_score
    
    # 2. Local Search / Hill Climbing
    # Strategy: Try to increase radii. If invalid, try to move positions to resolve overlap.
    # Simpler: Simulated Annealing on positions and radii.
    # But SA is slow. Let's try a simple "push" method.
    
    # We will iterate:
    # 1. Try to increase all radii by a small epsilon.
    # 2. If invalid, calculate overlap forces and move centers.
    
    step_size = 0.01
    iterations = 0
    max_iterations = 2000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Try to increase radii
        new_radii = radii * (1 + 0.001)
        
        if is_valid(pos, new_radii):
            radii = new_radii
            s = score(pos, radii)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                best_radii = radii.copy()
            continue
        
        # If invalid, we need to move positions to make room for larger radii (or current radii)
        # Let's try to resolve overlaps for the *current* radii first to ensure stability,
        # then try to grow.
        # Actually, let's just do gradient ascent on a penalty function.
        
        # Calculate penalty and gradient
        penalty = 0.0
        grad = np.zeros_like(pos)
        
        # Overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-8:
                    d = 1e-8
                    d_vec = np.array([1e-8, 0])
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    penalty += overlap**2
                    # Gradient wrt pos[i]
                    force = 2 * overlap * (d_vec / d)
                    grad[i] += force
                    grad[j] -= force
        
        # Boundary penalty
        for i in range(N):
            # Left
            pen = radii[i] - pos[i][0]
            if pen > 0:
                penalty += pen**2
                grad[i][0] += 2 * pen
            # Right
            pen = pos[i][0] + radii[i] - 1
            if pen > 0:
                penalty += pen**2
                grad[i][0] -= 2 * pen
            # Bottom
            pen = radii[i] - pos[i][1]
            if pen > 0:
                penalty += pen**2
                grad[i][1] += 2 * pen
            # Top
            pen = pos[i][1] + radii[i] - 1
            if pen > 0:
                penalty += pen**2
                grad[i][1] -= 2 * pen
        
        # Move in direction of gradient descent (reduce penalty)
        # Use a small step
        pos = pos - 0.01 * grad
        
        # Clamp positions to [0,1]
        pos = np.clip(pos, 0, 1)
        
        # Check if we improved
        s = score(pos, radii)
        if s > best_score:
            best_score = s
            best_pos = pos.copy()
            best_radii = radii.copy()
            
        # If penalty is low, try growing radii again
        if penalty < 1e-6:
             # Try to grow
             new_radii = radii * (1 + 0.0005)
             if is_valid(pos, new_radii):
                 radii = new_radii
             else:
                 # If can't grow, maybe shrink a bit to allow movement?
                 # Or just continue moving
                 pass

    # Final check
    if not is_valid(best_pos, best_radii):
        # Fallback to a very safe small radius
        min_d = 0.1
        for i in range(N):
            for j in range(i+1, N):
                min_d = min(min_d, np.linalg.norm(best_pos[i] - best_pos[j]))
        for i in range(N):
            min_d = min(min_d, best_pos[i][0], 1-best_pos[i][0], best_pos[i][1], 1-best_pos[i][1])
        best_radii = np.full(N, min_d * 0.5)
        # Re-verify
        if not is_valid(best_pos, best_radii):
            # Absolute fallback: tiny circles
            best_radii = np.full(N, 0.01)

    result = np.column_stack((best_pos, best_radii))
    return result
