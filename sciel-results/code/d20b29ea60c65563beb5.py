"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(circles, rect_w, rect_h):
    """Compute the sum of radii, returning 0 if invalid."""
    if len(circles) != N:
        return 0.0
    # Check perimeter
    if 2 * (rect_w + rect_h) > 4 + 1e-9:
        return 0.0
    # Check bounds
    for i in range(N):
        x, y, r = circles[i]
        if r < 0:
            return 0.0
        if x - r < -1e-9 or y - r < -1e-9 or x + r > rect_w + 1e-9 or y + r > rect_h + 1e-9:
            return 0.0
    # Check overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return 0.0
    return np.sum(circles[:, 2])

def _force_directed_optimize(circles, rect_w, rect_h, max_iters=5000, tol=1e-8):
    """
    Optimize the packing using force-directed relaxation.
    circles: (N, 3) array of [x, y, r]
    rect_w, rect_h: dimensions of the rectangle
    Returns optimized circles.
    """
    circles = circles.copy()
    N_local = len(circles)
    
    # Adaptive step size
    step = 0.05
    best_score = _compute_score(circles, rect_w, rect_h)
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros_like(circles)
        
        # 1. Repulsion between overlapping circles
        for i in range(N_local):
            for j in range(i + 1, N_local):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    dx = 0.01 * (np.random.rand() - 0.5)
                    dy = 0.01 * (np.random.rand() - 0.5)
                    dist = np.sqrt(dx * dx + dy * dy)
                
                overlap = circles[i][2] + circles[j][2] - dist
                if overlap > 0:
                    # Force proportional to overlap
                    f = overlap * 10.0
                    fx = f * dx / dist
                    fy = f * dy / dist
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
        
        # 2. Repulsion from walls
        for i in range(N_local):
            # Left wall
            if circles[i][0] - circles[i][2] < 0:
                forces[i][0] += (0 - (circles[i][0] - circles[i][2])) * 10.0
            # Right wall
            if circles[i][0] + circles[i][2] > rect_w:
                forces[i][0] -= ((circles[i][0] + circles[i][2]) - rect_w) * 10.0
            # Bottom wall
            if circles[i][1] - circles[i][2] < 0:
                forces[i][1] += (0 - (circles[i][1] - circles[i][2])) * 10.0
            # Top wall
            if circles[i][1] + circles[i][2] > rect_h:
                forces[i][1] -= ((circles[i][1] + circles[i][2]) - rect_h) * 10.0
        
        # 3. Attraction to maximize radius (simulated by expanding slightly)
        # We can simulate this by a small outward push or just rely on the relaxation
        # to fill space. A better way is to actually increase radii if there's slack.
        # For now, let's just relax positions.
        
        # Update positions
        circles[:, :2] += step * forces
        
        # Ensure circles stay within bounds (clamp)
        for i in range(N_local):
            circles[i][0] = np.clip(circles[i][0], circles[i][2], rect_w - circles[i][2])
            circles[i][1] = np.clip(circles[i][1], circles[i][2], rect_h - circles[i][2])
        
        # Check convergence
        move = np.max(np.abs(forces))
        if move < tol:
            break
        
        # Adaptive step size
        new_score = _compute_score(circles, rect_w, rect_h)
        if new_score > best_score:
            best_score = new_score
            step *= 1.05  # Increase step if improving
        else:
            step *= 0.9  # Decrease step if not improving
        
        # Cap step size
        step = np.clip(step, 1e-4, 0.5)
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0
    best_circles = None
    best_rect = (1.0, 1.0)
    
    # Define a set of aspect ratios to try
    # Perimeter = 2(w + h) = 4 => w + h = 2
    # Aspect ratio r = w/h => w = r*h, (r+1)h = 2 => h = 2/(r+1), w = 2r/(r+1)
    aspect_ratios = np.linspace(0.5, 2.0, 10)
    
    # Initial best: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    init_circles = np.array(cells[:N])
    # Normalize to fit in 1x1
    init_circles[:, 0] /= init_circles[:, 0].max() + init_circles[:, 2].max()
    init_circles[:, 1] /= init_circles[:, 1].max() + init_circles[:, 2].max()
    
    score = _compute_score(init_circles, 1.0, 1.0)
    if score > best_score:
        best_score = score
        best_circles = init_circles.copy()
        best_rect = (1.0, 1.0)
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 / (ar + 1.0)
        w = 2.0 * ar / (ar + 1.0)
        
        # Initialize circles in a grid pattern scaled to the rectangle
        side = int(np.ceil(np.sqrt(N)))
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i][0] = (col + 0.5) * w / side
            circles[i][1] = (row + 0.5) * h / side
            circles[i][2] = 0.4 * min(w, h) / side  # Initial radius
        
        # Run optimization
        optimized = _force_directed_optimize(circles, w, h, max_iters=2000)
        score = _compute_score(optimized, w, h)
        
        if score > best_score:
            best_score = score
            best_circles = optimized.copy()
            best_rect = (w, h)
    
    return best_circles
# EVOLVE-BLOCK-END
