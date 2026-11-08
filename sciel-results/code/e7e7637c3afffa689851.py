"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _check_valid(circles, tol=1e-7):
    """Check if circles are valid (disjoint, inside unit square)."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or r > 1:
            return False
        if x - r < -tol or x + r > 1 + tol:
            return False
        if y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _objective_and_grad(pos, radii):
    """
    Objective: -sum(radii)
    pos: (2N,) flattened positions
    radii: (N,) radii
    Returns (obj, grad_pos, grad_radii)
    """
    N = len(radii)
    pos2 = pos.reshape(N, 2)
    xs = pos2[:, 0]
    ys = pos2[:, 1]
    
    # Compute pairwise distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    # Constraints: dist[i,j] >= radii[i] + radii[j]
    # Boundary: xs[i] >= radii[i], 1-xs[i] >= radii[i], same for y
    # We use a penalty approach: minimize -sum(r) + penalty * violations
    
    # For gradient-based approach, we'll use a soft penalty
    # violation_ij = max(0, radii[i] + radii[j] - dist[i,j])
    
    # Actually, let's use a different approach:
    # We'll optimize positions given fixed radii, then grow radii.
    # Or use a combined optimization.
    
    # Let's use a penalty method:
    # f = -sum(r) + lambda * sum(violations^2)
    
    # For now, return simple objective
    obj = -np.sum(radii)
    grad_pos = np.zeros(2*N)
    grad_radii = -np.ones(N)
    
    return obj, grad_pos, grad_radii

def _relax_and_grow(circles, max_iter=500, growth_rate=0.002, shrink_rate=0.001):
    """
    Iteratively relax overlaps and grow radii.
    circles: (N, 3) array
    """
    circles = circles.copy()
    n = len(circles)
    
    for it in range(max_iter):
        # Check for overlaps and push apart
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dx = xj - xi
                dy = yj - yi
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = ri + rj
                if dist < min_dist - 1e-10:
                    if dist < 1e-10:
                        # Random push
                        angle = random.random() * 2 * np.pi
                        dx = np.cos(angle)
                        dy = np.sin(angle)
                        dist = 1.0
                    overlap = min_dist - dist
                    # Push each circle by half the overlap
                    ux = dx / dist
                    uy = dy / dist
                    push = overlap / 2.0
                    circles[i, 0] -= push * ux
                    circles[i, 1] -= push * uy
                    circles[j, 0] += push * ux
                    circles[j, 1] += push * uy
                    moved = True
        
        # Keep circles inside the square
        for i in range(n):
            r = circles[i, 2]
            circles[i, 0] = np.clip(circles[i, 0], r, 1 - r)
            circles[i, 1] = np.clip(circles[i, 1], r, 1 - r)
        
        # Try to grow radii
        if it % 10 == 0:
            # Check if we can grow
            can_grow = True
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi, ri = circles[i]
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < ri + rj + growth_rate:
                        can_grow = False
                        break
                if not can_grow:
                    break
            # Check boundary
            for i in range(n):
                r = circles[i, 2]
                if circles[i, 0] - r < growth_rate or 1 - circles[i, 0] - r < growth_rate:
                    can_grow = False
                    break
                if circles[i, 1] - r < growth_rate or 1 - circles[i, 1] - r < growth_rate:
                    can_grow = False
                    break
            
            if can_grow:
                circles[:, 2] += growth_rate
            else:
                # Shrink slightly to allow movement
                circles[:, 2] = np.maximum(circles[:, 2] - shrink_rate, 0.01)
    
    return circles

def _grid_init(n):
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
            count += 1
    return np.array(cells[:n])

def _hex_init(n):
    """Hexagonal initialization."""
    # Start with a hexagonal lattice
    circles = []
    r = 0.04
    spacing = 2 * r * 1.01
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    row = 0
    while len(circles) < n:
        for col in range(int(1/dx) + 1):
            if len(circles) >= n:
                break
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                circles.append([x, y, r])
        row += 1
    return np.array(circles[:n])

def _random_init(n, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    circles = []
    for i in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.03
        circles.append([x, y, r])
    return np.array(circles)

def _optimize_positions(circles, max_iter=1000, step_size=0.01):
    """
    Optimize positions to maximize minimum distance between circles.
    Uses a simple gradient ascent on the minimum pairwise distance.
    """
    circles = circles.copy()
    n = len(circles)
    
    for it in range(max_iter):
        pos = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Compute all pairwise distances
        dx = pos[:, 0, None] - pos[None, :, 0]
        dy = pos[:, 1, None] - pos[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_dist = dist[i, j]
        min_required = radii[i] + radii[j]
        
        if min_dist >= min_required - 1e-10:
            break
        
        # Push apart the closest pair
        if min_dist < 1e-10:
            angle = random.random() * 2 * np.pi
            dx = np.cos(angle)
            dy = np.sin(angle)
            min_dist = 1.0
        else:
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
        
        ux = dx / min_dist
        uy = dy / min_dist
        push = (min_required - min_dist) / 2.0 + 0.001
        
        pos[i, 0] -= push * ux
        pos[i, 1] -= push * uy
        pos[j, 0] += push * ux
        pos[j, 1] += push * uy
        
        # Clamp to boundaries
        for k in range(n):
            r = radii[k]
            pos[k, 0] = np.clip(pos[k, 0], r, 1 - r)
            pos[k, 1] = np.clip(pos[k, 1], r, 1 - r)
        
        circles[:, :2] = pos
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_circles = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [
        ("grid", _grid_init(N)),
        ("hex", _hex_init(N)),
        ("random1", _random_init(N, seed=42)),
        ("random2", _random_init(N, seed=123)),
        ("random3", _random_init(N, seed=999)),
    ]
    
    for init_name, init_circles in inits:
        if time.time() > deadline:
            break
        
        # Relax and grow
        circles = _relax_and_grow(init_circles, max_iter=300, growth_rate=0.0015, shrink_rate=0.0005)
        
        # Optimize positions
        circles = _optimize_positions(circles, max_iter=500)
        
        # Check validity
        if _check_valid(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
        
        # Additional relaxation
        if time.time() < deadline:
            circles2 = _relax_and_grow(circles, max_iter=200, growth_rate=0.001, shrink_rate=0.0003)
            circles2 = _optimize_positions(circles2, max_iter=300)
            if _check_valid(circles2):
                score2 = np.sum(circles2[:, 2])
                if score2 > best_score:
                    best_score = score2
                    best_circles = circles2.copy()
    
    if best_circles is None:
        # Fallback to grid
        best_circles = _grid_init(N)
    
    return best_circles
# EVOLVE-BLOCK-END
