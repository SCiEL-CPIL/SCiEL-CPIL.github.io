"""Circle packing in perimeter-4 rectangle - force-directed contact graph approach."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def _validate_packing(x, y, r, w, h, tol=1e-7):
    """Check if packing is valid: all circles inside rectangle, no overlaps."""
    for i in range(len(x)):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True


def _force_directed_packing(w, h, seed=0):
    """
    Force-directed layout for N circles in rectangle w x h.
    Uses contact graph with spring forces and wall repulsion.
    Returns (x, y, r) arrays.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with grid layout
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
        r[i] = min(w / cols, h / rows) * 0.4
    
    # Add small random perturbation
    x += rng.randn(N) * 1e-4
    y += rng.randn(N) * 1e-4
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    n_iters = 8000
    lr_start = 0.003
    lr_end = 0.00005
    
    for it in range(n_iters):
        t = it / n_iters
        # Exponential decay with floor
        lr = lr_end + (lr_start - lr_end) * np.exp(-5.0 * t)
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                
                if dist < 1e-12:
                    dx = rng.randn() * 1e-6
                    dy = rng.randn() * 1e-6
                    dist = np.sqrt(dx * dx + dy * dy)
                
                if dist < min_dist:
                    # Overlap: strong repulsion
                    overlap = min_dist - dist
                    force = 100.0 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * 1.05:
                    # Near contact: spring to maintain exact distance
                    diff = dist - min_dist
                    force = 5.0 * diff
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Wall forces
        for i in range(N):
            if x[i] < r[i]:
                pen = r[i] - x[i]
                fx[i] += 50.0 * pen
            if x[i] > w - r[i]:
                pen = x[i] - (w - r[i])
                fx[i] -= 50.0 * pen
            if y[i] < r[i]:
                pen = r[i] - y[i]
                fy[i] += 50.0 * pen
            if y[i] > h - r[i]:
                pen = y[i] - (h - r[i])
                fy[i] -= 50.0 * pen
        
        # Update positions
        x += lr * fx
        y += lr * fy
        
        # Clamp
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    # Grow radii iteratively
    for grow_iter in range(200):
        r_new = r * 1.003
        valid = True
        for i in range(N):
            if x[i] < r_new[i] or x[i] > w - r_new[i] or y[i] < r_new[i] or y[i] > h - r_new[i]:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r_new[i] + r_new[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            r = r_new
        else:
            break
    
    return x, y, r


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios (w + h = 2)
    # Symmetric: only need w in [0.5, 1.5]
    aspect_ratios = np.linspace(0.5, 1.5, 21)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Multiple restarts
        for seed in range(5):
            x, y, r = _force_directed_packing(w, h, seed=seed)
            
            if _validate_packing(x, y, r, w, h):
                val = np.sum(r)
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid in square
    w = 1.0
    h = 1.0
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(N):
        col = i % side
        row = i // side
        cx = (col + 0.5) * w / side
        cy = (row + 0.5) * h / side
        cr = 0.5 / side - 0.01
        cells.append([cx, cy, cr])
    return np.array(cells)

# EVOLVE-BLOCK-END
