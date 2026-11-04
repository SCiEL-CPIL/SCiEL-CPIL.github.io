"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _force_directed_optimal(N, w, h, n_iters=5000, lr_start=0.01, lr_end=0.00005,
                            repulse_strength=100.0, spring_strength=5.0, wall_strength=50.0):
    """
    Force-directed layout with contact graph optimization.
    Uses repulsive forces for overlaps, spring forces for near-contact pairs,
    and boundary forces to keep circles within the rectangle.
    Returns (x, y, r) arrays.
    """
    # Initialize with a grid layout
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        x[i] = (col + 0.5) * w / cols
        y[i] = (row + 0.5) * h / rows
        # Initial radius based on cell size
        cell_w = w / cols
        cell_h = h / rows
        r[i] = 0.4 * min(cell_w, cell_h)
    
    # Force-directed relaxation
    for it in range(n_iters):
        # Learning rate schedule (exponential decay)
        t = it / n_iters
        lr = lr_start * (lr_end / lr_start) ** t
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle interactions
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                
                if dist_sq < 1e-12:
                    # Perturb to avoid singularity
                    dx = np.random.randn() * 1e-6
                    dy = np.random.randn() * 1e-6
                    dist_sq = dx * dx + dy * dy
                
                dist = np.sqrt(dist_sq)
                
                if dist < min_dist:
                    # Overlapping: strong repulsive force
                    overlap = min_dist - dist
                    force = repulse_strength * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist < min_dist * 1.05:
                    # Near contact: spring force to maintain exact distance
                    diff = dist - min_dist
                    force = spring_strength * diff
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            # Left wall
            if x[i] < r[i]:
                penetration = r[i] - x[i]
                fx[i] += wall_strength * penetration
            # Right wall
            if x[i] > w - r[i]:
                penetration = x[i] - (w - r[i])
                fx[i] -= wall_strength * penetration
            # Bottom wall
            if y[i] < r[i]:
                penetration = r[i] - y[i]
                fy[i] += wall_strength * penetration
            # Top wall
            if y[i] > h - r[i]:
                penetration = y[i] - (h - r[i])
                fy[i] -= wall_strength * penetration
        
        # Update positions with velocity damping
        x += lr * fx
        y += lr * fy
        
        # Clamp to valid range
        for i in range(N):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])
    
    # After relaxation, iteratively grow radii while maintaining validity
    for grow_iter in range(200):
        # Try to grow all radii by a small factor
        growth_factor = 1.002
        r_new = r * growth_factor
        
        # Check if the packing is still valid with new radii
        valid = True
        for i in range(N):
            if x[i] < r_new[i] or x[i] > w - r_new[i]:
                valid = False
                break
            if y[i] < r_new[i] or y[i] > h - r_new[i]:
                valid = False
                break
        if not valid:
            break
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r_new[i] + r_new[j]
                if dist_sq < min_dist * min_dist - 1e-10:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            r = r_new
        else:
            break
    
    return x, y, r


def _validate_packing(x, y, r, w, h, tol=1e-6):
    """Check if packing is valid."""
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


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Use force-directed contact graph layout with multiple aspect ratios.
    """
    best_val = -1
    best_result = None
    
    # Try multiple aspect ratios
    # Perimeter = 2(w+h) = 4, so w+h = 2
    # w ranges from 0.5 to 1.5 (square is w=h=1)
    aspect_ratios = np.linspace(0.6, 1.4, 15)
    
    for w in aspect_ratios:
        h = 2.0 - w
        
        # Run force-directed packing
        x, y, r = _force_directed_optimal(N, w, h, n_iters=8000)
        
        # Validate
        if _validate_packing(x, y, r, w, h):
            val = np.sum(r)
            if val > best_val:
                best_val = val
                best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
