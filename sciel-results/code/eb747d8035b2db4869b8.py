"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def _relax_circles(W, H, n_circles, n_iter=500, lr=0.05):
    """Relax circle positions and radii to maximize sum of radii in W x H rectangle."""
    # Initialize with grid layout
    cols = int(np.ceil(np.sqrt(n_circles * W / H))) if W > 0 else int(np.ceil(np.sqrt(n_circles)))
    rows = int(np.ceil(n_circles / cols))
    
    xs = np.zeros(n_circles)
    ys = np.zeros(n_circles)
    r = np.zeros(n_circles)
    
    for i in range(n_circles):
        col = i % cols
        row = i // cols
        xs[i] = (col + 0.5) * W / cols
        ys[i] = (row + 0.5) * H / rows
        r[i] = min(W / cols, H / rows) * 0.45
    
    for it in range(n_iter):
        # Compute forces
        fx = np.zeros(n_circles)
        fy = np.zeros(n_circles)
        
        # Repulsion between overlapping circles
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = (min_dist - dist) / min_dist
                    force = overlap * 0.5
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Wall constraints: push circles inward if they extend beyond walls
        for i in range(n_circles):
            # Left wall
            if xs[i] - r[i] < 0:
                fx[i] += (r[i] - xs[i]) * 0.5
                # Try to shrink
                r[i] -= (r[i] - xs[i]) * 0.1
            # Right wall
            if xs[i] + r[i] > W:
                fx[i] -= (xs[i] + r[i] - W) * 0.5
                r[i] -= (xs[i] + r[i] - W) * 0.1
            # Bottom wall
            if ys[i] - r[i] < 0:
                fy[i] += (r[i] - ys[i]) * 0.5
                r[i] -= (r[i] - ys[i]) * 0.1
            # Top wall
            if ys[i] + r[i] > H:
                fy[i] -= (ys[i] + r[i] - H) * 0.5
                r[i] -= (ys[i] + r[i] - H) * 0.1
        
        # Update positions
        xs += lr * fx
        ys += lr * fy
        
        # Try to grow radii slightly
        r += 0.001
        
        # Ensure minimum radius
        r = np.maximum(r, 0.01)
    
    # Final validation and adjustment
    valid = True
    for i in range(n_circles):
        if xs[i] - r[i] < -1e-6 or xs[i] + r[i] > W + 1e-6:
            valid = False
        if ys[i] - r[i] < -1e-6 or ys[i] + r[i] > H + 1e-6:
            valid = False
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-6:
                valid = False
    
    if valid:
        return np.column_stack([xs, ys, r])
    else:
        return None


def _try_aspect_ratio(W, H, n_circles, time_budget):
    """Try packing circles in a W x H rectangle, return best result or None."""
    start = time.time()
    result = _relax_circles(W, H, n_circles, n_iter=300, lr=0.05)
    elapsed = time.time() - start
    if result is not None and elapsed < time_budget:
        return result
    return None


def construct():
    deadline = time.time() + 120  # 120 second limit
    
    # Try various aspect ratios for perimeter 4: 2*(W+H) = 4, so W+H = 2
    # H = 2 - W, W in (0, 2)
    best_result = None
    best_sum = 0
    
    # Start with square-ish and elongated ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.6, 1.67, 0.5, 2.0, 0.4, 2.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
        
        if ar >= 1:
            W = 2 * ar / (1 + ar)
            H = 2 / (1 + ar)
        else:
            W = 2 / (1 + 1/ar)
            H = 2 * (1/ar) / (1 + 1/ar)
        
        # Ensure perimeter is exactly 4
        W = 2 * ar / (1 + ar) if ar >= 1 else 2 / (1 + 1/ar)
        H = 2 / (1 + ar) if ar >= 1 else 2 * (1/ar) / (1 + 1/ar)
        
        # Simpler: W + H = 2
        W = 2 * ar / (1 + ar)
        H = 2 / (1 + ar)
        
        time_budget = max(0, deadline - time.time() - 5)
        if time_budget < 2:
            break
        
        result = _try_aspect_ratio(W, H, N, time_budget)
        if result is not None:
            total_r = np.sum(result[:, 2])
            if total_r > best_sum:
                best_sum = total_r
                best_result = result.copy()
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
