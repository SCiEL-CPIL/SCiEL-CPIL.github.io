"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Use force-directed layout with contact constraints.
    """
    # Rectangle with perimeter 4: let width = w, height = h, 2(w+h) = 4 => w+h = 2
    # We'll optimize over w and h as well, or fix aspect ratio and scale.
    
    # Force-directed approach:
    # - Variables: x_i, y_i, r_i for each circle i
    # - Forces:
    #   - Boundary repulsion when circle touches/exceeds wall
    #   - Non-overlap repulsion when circles overlap
    #   - Attraction to increase radii (maximize sum of radii)
    
    best_val = -1
    best_result = None
    
    # Try several aspect ratios
    for w in np.linspace(0.6, 1.4, 9):
        h = 2.0 - w
        
        # Initialize with grid layout
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                x[count] = (i + 0.5) * w / cols
                y[count] = (j + 0.5) * h / rows
                r[count] = min(w / cols, h / rows) / 2 * 0.8
                count += 1
        
        # Force-directed relaxation
        num_iters = 500
        step_size = 0.01
        damping = 0.95
        
        for it in range(num_iters):
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Boundary forces: repel from walls
            for i in range(N):
                # Left wall
                if x[i] - r[i] < 0:
                    fx[i] += 10.0 * (-(x[i] - r[i]))
                # Right wall
                if x[i] + r[i] > w:
                    fx[i] -= 10.0 * (x[i] + r[i] - w)
                # Bottom wall
                if y[i] - r[i] < 0:
                    fy[i] += 10.0 * (-(y[i] - r[i]))
                # Top wall
                if y[i] + r[i] > h:
                    fy[i] -= 10.0 * (y[i] + r[i] - h)
            
            # Non-overlap forces: repel overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        # Repulsive force proportional to overlap
                        overlap = min_dist - dist
                        force = 10.0 * overlap
                        fx[i] += force * dx / dist
                        fy[i] += force * dy / dist
                        fx[j] -= force * dx / dist
                        fy[j] -= force * dy / dist
            
            # Attraction to increase radii
            # We want to maximize sum(r), so add a small positive force on r
            fr[:] = 0.1
            
            # Update positions with damping
            x += step_size * fx * damping
            y += step_size * fy * damping
            r += step_size * fr * damping
            
            # Enforce bounds
            r = np.maximum(r, 0.0)
            r = np.minimum(r, min(w, h) / 2)
            x = np.maximum(x, r)
            x = np.minimum(x, w - r)
            y = np.maximum(y, r)
            y = np.minimum(y, h - r)
            
            # If no more overlap, we can stop early
            has_overlap = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < (r[i] + r[j]) ** 2 - 1e-10:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            if not has_overlap:
                break
        
        # Verify constraints
        valid = True
        for i in range(N):
            if r[i] < -1e-6:
                valid = False
                break
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                valid = False
                break
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < (r[i] + r[j]) ** 2 - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
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
