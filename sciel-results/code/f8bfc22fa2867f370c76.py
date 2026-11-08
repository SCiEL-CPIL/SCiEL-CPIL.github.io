"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _project_constraints(pos, radii, n_iter=50):
    """Project positions to satisfy non-overlap and boundary constraints."""
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(n_iter):
        # Project to boundaries
        for i in range(n):
            x, y = pos[i]
            r = radii[i]
            # Clamp to [r, 1-r]
            if x < r:
                x = r
            elif x > 1 - r:
                x = 1 - r
            if y < r:
                y = r
            elif y > 1 - r:
                y = 1 - r
            pos[i] = [x, y]
        
        # Project to non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-10:
                    if dist < 1e-10:
                        # Circles at same position, push apart arbitrarily
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    # Push apart
                    push = (min_dist - dist) / 2.0
                    nx, ny = dx / dist, dy / dist
                    pos[i][0] -= push * nx
                    pos[i][1] -= push * ny
                    pos[j][0] += push * nx
                    pos[j][1] += push * ny
                    # Re-clamp after push
                    for k in [i, j]:
                        x, y = pos[k]
                        r = radii[k]
                        if x < r:
                            x = r
                        elif x > 1 - r:
                            x = 1 - r
                        if y < r:
                            y = r
                        elif y > 1 - r:
                            y = 1 - r
                        pos[k] = [x, y]
    
    return np.array(pos)


def _check_validity(pos, radii):
    """Check if packing is valid."""
    n = len(pos)
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x < -1e-9 or x > 1 + 1e-9 or y < -1e-9 or y > 1 + 1e-9:
            return False
        if r < -1e-9:
            return False
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Initialize with grid
    side = int(np.ceil(np.sqrt(N)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side] 
                    for i in range(side) for j in range(side)][:N])
    radii = np.full(N, 0.5 / side - 1e-3)
    
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # PBD iterations with adaptive radius growth
    n_iterations = 2000
    for it in range(n_iterations):
        if time.time() - start_time > deadline:
            break
        
        # Try to grow radii slightly
        growth_rate = 0.001 * (1 - it / n_iterations) + 0.0001
        radii_new = radii + growth_rate
        
        # Check if we can grow without violating boundaries
        feasible = True
        for i in range(N):
            if radii_new[i] > 0.5:
                radii_new[i] = 0.5
            if radii_new[i] <= 0:
                feasible = False
                break
        
        if not feasible:
            break
        
        # Project to satisfy constraints with new radii
        pos_new = _project_constraints(pos, radii_new, n_iter=30)
        
        # Check validity
        if _check_validity(pos_new, radii_new):
            pos = pos_new
            radii = radii_new
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
        else:
            # Try smaller growth
            for attempt in range(5):
                growth_rate_small = growth_rate / (2 ** (attempt + 1))
                radii_try = radii + growth_rate_small
                pos_try = _project_constraints(pos, radii_try, n_iter=30)
                if _check_validity(pos_try, radii_try):
                    pos = pos_try
                    radii = radii_try
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
                        best_radii = radii.copy()
                    break
    
    # Final polish: try to improve with more iterations
    if time.time() - start_time < deadline - 10:
        for _ in range(100):
            if time.time() - start_time > deadline:
                break
            # Small random perturbations
            for i in range(N):
                if np.random.random() < 0.3:
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    pos[i][0] += dx
                    pos[i][1] += dy
            
            pos_polished = _project_constraints(pos, radii, n_iter=50)
            if _check_validity(pos_polished, radii):
                pos = pos_polished
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()
    
    # Return best result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_pos[i][0], best_pos[i][1], best_radii[i]]
    
    return result
# EVOLVE-BLOCK-END
