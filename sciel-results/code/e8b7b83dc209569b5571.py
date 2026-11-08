"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(circles):
    """Compute sum of radii if all circles are valid, else -1."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0 or x < -r or x > 1 + r or y < -r or y > 1 + r:
            return -1.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-12:
                return -1.0
    return np.sum(circles[:, 2])

def _pbd_step(circles, dt, iterations=20):
    """One PBD step: project onto constraints."""
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for _ in range(iterations):
        # Wall constraints
        for i in range(n):
            x, y = pos[i]
            r = rad[i]
            if x < r:
                pos[i][0] = r
            elif x > 1 - r:
                pos[i][0] = 1 - r
            if y < r:
                pos[i][1] = r
            elif y > 1 - r:
                pos[i][1] = 1 - r
        
        # Circle-circle constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    nx, ny = dx / dist, dy / dist
                    pos[i][0] += 0.5 * overlap * nx
                    pos[i][1] += 0.5 * overlap * ny
                    pos[j][0] -= 0.5 * overlap * nx
                    pos[j][1] -= 0.5 * overlap * ny
                elif dist <= 1e-10:
                    # Random separation
                    angle = np.random.uniform(0, 2 * np.pi)
                    push = 0.5 * min_dist
                    pos[i][0] += push * np.cos(angle)
                    pos[i][1] += push * np.sin(angle)
                    pos[j][0] -= push * np.cos(angle)
                    pos[j][1] -= push * np.sin(angle)
    
    circles[:, :2] = pos
    return circles

def _grow_radii(circles, dt_growth, iterations=10):
    """Grow radii while maintaining feasibility via PBD."""
    n = len(circles)
    # Grow all radii slightly
    circles[:, 2] += dt_growth
    
    # Run PBD to fix violations
    for _ in range(iterations):
        _pbd_step(circles, dt_growth, iterations=5)
        # Check if any circle went out of bounds or overlapping
        valid = True
        for i in range(n):
            x, y, r = circles[i]
            if x < r - 1e-10 or x > 1 - r + 1e-10 or y < r - 1e-10 or y > 1 - r + 1e-10:
                valid = False
                break
        if not valid:
            # Revert growth
            circles[:, 2] -= dt_growth
            return circles, False
    
    return circles, True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Initialize with grid placement
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.01) for i in range(side) for j in range(side)]
    circles = np.array(cells[:N])
    
    best_score = _compute_score(circles)
    best_circles = circles.copy()
    
    # PBD optimization with growing radii
    dt_growth = 0.001
    growth_count = 0
    
    while time.time() < deadline:
        # Try to grow radii
        circles, success = _grow_radii(circles, dt_growth, iterations=10)
        if success:
            growth_count += 1
            # Check score
            score = _compute_score(circles)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
            # Increase growth rate slightly
            dt_growth = min(0.005, dt_growth * 1.01)
        else:
            # Decrease growth rate
            dt_growth = max(0.0001, dt_growth * 0.9)
        
        # Periodic full PBD pass
        if growth_count % 50 == 0:
            _pbd_step(circles, 0.001, iterations=30)
    
    # Final validation
    final_score = _compute_score(best_circles)
    if final_score > 0:
        return best_circles
    return circles
# EVOLVE-BLOCK-END
