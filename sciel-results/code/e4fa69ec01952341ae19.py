"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _valid_config(x, y, r):
    """Check if configuration is valid: inside unit square and no overlaps."""
    # Check boundaries
    if np.any(r < 0) or np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True

def _fix_boundaries(x, y, r, max_iter=50):
    """Ensure circles are inside the unit square and disjoint by shrinking/moving."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_iter):
        # Clip to boundaries
        for i in range(N):
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] < 1e-6:
                r[i] = 1e-6
        # Resolve overlaps by shrinking
        changed = False
        for it in range(10):
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    min_dist = r[i] + r[j] + 1e-9
                    if dist < min_dist:
                        # Shrink both proportionally
                        shrink = (min_dist - dist) / 2 + 1e-6
                        r[i] = max(r[i] - shrink, 1e-6)
                        r[j] = max(r[j] - shrink, 1e-6)
                        changed = True
            if not changed:
                break
        if not changed:
            break
    # Final boundary check
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i], 1e-6)
    return x, y, r

def _score(x, y, r):
    return float(np.sum(r))

def _gradient_descent(x, y, r, steps=200, lr=0.001, deadline=None):
    """Simple gradient ascent on sum of radii with constraint handling."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    best_score = _score(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # Boundary forces (push inward if too close)
        for i in range(N):
            # Left boundary
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10
            # Right boundary
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 10
            # Bottom boundary
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10
            # Top boundary
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 10
        
        # Overlap forces
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy) + 1e-9
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Push apart
                    fx[i] += (dx / dist) * overlap * 10
                    fy[i] += (dy / dist) * overlap * 10
                    fx[j] -= (dx / dist) * overlap * 10
                    fy[j] -= (dy / dist) * overlap * 10
                    # Try to shrink
                    fr[i] -= overlap * 5
                    fr[j] -= overlap * 5
        
        # Gradient of objective: +1 for each radius
        fr += 1.0
        
        # Update
        x += lr * fx
        y += lr * fy
        r += lr * fr
        
        # Keep radii positive
        r = np.maximum(r, 1e-6)
        
        # Keep positions in a reasonable range
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Fix boundaries
        x, y, r = _fix_boundaries(x, y, r, max_iter=5)
        
        # Track best
        if _valid_config(x, y, r):
            s = _score(x, y, r)
            if s > best_score:
                best_score = s
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    return best_x, best_y, best_r, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Initial configuration: hexagonal-ish packing
    # Start with a grid and perturb
    side = 6
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add small random perturbation
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            positions.append((x, y))
    positions = positions[:N]
    
    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    r = np.array([0.08] * N)
    
    # Fix initial boundaries
    x, y, r = _fix_boundaries(x, y, r)
    
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_score = _score(x, y, r)
    
    # Multiple restarts with different initializations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart % 4 == 0:
            # Grid-based
            side = 6
            positions = []
            for i in range(side):
                for j in range(side):
                    if len(positions) >= N:
                        break
                    x0 = (i + 0.5) / side
                    y0 = (j + 0.5) / side
                    x0 += np.random.uniform(-0.03, 0.03)
                    y0 += np.random.uniform(-0.03, 0.03)
                    positions.append((x0, y0))
            x = np.array([p[0] for p in positions[:N]])
            y = np.array([p[1] for p in positions[:N]])
            r = np.array([0.08] * N)
        elif restart % 4 == 1:
            # Random positions
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.array([0.07] * N)
        elif restart % 4 == 2:
            # Hexagonal pattern
            positions = []
            row = 0
            while len(positions) < N:
                col = 0
                while len(positions) < N:
                    x0 = 0.1 + col * 0.15 + (0.075 if row % 2 else 0)
                    y0 = 0.1 + row * 0.13
                    if x0 < 0.9 and y0 < 0.9:
                        x0 += np.random.uniform(-0.01, 0.01)
                        y0 += np.random.uniform(-0.01, 0.01)
                        positions.append((x0, y0))
                    col += 1
                row += 1
            x = np.array([p[0] for p in positions[:N]])
            y = np.array([p[1] for p in positions[:N]])
            r = np.array([0.075] * N)
        else:
            # Cluster-based: two groups
            cx1, cy1 = 0.25, 0.25
            cx2, cy2 = 0.75, 0.75
            x = []
            y = []
            for i in range(N // 2):
                x.append(cx1 + np.random.uniform(-0.15, 0.15))
                y.append(cy1 + np.random.uniform(-0.15, 0.15))
            for i in range(N - N // 2):
                x.append(cx2 + np.random.uniform(-0.15, 0.15))
                y.append(cy2 + np.random.uniform(-0.15, 0.15))
            x = np.array(x)
            y = np.array(y)
            r = np.array([0.09] * N)
        
        x, y, r = _fix_boundaries(x, y, r)
        
        # Gradient descent
        x_new, y_new, r_new, score = _gradient_descent(x, y, r, steps=150, lr=0.002, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x_new, y_new, r_new
    
    # Final polish: more iterations with smaller learning rate
    if time.time() < deadline - 10:
        x_final, y_final, r_final, score_final = _gradient_descent(
            best_x, best_y, best_r, steps=300, lr=0.0005, deadline=deadline
        )
        if score_final > best_score:
            best_score = score_final
            best_x, best_y, best_r = x_final, y_final, r_final
    
    # Final fix
    best_x, best_y, best_r = _fix_boundaries(best_x, best_y, best_r, max_iter=100)
    
    result = np.column_stack((best_x, best_y, best_r))
    return result
# EVOLVE-BLOCK-END
