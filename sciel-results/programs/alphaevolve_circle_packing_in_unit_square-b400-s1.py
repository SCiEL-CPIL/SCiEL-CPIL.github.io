"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    # Check radii positive
    if np.any(r <= 0):
        return None
    # Check inside unit square
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    # Check disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return None
    return float(np.sum(r))


def _grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _random_initial(n=N, seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.02 + 0.03 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _penalty_objective(circles):
    """Objective to minimize: -sum(r) + penalty for violations."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(r)
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    boundary_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                    np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    obj += 100 * np.sum(boundary_viol)
    
    # Penalty for circle overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = max(0, r[i] + r[j] - dist)
            obj += 50 * overlap
    
    return obj


def _gradient_descent(circles, max_iter=20000, lr_init=0.01, lr_decay=0.9999, 
                       time_limit=100, seed=0):
    """Gradient descent with penalty method to optimize circle packing."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradients
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        
        # Boundary constraints
        for i in range(n):
            # Left boundary
            if x[i] < r[i]:
                gx[i] -= 100
                gr[i] += 100
            # Right boundary
            if 1 - x[i] < r[i]:
                gx[i] += 100
                gr[i] += 100
            # Bottom boundary
            if y[i] < r[i]:
                gy[i] -= 100
                gr[i] += 100
            # Top boundary
            if 1 - y[i] < r[i]:
                gy[i] += 100
                gr[i] += 100
        
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    dist = 1e-12
                    dx = 1e-12
                    dy = 0
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    grad_dist_i = 50
                    gx[i] -= grad_dist_i * dx / dist
                    gy[i] -= grad_dist_i * dy / dist
                    gr[i] += grad_dist_i
                    
                    gx[j] += grad_dist_i * dx / dist
                    gy[j] += grad_dist_i * dy / dist
                    gr[j] += grad_dist_i
        
        # Objective gradient (maximize sum r => minimize -sum r)
        gr -= 1.0
        
        # Update with learning rate
        x -= lr * gx
        y -= lr * gy
        r -= lr * gr
        
        # Clamp radii to positive
        r = np.maximum(r, 0.001)
        
        # Clamp positions to stay roughly in bounds
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # Ensure circles stay within bounds (shrink if needed)
        for i in range(n):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] > max_r:
                r[i] = max(max_r, 0.001)
        
        lr *= lr_decay
        
        # Check score periodically
        if it % 100 == 0:
            current_circles = np.column_stack([x, y, r])
            score = _validate_and_score(current_circles)
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
                    best_circles = current_circles.copy()
    
    return best_circles, best_score


def _simulated_annealing(circles, max_iter=50000, t_init=0.05, t_min=0.001,
                          time_limit=80, seed=0):
    """Simulated annealing to optimize circle packing."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    
    start_time = time.time()
    
    def _score(cx, cy, cr):
        return _validate_and_score(np.column_stack([cx, cy, cr]))
    
    best_score = _score(x, y, r)
    best_circles = np.column_stack([x, y, r]).copy()
    current_score = best_score
    if current_score is None:
        current_score = 0.0
    
    t = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb a random circle
        i = rng.randint(n)
        move_type = rng.rand()
        
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.4:
            # Move position
            delta = 0.02 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.7:
            # Change radius
            new_r[i] *= (1 + 0.1 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        else:
            # Swap-like: adjust two circles
            j = rng.randint(n)
            if j == i:
                j = (i + 1) % n
            delta_r = 0.01 * rng.randn()
            new_r[i] += delta_r
            new_r[j] -= delta_r
            new_r[i] = max(0.005, new_r[i])
            new_r[j] = max(0.005, new_r[j])
        
        new_score = _score(new_x, new_y, new_r)
        
        if new_score is not None:
            if new_score > current_score:
                x, y, r = new_x, new_y, new_r
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_circles = np.column_stack([x, y, r]).copy()
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / max(t, 1e-10))
                    if rng.rand() < prob:
                        x, y, r = new_x, new_y, new_r
                        current_score = new_score
        else:
            # Invalid configuration, reject
            pass
        
        t *= cooling
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 120  # seconds
    safety_margin = 10  # seconds
    
    # Start with grid initialization
    circles = _grid_initial(N)
    
    # Try gradient descent
    gd_circles, gd_score = _gradient_descent(circles, max_iter=30000, lr_init=0.005,
                                              lr_decay=0.99995, time_limit=40, seed=0)
    
    best_circles = gd_circles
    best_score = gd_score
    
    # Try simulated annealing from best result
    remaining_time = time_budget - (time.time() - start_time) - safety_margin
    if remaining_time > 20:
        sa_circles, sa_score = _simulated_annealing(best_circles, max_iter=80000,
                                                     t_init=0.03, t_min=0.0005,
                                                     time_limit=remaining_time, seed=1)
        if sa_score is not None and (best_score is None or sa_score > best_score):
            best_circles = sa_circles
            best_score = sa_score
    
    # Try more gradient descent from SA result
    remaining_time = time_budget - (time.time() - start_time) - safety_margin
    if remaining_time > 10:
        gd2_circles, gd2_score = _gradient_descent(best_circles, max_iter=20000, lr_init=0.003,
                                                    lr_decay=0.99998, time_limit=remaining_time, seed=2)
        if gd2_score is not None and (best_score is None or gd2_score > best_score):
            best_circles = gd2_circles
            best_score = gd2_score
    
    # Final validation
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        # Fallback to grid
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
