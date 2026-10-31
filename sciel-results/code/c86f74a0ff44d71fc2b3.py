"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _penalty(x, y, r, eps=1e-4):
    """Compute a penalty for constraint violations."""
    n = len(r)
    # Boundary constraints
    pen = 0.0
    for i in range(n):
        if r[i] > 0:
            pen += max(0, r[i] - x[i]) ** 2
            pen += max(0, r[i] - (1 - x[i])) ** 2
            pen += max(0, r[i] - y[i]) ** 2
            pen += max(0, r[i] - (1 - y[i])) ** 2
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            min_dist = r[i] + r[j]
            if dist_sq < min_dist ** 2:
                diff = min_dist - np.sqrt(dist_sq)
                pen += diff ** 2
    return pen


def _objective(x, y, r):
    """Maximize sum of radii subject to constraints."""
    return np.sum(r) - 10 * _penalty(x, y, r)


def _is_valid(x, y, r, tol=1e-6):
    """Check if the packing is valid."""
    n = len(r)
    for i in range(n):
        if r[i] < -tol:
            return False
        if x[i] < -tol or x[i] > 1 + tol:
            return False
        if y[i] < -tol or y[i] > 1 + tol:
            return False
        if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _post_process(x, y, r):
    """Ensure validity by shrinking radii if needed."""
    n = len(r)
    for _ in range(100):
        valid = True
        for i in range(n):
            # Check boundaries
            if x[i] < r[i]:
                r[i] = max(0, x[i] - 1e-8)
                valid = False
            if x[i] > 1 - r[i]:
                r[i] = max(0, 1 - x[i] - 1e-8)
                valid = False
            if y[i] < r[i]:
                r[i] = max(0, y[i] - 1e-8)
                valid = False
            if y[i] > 1 - r[i]:
                r[i] = max(0, 1 - y[i] - 1e-8)
                valid = False
        # Check pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                if dist < r[i] + r[j]:
                    # Scale both radii down
                    scale = dist / (r[i] + r[j]) * 0.999
                    r[i] *= scale
                    r[j] *= scale
                    valid = False
        if valid:
            break
    return x, y, r


def _make_grid_initial(n):
    """Create a grid-based initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    x = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:n])
    y = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:n])
    r = np.full(n, 0.5 / side - 1e-3)
    return x, y, r


def _make_random_initial(n, seed=None):
    """Create a random initial guess."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    x = np.array([rng.random() for _ in range(n)])
    y = np.array([rng.random() for _ in range(n)])
    r = np.array([0.05 + rng.random() * 0.05 for _ in range(n)])
    return x, y, r


def _gradient_descent(x, y, r, max_iter=5000, lr=0.01, deadline=None):
    """Simple gradient descent on the penalized objective."""
    n = len(r)
    best_obj = _objective(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradients numerically
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        grad_r = np.zeros(n)
        
        for i in range(n):
            h = 1e-5
            # x gradient
            x[i] += h
            f = _objective(x, y, r)
            x[i] -= 2 * h
            f0 = _objective(x, y, r)
            x[i] += h
            grad_x[i] = (f - f0) / (2 * h)
            
            # y gradient
            y[i] += h
            f = _objective(x, y, r)
            y[i] -= 2 * h
            f0 = _objective(x, y, r)
            y[i] += h
            grad_y[i] = (f - f0) / (2 * h)
            
            # r gradient
            r[i] += h
            f = _objective(x, y, r)
            r[i] -= 2 * h
            f0 = _objective(x, y, r)
            r[i] += h
            grad_r[i] = (f - f0) / (2 * h)
        
        # Update
        x -= lr * grad_x
        y -= lr * grad_y
        r -= lr * grad_r
        
        # Clamp to valid range
        r = np.maximum(r, 0)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        obj = _objective(x, y, r)
        if obj > best_obj:
            best_obj = obj
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    return best_x, best_y, best_r


def _simulated_annealing(x, y, r, max_iter=10000, t_start=0.01, t_end=1e-6, deadline=None):
    """Simulated annealing for circle packing."""
    n = len(r)
    best_obj = _objective(x, y, r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    current_obj = best_obj
    current_x, current_y, current_r = x.copy(), y.copy(), r.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Random perturbation
        new_x = current_x.copy()
        new_y = current_y.copy()
        new_r = current_r.copy()
        
        idx = np.random.randint(n)
        delta = t * 0.1
        new_x[idx] += np.random.randn() * delta
        new_y[idx] += np.random.randn() * delta
        new_r[idx] += np.random.randn() * delta * 0.5
        
        # Clamp
        new_r = np.maximum(new_r, 0)
        new_x = np.clip(new_x, 0, 1)
        new_y = np.clip(new_y, 0, 1)
        
        new_obj = _objective(new_x, new_y, new_r)
        delta_obj = new_obj - current_obj
        
        if delta_obj > 0 or np.random.random() < np.exp(delta_obj / t):
            current_obj = new_obj
            current_x, current_y, current_r = new_x, new_y, new_r
            
            if current_obj > best_obj:
                best_obj = current_obj
                best_x, best_y, best_r = current_x.copy(), current_y.copy(), current_r.copy()
    
    return best_x, best_y, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_sum = -np.inf
    best_result = None
    
    # Start with grid initial
    x, y, r = _make_grid_initial(N)
    
    # Run gradient descent
    x, y, r = _gradient_descent(x, y, r, max_iter=3000, lr=0.005, deadline=deadline)
    x, y, r = _post_process(x, y, r)
    if _is_valid(x, y, r):
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_result = np.column_stack([x, y, r])
    
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Try simulated annealing from the current best
    x, y, r = _simulated_annealing(x, y, r, max_iter=5000, t_start=0.005, t_end=1e-6, deadline=deadline)
    x, y, r = _post_process(x, y, r)
    if _is_valid(x, y, r):
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_result = np.column_stack([x, y, r])
    
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Try a few random restarts
    for seed in range(5):
        if time.time() > deadline:
            break
        
        x, y, r = _make_random_initial(N, seed=seed)
        x, y, r = _gradient_descent(x, y, r, max_iter=2000, lr=0.005, deadline=deadline)
        x, y, r = _post_process(x, y, r)
        if _is_valid(x, y, r):
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_result = np.column_stack([x, y, r])
    
    if best_result is not None:
        return best_result
    
    # Fallback to grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
