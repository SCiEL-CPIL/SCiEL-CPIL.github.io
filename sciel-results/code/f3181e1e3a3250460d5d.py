"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

# Set random seed for reproducibility in local tests, but evaluator might want variety
# We will use multiple restarts with different seeds inside construct()

def _hex_init(side=5):
    """
    Initialize circles in a hexagonal packing pattern.
    This provides a very good structural starting point.
    """
    circles = []
    # Hexagonal packing parameters
    # Distance between centers in a hex lattice with radius r is 2r
    # To fit in unit square, we need to scale.
    # Let's try to fit a grid of hexagons.
    
    # Approximate radius for a grid of size side x side in hex packing
    # Width of hex grid: side * 2r
    # Height: (side-1) * sqrt(3)*r + 2r
    # Let's just place them and let the optimizer fix the radii.
    
    r_init = 0.15
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    for i in range(side):
        for j in range(side):
            if len(circles) >= N:
                break
            x = 0.5 * dx + i * dx + (0.5 * dx if j % 2 == 1 else 0)
            y = 0.5 * dy + j * dy
            # Shift to center in [0,1]^2 roughly
            # We will normalize later or just let optimizer handle it
            circles.append([x, y, r_init])
            
    # If we didn't get enough, or if we have too many, adjust
    # For N=26, a 5x5 grid gives 25. We need 26.
    # Let's use a slightly denser or different grid.
    # Actually, let's just generate a random set of positions based on hex structure but jittered.
    
    # Better approach: Generate N points in a hex-like distribution
    points = []
    # Use a superposition of two hex grids or just a grid with hex offset
    side = int(np.ceil(np.sqrt(N)))
    # For 26, side=5 gives 25. side=6 gives 36.
    # Let's use a 5x5 grid + 1 extra circle in a gap, or just use a 6x6 grid and take 26.
    
    # Let's try a 5x5 grid (25 circles) + 1 circle in the center of a large gap?
    # Or just use a 6x6 grid and take the first 26.
    
    r = 0.12
    dx = 2 * r
    dy = np.sqrt(3) * r
    
    candidates = []
    for i in range(6):
        for j in range(6):
            x = 0.5 * dx + i * dx + (0.5 * dx if j % 2 == 1 else 0)
            y = 0.5 * dy + j * dy
            candidates.append([x, y])
            
    # Shuffle and take N
    random.shuffle(candidates)
    selected = candidates[:N]
    
    # Normalize to fit in [0,1]^2
    xs = [p[0] for p in selected]
    ys = [p[1] for p in selected]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Scale and translate
    # We want to fill the square.
    # Let's just place them in a grid that fits the square exactly for simplicity in init,
    # but with hex-like offsets.
    
    # Alternative Init: Grid with hex offset
    side = int(np.ceil(np.sqrt(N)))
    # For N=26, let's use a 5x5 grid + 1.
    # Or just use a 5x6 grid?
    
    # Let's stick to a robust grid init with hex jitter
    positions = []
    side = 5
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N: break
            # Base grid position
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            
            # Hex offset: shift odd rows
            if j % 2 == 1:
                x += 0.5 / side
            
            # Add small random jitter to break symmetry
            x += random.uniform(-0.02, 0.02)
            y += random.uniform(-0.02, 0.02)
            
            # Clamp to [0.05, 0.95]
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            positions.append([x, y, 0.1])
            count += 1
            
    # If we still don't have N (e.g. if side*side < N), add more
    while len(positions) < N:
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        positions.append([x, y, 0.1])
        
    # If we have more, truncate
    positions = positions[:N]
    
    return np.array(positions)


def _is_valid(xy, r, tol=1e-9):
    """Check if packing is valid."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -tol or 1.0 - x - ri < -tol or y - ri < -tol or 1.0 - y - ri < -tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _objective(params, n, penalty_weight=10000.0):
    """
    Objective function for optimization.
    Minimizes -sum(r) + penalty.
    """
    params = np.asarray(params)
    xy = params[0::3].reshape(-1, 2)
    r = np.maximum(params[2::3], 1e-6)
    
    # Wall constraints
    wall_pen = 0.0
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        # c1 = x - r, c2 = 1-x-r, c3 = y-r, c4 = 1-y-r
        # Penalty for c < 0
        c = np.array([x - ri, 1.0 - x - ri, y - ri, 1.0 - y - ri])
        viol = np.sum(np.maximum(-c, 0.0)**2)
        wall_pen += viol
        
    # Overlap constraints
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            c = dist - r[i] - r[j]
            if c < 0:
                overlap_pen += c**2
                
    penalty = penalty_weight * (wall_pen + overlap_pen)
    return -np.sum(r) + penalty


def _gradient_optimize(xy, r, deadline, max_iter=500):
    """
    Use L-BFGS-B to optimize the packing.
    """
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.001, 0.5))
        
    best_params = params.copy()
    best_score = _objective(params, n)
    best_xy = xy.copy()
    best_r = r.copy()
    
    try:
        result = minimize(
            _objective,
            params,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[2::3], 0.001)
        
        opt_score = _objective(opt_params, n)
        
        # We want to compare actual sum of radii for valid packings
        # If invalid, penalty makes score very negative.
        # Let's evaluate validity and sum of radii
        if _is_valid(opt_xy, opt_r):
            opt_sum_r = np.sum(opt_r)
            if _is_valid(best_xy, best_r):
                best_sum_r = np.sum(best_r)
            else:
                best_sum_r = -1e9
                
            if opt_sum_r > best_sum_r:
                best_params = opt_params.copy()
                best_xy = opt_xy.copy()
                best_r = opt_r.copy()
                
    except Exception as e:
        pass
        
    return best_xy, best_r


def _simulated_annealing(xy, r, deadline, T_start=0.01, T_end=0.001, alpha=0.999, max_iter=2000):
    """
    Simulated Annealing to explore the space.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    best_xy = xy.copy()
    best_r = r.copy()
    best_score = -1e9
    
    if _is_valid(current_xy, current_r):
        best_score = np.sum(current_r)
    else:
        best_score = -1e9
        
    T = T_start
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Perturb
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        
        # Move a random circle
        idx = random.randint(0, n - 1)
        new_xy[idx, 0] += random.gauss(0, 0.05)
        new_xy[idx, 1] += random.gauss(0, 0.05)
        new_r[idx] += random.gauss(0, 0.02)
        
        # Clamp
        new_xy[idx, 0] = np.clip(new_xy[idx, 0], new_r[idx], 1.0 - new_r[idx])
        new_xy[idx, 1] = np.clip(new_xy[idx, 1], new_r[idx], 1.0 - new_r[idx])
        new_r[idx] = np.clip(new_r[idx], 0.001, 0.5)
        
        # Evaluate
        new_valid = _is_valid(new_xy, new_r)
        new_score = np.sum(new_r) if new_valid else -1e9
        
        # Acceptance
        if new_score > best_score:
            best_score = new_score
            best_xy = new_xy.copy()
            best_r = new_r.copy()
            
        if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
            # Note: Using best_score in acceptance is not standard SA (usually current_score),
            # but using current_score might get stuck. Let's use current_score for standard SA.
            pass # Placeholder, let's implement standard SA below
            
    # Standard SA implementation
    current_xy = xy.copy()
    current_r = r.copy()
    current_score = np.sum(current_r) if _is_valid(current_xy, current_r) else -1e9
    
    T = T_start
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        idx = random.randint(0, n - 1)
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        
        new_xy[idx, 0] += random.gauss(0, 0.05)
        new_xy[idx, 1] += random.gauss(0, 0.05)
        new_r[idx] += random.gauss(0, 0.02)
        
        new_xy[idx, 0] = np.clip(new_xy[idx, 0], new_r[idx], 1.0 - new_r[idx])
        new_xy[idx, 1] = np.clip(new_xy[idx, 1], new_r[idx], 1.0 - new_r[idx])
        new_r[idx] = np.clip(new_r[idx], 0.001, 0.5)
        
        new_valid = _is_valid(new_xy, new_r)
        new_score = np.sum(new_r) if new_valid else -1e9
        
        if new_score > current_score:
            current_xy = new_xy
            current_r = new_r
            current_score = new_score
        else:
            # Accept with probability exp((new - current) / T)
            if new_score > -1e9 and current_score > -1e9: # Both valid
                prob = np.exp((new_score - current_score) / T)
                if random.random() < prob:
                    current_xy = new_xy
                    current_r = new_r
                    current_score = new_score
            # If new is invalid and current is valid, reject (unless prob is high, but -1e9 makes it 0)
            # If current is invalid and new is valid, accept
            
        if current_score > best_score:
            best_score = current_score
            best_xy = current_xy.copy()
            best_r = current_r.copy()
            
        T *= alpha
        if T < T_end:
            T = T_end
            
    return best_xy, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_xy = None
    best_r = None
    best_score = 0.0
    
    # Multiple restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # 1. Initialize
        xy, r = _hex_init()
        # Ensure initial validity by shrinking radii if needed
        for i in range(N):
            # Check walls
            max_r_wall = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = min(r[i], max_r_wall * 0.9)
            
        # Check overlaps and shrink
        for i in range(N):
            for j in range(i + 1, N):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j]:
                    # Shrink both
                    new_r_i = r[i] * 0.8
                    new_r_j = r[j] * 0.8
                    r[i] = new_r_i
                    r[j] = new_r_j
                    
        # 2. Simulated Annealing (Global Search)
        if time.time() < deadline - 20.0:
            sa_xy, sa_r = _simulated_annealing(xy, r, deadline - 20.0)
            if _is_valid(sa_xy, sa_r):
                sa_score = np.sum(sa_r)
                if sa_score > best_score:
                    best_score = sa_score
                    best_xy = sa_xy
                    best_r = sa_r
            
        # 3. Gradient Optimization (Local Polish)
        if time.time() < deadline - 10.0:
            # Use the best from SA or the initial if SA failed
            if best_xy is not None and best_score > 0:
                # We should polish the current best, or the current SA result?
                # Let's polish the SA result if it's better than current best, or just polish current best.
                # Actually, let's polish the specific trajectory.
                polish_xy, polish_r = _gradient_optimize(xy, r, deadline - 10.0, max_iter=200)
                if _is_valid(polish_xy, polish_r):
                    polish_score = np.sum(polish_r)
                    if polish_score > best_score:
                        best_score = polish_score
                        best_xy = polish_xy
                        best_r = polish_r
                        
                # Also try polishing the SA result if it was good
                if _is_valid(sa_xy, sa_r):
                    polish_sa_xy, polish_sa_r = _gradient_optimize(sa_xy, sa_r, deadline - 10.0, max_iter=200)
                    if _is_valid(polish_sa_xy, polish_sa_r):
                        polish_sa_score = np.sum(polish_sa_r)
                        if polish_sa_score > best_score:
                            best_score = polish_sa_score
                            best_xy = polish_sa_xy
                            best_r = polish_sa_r

    # Final Polish on the best found
    if best_xy is not None and time.time() < deadline - 5.0:
        final_xy, final_r = _gradient_optimize(best_xy, best_r, deadline - 2.0, max_iter=500)
        if _is_valid(final_xy, final_r):
            final_score = np.sum(final_r)
            if final_score > best_score:
                best_score = final_score
                best_xy = final_xy
                best_r = final_r

    # Construct result array
    if best_xy is None:
        # Fallback: Grid of small circles
        result = np.zeros((N, 3))
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_xy[i, 0]
        result[i, 1] = best_xy[i, 1]
        result[i, 2] = best_r[i]
        
    return result

# EVOLVE-BLOCK-END
