# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

TOL = 1e-7
N = 10
DEADLINE = time.time() + 610.0  # safety margin before 630s kill


def _is_valid(centers, sides):
    """Check all constraints: inside unit square, positive sides, no overlap."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside [-tol, 1+tol]^2
    if np.any(centers[:, 0] - sides / 2 < -TOL) or np.any(centers[:, 0] + sides / 2 > 1 + TOL):
        return False
    if np.any(centers[:, 1] - sides / 2 < -TOL) or np.any(centers[:, 1] + sides / 2 > 1 + TOL):
        return False
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(x):
    """
    Objective: maximize sum(sides) - penalty.
    x = [s_0..s_9, cx_0..cx_9, cy_0..cy_9]  (30 vars)
    Penalty for constraint violations (smooth soft-min for overlap).
    """
    s = x[:N]
    cx = x[N:2*N]
    cy = x[2*N:3*N]
    
    # Ensure positive sides
    s = np.maximum(s, 1e-8)
    
    penalty = 0.0
    
    # Boundary penalty: square must be in [0,1]^2
    for i in range(N):
        if cx[i] - s[i]/2 < 0:
            penalty += (cx[i] - s[i]/2)**2
        if cx[i] + s[i]/2 > 1:
            penalty += (cx[i] + s[i]/2 - 1)**2
        if cy[i] - s[i]/2 < 0:
            penalty += (cy[i] - s[i]/2)**2
        if cy[i] + s[i]/2 > 1:
            penalty += (cy[i] + s[i]/2 - 1)**2
    
    # Overlap penalty: for each pair, if they overlap, penalize the penetration
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            # Required separation in x: (s_i + s_j)/2
            # Required separation in y: (s_i + s_j)/2
            # If |dx| < req_x AND |dy| < req_y, they overlap.
            # Penalty: smooth approximation of max(0, req_x - |dx|) * max(0, req_y - |dy|)
            req_x = (s[i] + s[j]) / 2
            req_y = (s[i] + s[j]) / 2
            px = req_x - abs(dx)
            py = req_y - abs(dy)
            if px > 0 and py > 0:
                penalty += px * py
    
    return -(s).sum() + 1000.0 * penalty


def _objective_with_grad(x):
    """Objective with numerical gradient."""
    return _objective(x)


def _sa_search(initial_centers, initial_sides, n_iter=50000, T0=0.05, T_end=0.001, seed=42):
    """
    Simulated Annealing to maximize sum of sides with validity constraint.
    State: (centers, sides). We perturb both.
    """
    rng = np.random.default_rng(seed)
    centers = initial_centers.copy().astype(np.float64)
    sides = initial_sides.copy().astype(np.float64)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = sides.sum()
    
    # Start temperature
    T = T0
    cooling = (T_end / T0) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time.time() > DEADLINE:
            break
        T *= cooling
        
        # Perturbation: move a random square slightly and/or change its size
        i = rng.integers(N)
        dcx = rng.normal(0, 0.02)
        dcy = rng.normal(0, 0.02)
        ds = rng.normal(0, 0.01)
        
        new_cx = centers[i, 0] + dcx
        new_cy = centers[i, 1] + dcy
        new_s = max(sides[i] + ds, 0.01)
        
        # Clip to stay roughly in bounds
        new_cx = np.clip(new_cx, new_s/2 - TOL, 1 - new_s/2 + TOL)
        new_cy = np.clip(new_cy, new_s/2 - TOL, 1 - new_s/2 + TOL)
        
        old_cx, old_cy, old_s = centers[i, 0], centers[i, 1], sides[i]
        centers[i, 0] = new_cx
        centers[i, 1] = new_cy
        sides[i] = new_s
        
        new_valid = _is_valid(centers, sides)
        new_val = sides.sum()
        
        if new_valid:
            if new_val > best_val:
                best_val = new_val
                best_centers = centers.copy()
                best_sides = sides.copy()
        else:
            # Reject invalid moves with high probability; accept with small prob if close
            # Compute violation magnitude
            viol = 0.0
            for j in range(N):
                if j == i:
                    continue
                t = (sides[i] + sides[j]) / 2 - TOL
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    viol += t - abs(centers[i,0] - centers[j,0]) + t - abs(centers[i,1] - centers[j,1])
            # Boundary violation
            for coord in [0, 1]:
                if centers[i, coord] - sides[i]/2 < -TOL:
                    viol += (-TOL - (centers[i, coord] - sides[i]/2))
                if centers[i, coord] + sides[i]/2 > 1 + TOL:
                    viol += (centers[i, coord] + sides[i]/2 - (1 + TOL))
            
            # Accept with probability exp(-viol / T)
            if T > 1e-6:
                accept_prob = np.exp(-max(viol, 0.001) / T)
            else:
                accept_prob = 0.0
            
            if rng.random() > accept_prob:
                # Revert
                centers[i, 0] = old_cx
                centers[i, 1] = old_cy
                sides[i] = old_s
            else:
                # Keep invalid state but don't update best
                pass
    return best_centers, best_sides, best_val


def _grid_packing(n_cols, n_rows, n_squares):
    """Create a grid packing with given dimensions."""
    k = max(n_cols, n_rows)
    side = 1.0 / k
    idx = np.arange(n_squares)
    cols = idx % n_cols
    rows = idx // n_cols
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n_squares, side, dtype=np.float64)
    return centers, sides


def _scipy_polish(centers, sides):
    """Use scipy.optimize.minimize to polish the solution."""
    if time.time() > DEADLINE:
        return centers, sides, sides.sum()
    
    x0 = np.concatenate([sides, centers[:, 0], centers[:, 1]])
    
    # Bounds: sides in [0.01, 1], centers in [0, 1]
    bounds = []
    for i in range(N):
        bounds.append((0.01, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    x = result.x
    new_sides = np.maximum(x[:N], 0.01)
    new_cx = x[N:2*N]
    new_cy = x[2*N:3*N]
    new_centers = np.stack([new_cx, new_cy], axis=1)
    
    if _is_valid(new_centers, new_sides):
        return new_centers, new_sides, new_sides.sum()
    
    return centers, sides, sides.sum()


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Uses SA + scipy polish to maximize sum of side lengths.
    """
    # Initial grid packing
    k = int(np.ceil(np.sqrt(N)))
    centers, sides = _grid_packing(k, k, N)
    best_val = sides.sum()
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    # Try multiple SA restarts with different seeds and initial perturbations
    seeds = [42, 123, 7, 999, 2024, 555, 888, 314]
    
    for seed in seeds:
        if time.time() > DEADLINE - 30:
            break
        
        # Perturb initial solution for diversity
        rng = np.random.default_rng(seed)
        pert_centers = centers.copy()
        pert_sides = sides.copy()
        for i in range(N):
            pert_centers[i, 0] += rng.normal(0, 0.03)
            pert_centers[i, 1] += rng.normal(0, 0.03)
            pert_sides[i] = max(sides[i] + rng.normal(0, 0.02), 0.1)
        
        # Clip to bounds
        for i in range(N):
            pert_centers[i, 0] = np.clip(pert_centers[i, 0], pert_sides[i]/2 - TOL, 1 - pert_sides[i]/2 + TOL)
            pert_centers[i, 1] = np.clip(pert_centers[i, 1], pert_sides[i]/2 - TOL, 1 - pert_sides[i]/2 + TOL)
        
        c, s, val = _sa_search(pert_centers, pert_sides, n_iter=30000, T0=0.08, T_end=0.0005, seed=seed)
        
        if val > best_val:
            best_val = val
            best_centers = c.copy()
            best_sides = s.copy()
        
        # Also try polishing the current best
        if time.time() < DEADLINE - 10:
            pc, ps, pval = _scipy_polish(best_centers, best_sides)
            if pval > best_val:
                best_val = pval
                best_centers = pc.copy()
                best_sides = ps.copy()
    
    # Final polish
    if time.time() < DEADLINE - 5:
        pc, ps, pval = _scipy_polish(best_centers, best_sides)
        if pval > best_val:
            best_val = pval
            best_centers = pc.copy()
            best_sides = ps.copy()
    
    # Verify final result
    if not _is_valid(best_centers, best_sides):
        # Fall back to grid
        centers, sides = _grid_packing(k, k, N)
        return centers, sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
