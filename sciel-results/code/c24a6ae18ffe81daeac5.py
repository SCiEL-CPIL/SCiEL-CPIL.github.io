import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 26

def _elapsed(start):
    return time.time() - start

def _valid(params):
    """Check if configuration is valid (no overlaps, all in bounds)."""
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Boundary check
    if np.any(x < r - 1e-8) or np.any(x > 1 - r + 1e-8) or np.any(y < r - 1e-8) or np.any(y > 1 - r + 1e-8):
        return False
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j] - 1e-8:
                return False
    return True

def _score(params):
    """Return sum of radii if valid, else -inf."""
    if _valid(params):
        return np.sum(params[:, 2])
    return -np.inf

def _grid_init():
    """Create a grid-based initial configuration."""
    side = 5  # 5x5 grid gives 25 cells, we need 26
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    # Add 26th circle near center
    cells.append([0.5, 0.5, 0.1])
    return np.array(cells[:N])

def _fix_to_boundary(params):
    """Ensure all circles are within bounds by shrinking if needed."""
    params = params.copy()
    for i in range(N):
        min_dist = min(params[i, 0], 1 - params[i, 0], params[i, 1], 1 - params[i, 1])
        if params[i, 2] > min_dist:
            params[i, 2] = max(min_dist, 1e-6)
    return params

def _resolve_overlaps(params):
    """Iteratively resolve overlaps by shrinking smaller circles."""
    params = params.copy()
    for _ in range(20):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = params[i, 0] - params[j, 0]
                dy = params[i, 1] - params[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                min_dist = params[i, 2] + params[j, 2]
                if d < min_dist - 1e-10:
                    # Shrink the smaller circle
                    if params[i, 2] <= params[j, 2]:
                        new_r = max(d - params[j, 2], 1e-6)
                        if new_r < params[i, 2]:
                            params[i, 2] = new_r
                            changed = True
                    else:
                        new_r = max(d - params[i, 2], 1e-6)
                        if new_r < params[j, 2]:
                            params[j, 2] = new_r
                            changed = True
        if not changed:
            break
    return _fix_to_boundary(params)

def _min_slack(params):
    """Compute the minimum slack across all pairs and boundaries."""
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    min_slack = np.inf
    
    # Boundary slacks
    for i in range(N):
        min_dist = min(x[i], 1 - x[i], y[i], 1 - y[i])
        slack = min_dist - r[i]
        if slack < min_slack:
            min_slack = slack
    
    # Pair slacks
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            slack = d - r[i] - r[j]
            if slack < min_slack:
                min_slack = slack
    
    return min_slack

def _slsqp_objective(params):
    """Objective for SLSQP: maximize sum(r) subject to non-overlap constraints."""
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Primary objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Add small penalty for negative slack to help guide optimization
    slack = _min_slack(params)
    if slack < 0:
        obj += 1000.0 * slack**2
    
    return obj

def _slsqp_constraints(params):
    """Generate constraint functions for SLSQP."""
    constraints = []
    
    # Boundary constraints: r <= min(x, 1-x, y, 1-y)
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i, 0] - p[i, 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i, 0] - p[i, 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i, 1] - p[i, 2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i, 1] - p[i, 2]})
    
    # Non-overlap constraints: d_ij >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: np.sqrt((p[i, 0] - p[j, 0])**2 + (p[i, 1] - p[j, 1])**2) - p[i, 2] - p[j, 2]
            })
    
    # Positive radius constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i, 2] - 1e-6})
    
    return constraints

def _local_search(params, max_iter=50, step_size=0.02):
    """Simulated annealing-style local search to improve packing."""
    best_params = params.copy()
    best_score = _score(best_params)
    current_score = best_score
    
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        # Generate a perturbation
        params_new = best_params.copy()
        
        # Randomly choose which circles to move
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        
        for idx in indices:
            params_new[idx, 0] += rng.uniform(-step_size, step_size)
            params_new[idx, 1] += rng.uniform(-step_size, step_size)
            # Optionally adjust radius slightly
            if rng.random() < 0.3:
                params_new[idx, 2] *= (1 + rng.uniform(-0.1, 0.1))
        
        # Fix to boundary
        params_new = _fix_to_boundary(params_new)
        
        new_score = _score(params_new)
        
        # Accept if better, or with some probability if worse (annealing)
        if new_score > current_score:
            best_params = params_new
            best_score = new_score
            current_score = new_score
        elif current_score > -np.inf and new_score > -np.inf:
            # Accept worse solutions with decreasing probability
            delta = new_score - current_score
            accept_prob = np.exp(delta / (0.01 * (1 - iteration / max_iter)))
            if rng.random() < accept_prob:
                current_score = new_score
                best_params = params_new
        elif new_score > best_score:
            best_params = params_new
            best_score = new_score
    
    return best_params

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin under 150s kill
    
    # Start with grid initialization
    best_params = _grid_init()
    best_params = _resolve_overlaps(best_params)
    best_score = _score(best_params)
    
    # Phase 1: SLSQP optimization with constraints
    while _elapsed(start_time) < deadline * 0.5:
        try:
            # Use SLSQP with explicit constraints
            res = minimize(
                _slsqp_objective,
                best_params,
                method='SLSQP',
                constraints=_slsqp_constraints,
                bounds=[(1e-6, 0.99), (1e-6, 0.99), (1e-6, 0.4)] * N,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if res.success:
                candidate = res.x.reshape(N, 3)
                candidate = _fix_to_boundary(candidate)
                candidate = _resolve_overlaps(candidate)
                
                score = _score(candidate)
                if score > best_score:
                    best_score = score
                    best_params = candidate
        except Exception:
            pass
        
        # Check time
        if _elapsed(start_time) > deadline * 0.5:
            break
    
    # Phase 2: Local search with simulated annealing
    while _elapsed(start_time) < deadline:
        # Perform local search
        improved = _local_search(best_params, max_iter=30, step_size=0.015)
        
        score = _score(improved)
        if score > best_score:
            best_score = score
            best_params = improved
        
        # Try SLSQP again from improved position
        try:
            res = minimize(
                _slsqp_objective,
                best_params,
                method='SLSQP',
                constraints=_slsqp_constraints,
                bounds=[(1e-6, 0.99), (1e-6, 0.99), (1e-6, 0.4)] * N,
                options={'maxiter': 80, 'ftol': 1e-6}
            )
            
            if res.success:
                candidate = res.x.reshape(N, 3)
                candidate = _fix_to_boundary(candidate)
                candidate = _resolve_overlaps(candidate)
                
                score = _score(candidate)
                if score > best_score:
                    best_score = score
                    best_params = candidate
        except Exception:
            pass
        
        # Check time
        if _elapsed(start_time) > deadline:
            break
    
    # Final validation and cleanup
    final_params = _fix_to_boundary(best_params)
    final_params = _resolve_overlaps(final_params)
    
    # Ensure strict validity
    if not _valid(final_params):
        # Last resort: shrink all circles slightly
        for i in range(N):
            final_params[i, 2] *= 0.95
        final_params = _fix_to_boundary(final_params)
        final_params = _resolve_overlaps(final_params)
    
    return final_params
