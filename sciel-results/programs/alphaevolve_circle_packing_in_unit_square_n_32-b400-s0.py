import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s
    
    best_result = None
    best_score = -np.inf
    
    def penalty_function(params):
        """Penalty function to minimize: -sum(r) + overlap_penalty + boundary_penalty"""
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Boundary penalty
        # For each circle, compute how much it violates boundaries
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)
        
        # Overlap penalty
        # Compute pairwise distances
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        # Only upper triangle
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)
        
        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol
    
    def is_valid(result):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]
        
        # Check boundaries
        if np.any(r < -1e-9):
            return False
        if np.any(xy[:, 0] - r < -1e-9) or np.any(1 - xy[:, 0] - r < -1e-9):
            return False
        if np.any(xy[:, 1] - r < -1e-9) or np.any(1 - xy[:, 1] - r < -1e-9):
            return False
        
        # Check overlaps
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - 1e-9):
            return False
        
        return True
    
    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)
    
    rng = np.random.RandomState(42)
    
    # Try multiple initial configurations
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Initial configuration
        if restart == 0:
            # Grid initial
            side = int(np.ceil(np.sqrt(N)))
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart == 1:
            # Slightly perturbed grid
            side = int(np.ceil(np.sqrt(N)))
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(N, 2))
        elif restart == 2:
            # Hexagonal-like
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        else:
            # Random
            xy = rng.uniform(0.1, 0.9, size=(N, 2))
            r = rng.uniform(0.03, 0.08, size=N)
        
        params = np.concatenate([xy.flatten(), r])
        
        # Optimize with L-BFGS-B
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B', 
                             options={'maxiter': 300, 'ftol': 1e-12})
        except Exception:
            continue
        
        if is_valid(result.x):
            score = get_score(result.x)
            if score > best_score:
                best_score = score
                best_result = result.x.copy()
        else:
            # Try to repair: project to feasible region
            # Simple repair: reduce radii until no overlap
            xy = result.x[:2*N].reshape(N, 2)
            r = result.x[2*N:]
            # Reduce radii
            for _ in range(100):
                if is_valid(np.concatenate([xy.flatten(), r])):
                    break
                r *= 0.95
            if is_valid(np.concatenate([xy.flatten(), r])):
                score = get_score(np.concatenate([xy.flatten(), r]))
                if score > best_score:
                    best_score = score
                    best_result = np.concatenate([xy.flatten(), r]).copy()
    
    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2)
        r = best_result[2*n:]
        # Clamp to be safe
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])
    
    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
