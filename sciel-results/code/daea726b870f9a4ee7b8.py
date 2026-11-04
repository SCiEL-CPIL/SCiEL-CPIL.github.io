"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """Pack 26 circles in unit square, maximize sum of radii."""
    deadline = time.time() + 115  # safety margin for 120s limit
    
    # Precompute pair indices (i < j)
    n = N
    idx_i, idx_j = np.triu_indices(n, k=1)
    npairs = len(idx_i)
    
    def unpack(v):
        """v: (3n,) -> positions (n,2), radii (n,)"""
        pos = v[:2*n].reshape(n, 2)
        r = v[2*n:]
        return pos, r
    
    def pack(pos, r):
        return np.concatenate([pos.ravel(), r])
    
    def objective(v):
        pos, r = unpack(v)
        # Wall constraints: x in [r, 1-r], y in [r, 1-r]
        # We handle walls by clamping in a soft way or by using bounds.
        # Actually, we use bounds in L-BFGS-B, so walls are handled by the optimizer.
        # But we need to ensure r > 0. Bounds: r in [1e-6, 0.5], x in [1e-6, 1-1e-6] etc.
        # Wait, if x < r, the circle sticks out. The constraint is x >= r AND x <= 1-r.
        # So bounds on x depend on r. This is a non-separable constraint.
        # Standard approach: use variables x, y, r and enforce:
        # r <= x <= 1-r  =>  x - r >= 0 and 1 - x - r >= 0
        # We can use a penalty or soft constraints in the objective.
        
        # Soft penalty for wall violations:
        wall_pen = 0.0
        # x - r < 0 -> penalty
        viol_x_low = np.maximum(0, r - pos[:, 0])
        viol_x_high = np.maximum(0, pos[:, 0] + r - 1)
        viol_y_low = np.maximum(0, r - pos[:, 1])
        viol_y_high = np.maximum(0, pos[:, 1] + r - 1)
        wall_pen = np.sum(viol_x_low**2 + viol_x_high**2 + viol_y_low**2 + viol_y_high**2)
        
        # Pairwise overlap penalty:
        # dist_ij < r_i + r_j
        dx = pos[idx_i, 0] - pos[idx_j, 0]
        dy = pos[idx_i, 1] - pos[idx_j, 1]
        d = np.sqrt(dx**2 + dy**2 + 1e-12)
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - d)
        pair_pen = np.sum(overlap**2)
        
        # Maximize sum(r) => minimize -sum(r) + penalties
        return -np.sum(r) + 100.0 * (wall_pen + pair_pen)
    
    def gradient(v):
        pos, r = unpack(v)
        grad_pos = np.zeros_like(pos)
        grad_r = np.zeros_like(r)
        
        # Wall penalties
        # d/dx (max(0, r-x)^2) = -2*max(0, r-x) if r-x>0 else 0
        vx_l = np.maximum(0, r - pos[:, 0])
        vx_h = np.maximum(0, pos[:, 0] + r - 1)
        vy_l = np.maximum(0, r - pos[:, 1])
        vy_h = np.maximum(0, pos[:, 1] + r - 1)
        
        grad_pos[:, 0] += -2 * vx_l + 2 * vx_h
        grad_pos[:, 1] += -2 * vy_l + 2 * vy_h
        grad_r += 2 * vx_l + 2 * vx_h + 2 * vy_l + 2 * vy_h
        
        # Pairwise penalties
        dx = pos[idx_i, 0] - pos[idx_j, 0]
        dy = pos[idx_i, 1] - pos[idx_j, 1]
        d = np.sqrt(dx**2 + dy**2 + 1e-12)
        overlap = np.maximum(0, r[idx_i] + r[idx_j] - d)
        
        # d/pos_i: -2*overlap * (dx/dx_i, dy/dx_i) = -2*overlap * (1, 0) for i, (0,0) for j? 
        # Actually d = ||p_i - p_j||. d/dp_i = (p_i - p_j)/d = (dx, dy)/d
        # d/dp_j = (p_j - p_i)/d = (-dx, -dy)/d
        # penalty = (r_i + r_j - d)^2
        # d/dp_i = 2*(r_i+r_j-d) * (-d/dp_i) = -2*overlap * (dx/d, dy/d)
        # d/dp_j = 2*(r_i+r_j-d) * (-d/dp_j) = -2*overlap * (-dx/d, -dy/d) = 2*overlap * (dx/d, dy/d)
        # d/dr_i = 2*(r_i+r_j-d) * 1 = 2*overlap
        # d/dr_j = 2*overlap
        
        coeff = -2 * overlap / d
        # Accumulate for i
        np.add.at(grad_pos, idx_i, coeff[:, None] * np.stack([dx, dy], axis=1))
        np.add.at(grad_pos, idx_j, -coeff[:, None] * np.stack([dx, dy], axis=1))
        np.add.at(grad_r, idx_i, 2 * overlap)
        np.add.at(grad_r, idx_j, 2 * overlap)
        
        # Objective gradient: -1 for each r
        grad_r -= 1.0
        
        # Scale by penalty weight 100
        grad_pos *= 100.0
        grad_r *= 100.0
        
        return np.concatenate([grad_pos.ravel(), grad_r])
    
    # Initial guess: grid
    side = 5  # 5x5 grid, use 25, add 1? No, 26. Let's use a better init.
    # 5x5 grid has 25 spots. We need 26. 
    # Let's use a random init or a perturbed grid.
    # A 5x5 grid with 1 extra circle in the middle of one cell?
    # Better: random init with reasonable radii.
    
    best_v = None
    best_val = np.inf
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # Grid init
            xs = np.linspace(0.1, 0.9, 5)
            ys = np.linspace(0.1, 0.9, 5)
            pos = np.array([(x, y) for x in xs for y in ys])[:25]
            # Add 26th at center
            pos = np.vstack([pos, [0.5, 0.5]])
            r = np.full(26, 0.09)
        else:
            # Random init
            pos = np.random.rand(26, 2) * 0.8 + 0.1
            r = np.random.rand(26) * 0.05 + 0.05
        
        v0 = pack(pos, r)
        
        # Bounds: x in [0,1], y in [0,1], r in [1e-4, 0.5]
        bounds = [(0, 1)] * 2 * n + [(1e-4, 0.5)] * n
        
        try:
            res = minimize(objective, v0, jac=gradient, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9})
            if res.fun < best_val:
                best_val = res.fun
                best_v = res.x
        except Exception:
            continue
    
    if best_v is None:
        # Fallback
        return np.array([(0.5, 0.5, 0.01)] * 26)
    
    pos, r = unpack(best_v)
    
    # Post-process: ensure validity
    # Clamp positions and radii to satisfy constraints strictly
    # If a circle violates walls, shrink it or move it
    for i in range(n):
        # Wall constraints
        r[i] = min(r[i], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        r[i] = max(r[i], 1e-6)
    
    # Resolve overlaps by shrinking
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx**2 + dy**2)
                min_d = r[i] + r[j]
                if d < min_d - 1e-9:
                    # Shrink the smaller one
                    if r[i] < r[j]:
                        r[i] = max(1e-6, r[i] - (min_d - d) / 2)
                    else:
                        r[j] = max(1e-6, r[j] - (min_d - d) / 2)
                    changed = True
        if not changed:
            break
    
    # Final wall check
    for i in range(n):
        r[i] = min(r[i], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        r[i] = max(r[i], 1e-6)
    
    result = np.column_stack([pos, r])
    return result

# EVOLVE-BLOCK-END
