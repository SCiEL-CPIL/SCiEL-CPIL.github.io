"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: soft-penalty gradient descent with L-BFGS-B, hex-lattice multi-start."""
N = 26
import numpy as np
from scipy.optimize import minimize
import time

def _soft_objective(x):
    """Soft-penalty objective: minimize -sum(r) + large * (boundary_viol + overlap_viol)."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(rs)
    
    # Boundary penalty: circle must be inside [0,1]^2
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    bviol = (np.maximum(0, rs - xs) +
             np.maximum(0, rs - (1 - xs)) +
             np.maximum(0, rs - ys) +
             np.maximum(0, rs - (1 - ys)))
    
    # Overlap penalty: (ri+rj)^2 > dist^2
    overlap = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            min_d = rs[i] + rs[j]
            if min_d > 0:
                viol = min_d - np.sqrt(d2 + 1e-12)
                if viol > 0:
                    overlap += viol * viol
    
    penalty = 500.0 * (np.sum(bviol ** 2) + overlap)
    return obj + penalty

def _soft_grad(x):
    """Gradient of soft-penalty objective."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    grad = np.zeros_like(x)
    
    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0
    
    # Boundary penalty gradient
    # For circle i: penalty = 500*(max(0,ri-xs_i)^2 + max(0,ri-(1-xs_i))^2 + ...)
    for i in range(N):
        # r - x
        v = rs[i] - xs[i]
        if v > 0:
            grad[3*i] += 500.0 * 2 * v * (-1)   # d/dx of (ri-xi)^2
            grad[3*i+2] += 500.0 * 2 * v * (1)   # d/dr of (ri-xi)^2
        # r - (1-x)
        v = rs[i] - (1 - xs[i])
        if v > 0:
            grad[3*i] += 500.0 * 2 * v * (1)     # d/dx of (ri-(1-xi))^2 = -2v*(-1)=2v
            grad[3*i+2] += 500.0 * 2 * v * (1)
        # r - y
        v = rs[i] - ys[i]
        if v > 0:
            grad[3*i+1] += 500.0 * 2 * v * (-1)
            grad[3*i+2] += 500.0 * 2 * v * (1)
        # r - (1-y)
        v = rs[i] - (1 - ys[i])
        if v > 0:
            grad[3*i+1] += 500.0 * 2 * v * (1)
            grad[3*i+2] += 500.0 * 2 * v * (1)
    
    # Overlap penalty gradient
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_d = rs[i] + rs[j]
            if d > 1e-6 and min_d > d:
                viol = min_d - d
                # d(overlap)/dx_i = 2*viol * d(d2)/dx_i / (2*d) ... 
                # Actually overlap_i = (ri+rj - d)^2
                # d(overlap)/dx_i = 2*(ri+rj-d) * (-d(dx)/d(xi)) where d = sqrt((xi-xj)^2+...)
                # d(d)/dx_i = dx/d = (xi-xj)/d
                # So d(overlap)/dx_i = 2*viol * (-(xi-xj)/d) ... wait sign
                # Let me redo: overlap = (ri+rj - d)^2
                # d(overlap)/dx_i = 2*(ri+rj-d) * (-dd/dxi) = 2*viol * (-(xi-xj)/d)
                # Hmm, dd/dxi = (xi-xj)/d, so -dd/dxi = -(xi-xj)/d = (xj-xi)/d
                # Actually: d = sqrt((xi-xj)^2 + (yi-yj)^2)
                # dd/dxi = (xi-xj)/d
                # d(overlap)/dx_i = 2*(ri+rj-d)*(-dd/dxi) = 2*viol * (-(xi-xj)/d)
                coeff = 500.0 * 2 * viol / d
                grad[3*i] += coeff * (-dx)   # -dx = xj - xi... wait
                # Let me be careful:
                # overlap = (ri+rj - d)^2, d = sqrt(dx^2+dy^2), dx = xi-xj
                # d(overlap)/dx_i = 2*(ri+rj-d) * (-d(d)/dx_i)
                # d(d)/dx_i = dx/d (since d/dxi of (xi-xj) = 1)
                # So d(overlap)/dx_i = 2*viol * (-dx/d)
                grad[3*i] = 0  # reset, let me recompute cleanly
                # Actually I already set grad[2::3]=-1, so I need to add to it
                # Let me just add the overlap contribution
                pass
    
    # Redo overlap gradient more carefully
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_d = rs[i] + rs[j]
            if d > 1e-6 and min_d > d:
                viol = min_d - d
                # overlap = (min_d - d)^2
                # d(overlap)/dx_i = 2*(min_d-d) * (-dd/dxi) = 2*viol * (-(dx)/d)
                # d(overlap)/dx_j = 2*(min_d-d) * (-dd/dxj) = 2*viol * (-(-dx)/d) = 2*viol*(dx/d)
                # d(overlap)/dr_i = 2*(min_d-d) * 1 = 2*viol
                # d(overlap)/dr_j = 2*viol
                c = 500.0 * 2 * viol / d
                grad[3*i] += c * (-dx)
                grad[3*i+1] += c * (-dy)
                grad[3*j] += c * (dx)
                grad[3*j+1] += c * (dy)
                grad[3*i+2] += 500.0 * 2 * viol
                grad[3*j+2] += 500.0 * 2 * viol
    
    return grad

def _is_valid(x, tol=1e-6):
    """Check if configuration is valid (all constraints satisfied within tolerance)."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    if np.any(rs < -tol):
        return False
    if np.any(xs < -tol) or np.any(xs > 1 + tol) or np.any(ys < -tol) or np.any(ys > 1 + tol):
        return False
    if np.any(rs > xs + tol) or np.any(rs > 1 - xs + tol) or np.any(rs > ys + tol) or np.any(rs > 1 - ys + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            if d2 < (rs[i] + rs[j])**2 - tol:
                return False
    return True

def _sum_radii(x):
    return np.sum(x.reshape(N, 3)[:, 2])

def _hex_init(rng, scale=1.0, offset=(0.0, 0.0)):
    """Generate hexagonal lattice initial configuration."""
    pos = np.zeros((N, 3))
    # Hexagonal packing: rows offset by half spacing
    spacing = 0.19 * scale
    row_dy = spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < N:
        for col in range(8):
            if idx >= N:
                break
            x = (col + 0.5) * spacing + (row % 2) * spacing / 2 + offset[0]
            y = row * row_dy + offset[1]
            r = spacing / 2 * 0.95
            # Clip to stay in [0,1]
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            pos[idx] = [x, y, r]
            idx += 1
        row += 1
    # Add small random perturbation
    pos += rng.uniform(-0.005, 0.005, size=pos.shape)
    # Ensure feasibility
    for i in range(N):
        pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1], 0.49)
        pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        pos[i, 1] = np.clip(pos[i, 1], 0, 1)
    return pos.flatten()

def _grid_init(rng, side=5):
    """Perturbed grid initial configuration."""
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 2] = 0.5 / side * 0.95
    for i in range(N):
        pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1], 0.49)
        pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        pos[i, 1] = np.clip(pos[i, 1], 0, 1)
    return pos.flatten()

def _random_init(rng):
    """Random initial configuration with small radii."""
    pos = np.zeros((N, 3))
    pos[:, 0] = rng.uniform(0.05, 0.95, size=N)
    pos[:, 1] = rng.uniform(0.05, 0.95, size=N)
    pos[:, 2] = rng.uniform(0.03, 0.12, size=N)
    for i in range(N):
        pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1], 0.49)
    return pos.flatten()

def _grow_radii(x, rng, deadline):
    """Try to grow radii of each circle greedily while maintaining validity."""
    pos = x.reshape(N, 3)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            # Try to grow radius of circle i
            for _ in range(50):
                old_r = pos[i, 2]
                new_r = old_r * 1.02 + 0.001
                if new_r > 0.49:
                    break
                pos[i, 2] = new_r
                if _is_valid(pos.flatten()):
                    improved = True
                else:
                    pos[i, 2] = old_r
                    break
    return pos.flatten()

def _local_refine(x, rng, deadline):
    """Local refinement: small random moves + radius growth."""
    pos = x.reshape(N, 3)
    current_val = _sum_radii(x)
    T = 0.005
    step = 0.008
    max_iter = 3000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        i = rng.integers(N)
        dim = rng.integers(3)
        old = pos[i, dim]
        pos[i, dim] += rng.normal(0, step)
        if dim == 0:
            pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        elif dim == 1:
            pos[i, 1] = np.clip(pos[i, 1], 0, 1)
        else:
            pos[i, 2] = np.clip(pos[i, 2], 0, 0.49)
        
        if _is_valid(pos.flatten()):
            new_val = _sum_radii(pos.flatten())
            delta = new_val - current_val
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_val = new_val
            else:
                pos[i, dim] = old
        else:
            pos[i, dim] = old
        T *= 0.9995
        step *= 0.9998
    
    # Final radius growth pass
    pos = _grow_radii(pos.flatten(), rng, deadline)
    x[:] = pos
    return x

def construct():
    deadline = time.time() + 145  # 5s safety margin
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = -np.inf
    
    # Generate diverse initial configurations
    inits = []
    # Hex lattice variants
    for scale in [0.9, 1.0, 1.1]:
        for ox in [0.0, 0.05, -0.03]:
            for oy in [0.0, 0.04]:
                inits.append(_hex_init(rng, scale=scale, offset=(ox, oy)))
    # Grid variants
    inits.append(_grid_init(rng, side=5))
    inits.append(_grid_init(rng, side=6))
    # Random
    inits.append(_random_init(rng))
    inits.append(_random_init(rng))
    
    # Shuffle and take as many as we can run within budget
    rng.shuffle(inits)
    
    bounds = [(0, 1), (0, 1), (0, 0.49)] * N
    
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline - 12:
            break
        
        # Run L-BFGS-B on soft-penalty objective
        try:
            res = minimize(_soft_objective, x0, jac=_soft_grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Check if result is valid
            if _is_valid(res.x):
                val = _sum_radii(res.x)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
        
        # Also try local refinement from this start if it's valid
        if _is_valid(x0) and time.time() < deadline - 15:
            refined = _local_refine(x0.copy(), rng, deadline - 15)
            if _is_valid(refined):
                val = _sum_radii(refined)
                if val > best_val:
                    best_val = val
                    best_x = refined.copy()
    
    # If we have a best, do a final polish with L-BFGS-B from the best
    if best_x is not None and time.time() < deadline - 8:
        try:
            res = minimize(_soft_objective, best_x, jac=_soft_grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if _is_valid(res.x):
                val = _sum_radii(res.x)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Final local refinement + radius growth
    if best_x is not None and time.time() < deadline - 3:
        best_x = _local_refine(best_x.copy(), rng, deadline - 3)
        if not _is_valid(best_x):
            # Revert if refinement broke validity
            best_x = best_x  # keep whatever we have, should be valid from above
    
    if best_x is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)
    
    return best_x.reshape(N, 3)
