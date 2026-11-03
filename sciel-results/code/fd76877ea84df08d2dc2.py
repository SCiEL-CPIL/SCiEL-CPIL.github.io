"""Circle packing in unit square - 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 110.0  # seconds, safety margin before 120s limit

def _penalty_and_grad(z, lam):
    """
    z: (3N,) vector, groups of (x_i, y_i, r_i)
    Returns (penalty_value, penalty_grad)
    Penalty encourages feasibility; we minimize -sum(r) + lam * penalty
    """
    n = len(z) // 3
    pts = z.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    # Boundary violations: x < r, x > 1-r, y < r, y > 1-r
    # For each circle, 4 constraints
    # c1 = x - r >= 0  -> violation if x - r < 0
    # c2 = 1 - x - r >= 0 -> violation if 1 - x - r < 0
    # c3 = y - r >= 0
    # c4 = 1 - y - r >= 0
    
    v1 = np.maximum(0.0, r - x)      # left wall
    v2 = np.maximum(0.0, x + r - 1.0)  # right wall
    v3 = np.maximum(0.0, r - y)      # bottom wall
    v4 = np.maximum(0.0, y + r - 1.0)  # top wall
    
    # Pairwise overlap violations
    # For i < j: violation if dist < r_i + r_j
    # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    # violation = max(0, r_i + r_j - d_ij)
    
    pair_viol = 0.0
    pair_grad = np.zeros_like(z)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            d = np.sqrt(d2 + 1e-12)
            s = r[i] + r[j]
            if d < s:
                viol = s - d
                pair_viol += viol * viol
                # d(viol^2)/d(x_i) = 2*viol * d(s-d)/d(x_i) = 2*viol * (-dx/d)
                # d(viol^2)/d(r_i) = 2*viol * 1
                factor = 2.0 * viol / d
                pair_grad[3*i + 0] += -factor * dx
                pair_grad[3*i + 1] += -factor * dy
                pair_grad[3*i + 2] += 2.0 * viol
                pair_grad[3*j + 0] += factor * dx
                pair_grad[3*j + 1] += factor * dy
                pair_grad[3*j + 2] += 2.0 * viol
    
    # Boundary penalty
    bnd_pen = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
    bnd_grad = np.zeros_like(z)
    # d/dx: v1^2 -> 2*v1*(-1), v2^2 -> 2*v2*(1)
    # d/dy: v3^2 -> 2*v3*(-1), v4^2 -> 2*v4*(1)
    # d/dr: v1^2 -> 2*v1*(1), v2^2 -> 2*v2*(1), v3^2 -> 2*v3*(1), v4^2 -> 2*v4*(1)
    bnd_grad[0::3] += -2.0*v1 + 2.0*v2   # x component
    bnd_grad[1::3] += -2.0*v3 + 2.0*v4   # y component
    bnd_grad[2::3] += 2.0*(v1 + v2 + v3 + v4)  # r component
    
    total_pen = bnd_pen + pair_viol
    total_grad = bnd_grad + pair_grad
    
    return total_pen, total_grad


def _objective_and_grad(z, lam):
    """
    Minimize: -sum(r_i) + lam * penalty
    """
    n = len(z) // 3
    r = z[2::3]
    obj = -np.sum(r)
    obj_grad = np.zeros_like(z)
    obj_grad[2::3] = -1.0
    
    pen, pen_grad = _penalty_and_grad(z, lam)
    obj += lam * pen
    obj_grad += lam * pen_grad
    
    return obj, obj_grad


def _is_valid(pts, tol=1e-6):
    """Check if packing is valid: all circles inside [0,1]^2, no overlaps."""
    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]
    
    # Boundary check
    if np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
        return False
    if np.any(r < -tol):
        return False
    # Check x - r >= -tol, etc.
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol):
        return False
    
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            s = r[i] + r[j]
            if d2 < s*s - tol:
                return False
    return True


def _score(pts):
    """Return sum of radii if valid, else -inf."""
    if _is_valid(pts):
        return np.sum(pts[:, 2])
    return -np.inf


def construct():
    start_time = time.time()
    
    best_score = -np.inf
    best_pts = None
    
    # Initial grid-based solution as fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
             for i in range(side) for j in range(side)]
    fallback = np.array(cells[:N])
    fallback_score = _score(fallback)
    if fallback_score > best_score:
        best_score = fallback_score
        best_pts = fallback.copy()
    
    # Multi-start optimization
    np.random.seed(42)
    
    # Initial radius estimate: for N circles in unit square, max r ~ 1/sqrt(N)
    # More carefully: if packed in grid, r ~ 0.5/sqrt(N)
    r_est = 0.5 / np.sqrt(N)
    
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate initial positions
        if s == 0:
            # Grid initialization
            positions = []
            for i in range(N):
                row = i // side
                col = i % side
                px = (col + 0.5) / side
                py = (row + 0.5) / side
                positions.append([px, py, r_est])
            z0 = np.array(positions).flatten()
        else:
            # Random perturbation around grid
            positions = []
            for i in range(N):
                row = i // side
                col = i % side
                px = (col + 0.5) / side + np.random.randn() * 0.1
                py = (row + 0.5) / side + np.random.randn() * 0.1
                r = r_est * (0.8 + 0.4 * np.random.rand())
                # Clamp
                px = np.clip(px, 0.01, 0.99)
                py = np.clip(py, 0.01, 0.99)
                r = np.clip(r, 0.01, 0.4)
                positions.append([px, py, r])
            z0 = np.array(positions).flatten()
        
        # Optimize with increasing penalty
        z = z0.copy()
        lam = 10.0
        
        for it in range(30):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            obj, grad = _objective_and_grad(z, lam)
            
            # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))  # x
                bounds.append((0.0, 1.0))  # y
                bounds.append((0.0, 0.5))  # r
            
            res = minimize(
                lambda z: _objective_and_grad(z, lam)[0],
                z,
                jac=lambda z: _objective_and_grad(z, lam)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            z = res.x
            
            # Check improvement
            pts = z.reshape(N, 3)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
            
            # Increase penalty if still infeasible
            pen, _ = _penalty_and_grad(z, lam)
            if pen > 1e-6:
                lam *= 2.0
                lam = min(lam, 1e6)
            else:
                break
        
        # Try to polish: increase lambda significantly for final feasibility
        if time.time() - start_time > DEADLINE - 5:
            break
        
        lam_final = max(lam, 1e4)
        for it in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            obj, grad = _objective_and_grad(z, lam_final)
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 0.5))
            
            res = minimize(
                lambda z: _objective_and_grad(z, lam_final)[0],
                z,
                jac=lambda z: _objective_and_grad(z, lam_final)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            z = res.x
            pts = z.reshape(N, 3)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
            
            pen, _ = _penalty_and_grad(z, lam_final)
            if pen < 1e-8:
                break
    
    # Final check and return
    if best_pts is not None and _is_valid(best_pts):
        return best_pts
    return fallback
