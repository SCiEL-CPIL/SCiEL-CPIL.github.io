"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _make_hexagonal_initial(w, h):
    """
    Create a hexagonal-like arrangement for 21 circles.
    Rows: 5, 4, 5, 4, 3 = 21 circles
    The key insight: in optimal packings, circles in the same row are equally spaced,
    and adjacent rows are offset by half a spacing.
    """
    rows = [5, 4, 5, 4, 3]
    n_rows = len(rows)
    circles = []
    
    # Estimate base radius from area considerations
    # For hexagonal packing, each circle occupies roughly (2r)^2 * sqrt(3)/2 area
    # Total area = w*h, so 21 * 2 * r^2 * sqrt(3) ≈ w*h
    # r ≈ sqrt(w*h / (42 * sqrt(3)))
    r_est = np.sqrt(w * h / (42 * np.sqrt(3)))
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / n_rows
        for col_j in range(count):
            # Offset alternating rows
            if row_i % 2 == 1:
                x = (col_j + 0.5) * w / 4.5
            else:
                x = (col_j + 0.5) * w / 5.0
            r = r_est
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _relax_circles_safe(circles, w, h, iterations=1000, tol=1e-9):
    """
    Careful relaxation that preserves validity.
    Only moves circles when they are actually violating constraints.
    """
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    for it in range(iterations):
        moved = False
        
        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] < rad[i]:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] > w - rad[i]:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] < rad[i]:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] > h - rad[i]:
                pos[i, 1] = h - rad[i]
                moved = True
        
        # 2. Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                    else:
                        push = (min_dist - dist) / 2.0
                        ux = dx / dist
                        uy = dy / dist
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[j, 0] += ux * push
                        pos[j, 1] += uy * push
                    moved = True
        
        if not moved:
            break
    
    new_circles = np.column_stack((pos, rad))
    
    # Verify validity; if broken, revert
    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles
    return circles

def _objective_and_grad(x, w, h, beta, eps):
    """
    Compute objective and gradient for penalty-based optimization.
    x = [x1,...,xn, y1,...,yn, r1,...,rn]
    """
    n = N
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    obj = -np.sum(r)
    grad = np.zeros_like(x)
    grad[2*n:] = -1.0
    
    # Boundary penalty (smooth)
    for i in range(n):
        # Left boundary
        val = cx[i] - r[i]
        if val < eps:
            t = val / eps
            pen = beta * eps * eps * np.logaddexp(0, -t)
            sig = 1.0 / (1.0 + np.exp(t))
            dpen_dval = -beta * eps * sig
            grad[i] += dpen_dval
            grad[2*n+i] += -dpen_dval
            obj += pen
        
        # Right boundary
        val = w - cx[i] - r[i]
        if val < eps:
            t = val / eps
            pen = beta * eps * eps * np.logaddexp(0, -t)
            sig = 1.0 / (1.0 + np.exp(t))
            dpen_dval = -beta * eps * sig
            grad[i] += -dpen_dval
            grad[2*n+i] += -dpen_dval
            obj += pen
        
        # Bottom boundary
        val = cy[i] - r[i]
        if val < eps:
            t = val / eps
            pen = beta * eps * eps * np.logaddexp(0, -t)
            sig = 1.0 / (1.0 + np.exp(t))
            dpen_dval = -beta * eps * sig
            grad[n+i] += dpen_dval
            grad[2*n+i] += -dpen_dval
            obj += pen
        
        # Top boundary
        val = h - cy[i] - r[i]
        if val < eps:
            t = val / eps
            pen = beta * eps * eps * np.logaddexp(0, -t)
            sig = 1.0 / (1.0 + np.exp(t))
            dpen_dval = -beta * eps * sig
            grad[n+i] += -dpen_dval
            grad[2*n+i] += -dpen_dval
            obj += pen
    
    # Overlap penalty (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            
            if dist_sq < (min_dist + eps)**2:
                dist = np.sqrt(dist_sq + 1e-12)
                gap = dist - min_dist
                if gap < eps:
                    t = -gap / eps
                    pen = beta * eps * eps * np.logaddexp(0, t)
                    sig = 1.0 / (1.0 + np.exp(-t))
                    dpen_dgap = -beta * eps * sig
                    ux = dx / dist
                    uy = dy / dist
                    grad[i] += dpen_dgap * (-ux)
                    grad[j] += dpen_dgap * ux
                    grad[n+i] += dpen_dgap * (-uy)
                    grad[n+j] += dpen_dgap * uy
                    grad[2*n+i] += dpen_dgap * (-1)
                    grad[2*n+j] += dpen_dgap * (-1)
                    obj += pen
    
    return obj, grad

def _penalty_optimize(circles, w, h, deadline):
    """
    Progressive penalty optimization with L-BFGS-B.
    """
    if time.time() > deadline - 5.0:
        return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-8, w))
        bounds.append((1e-8, h))
    for i in range(n):
        bounds.append((1e-8, min(w, h) / 2))

    # Progressive schedule
    epsilons = np.geomspace(0.01, 1e-7, 15)
    betas = np.geomspace(10, 5000, 15)
    x = x0.copy()
    
    for step in range(len(epsilons)):
        if time.time() > deadline - 3.0:
            break
        
        beta = betas[step]
        eps = epsilons[step]
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, w, h, beta, eps)[0],
                x,
                jac=lambda x: _objective_and_grad(x, w, h, beta, eps)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < 0 or res.success:
                x = res.x
        except Exception:
            pass
    
    new_circles = np.column_stack((x[:n], x[n:2*n], x[2*n:]))
    new_circles = _relax_circles_safe(new_circles, w, h, iterations=500)
    
    if _validate(new_circles, w, h, tol=1e-5):
        return new_circles
    
    return None

def _slsqp_polish(circles, w, h, deadline):
    """
    SLSQP polish with only near-contact constraints for speed.
    """
    if time.time() > deadline - 8.0:
        return None

    n = N
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[n+i] - x[2*n+i]})
    
    # Only add overlap constraints for pairs that are close
    cx = x0[:n]
    cy = x0[n:2*n]
    r = x0[2*n:]
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist * 1.5:  # Only near pairs
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sqrt((x[i]-x[j])**2 + (x[n+i]-x[n+j])**2) - x[2*n+i] - x[2*n+j]
                })
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 30, 'ftol': 1e-10}
        )
        if res.success:
            new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
            if _validate(new_circles, w, h, tol=1e-5):
                return new_circles
    except Exception:
        pass
    
    return None

def _try_aspect(ar, deadline):
    """Try a single aspect ratio and return best result."""
    w = ar / (1 + ar) * 2
    h = 1 / (1 + ar) * 2
    
    best_val = 0.0
    best_circles = None
    
    for (ww, hh) in [(w, h), (h, w)]:
        if time.time() > deadline - 10.0:
            break
        
        try:
            circles = _make_hexagonal_initial(ww, hh)
            circles = _relax_circles_safe(circles, ww, hh, iterations=500)
            
            if not _validate(circles, ww, hh):
                continue
            
            val = _total_radius(circles)
            
            if time.time() < deadline - 15.0:
                opt = _penalty_optimize(circles, ww, hh, deadline)
                if opt is not None:
                    circles = opt
                    val = _total_radius(circles)
            
            if time.time() < deadline - 8.0:
                polished = _slsqp_polish(circles, ww, hh, deadline)
                if polished is not None:
                    circles = polished
                    val = _total_radius(circles)
            
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
                
        except Exception:
            continue
    
    return best_circles, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_value = 0.0
    best_result = None
    
    # Targeted aspect ratios based on hexagonal packing geometry
    # For 5-4-5-4-3 rows, optimal aspect ratio is around 1.0-1.2
    ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75]
    
    for ar in ratios:
        if time.time() > deadline - 12.0:
            break
        
        circles, val = _try_aspect(ar, deadline)
        if circles is not None and val > best_value:
            best_value = val
            best_result = circles
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
