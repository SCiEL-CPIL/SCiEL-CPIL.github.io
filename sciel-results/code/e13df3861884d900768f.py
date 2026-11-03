"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21


def _hex_init(w, h):
    """Generate hexagonal packing initialization in rectangle w x h."""
    area = w * h
    r_est = np.sqrt(area * 0.9069 / (N * np.pi))
    r_est = min(r_est, w / 4, h / 4)
    
    points = []
    dx = 2.0 * r_est
    dy = np.sqrt(3) * r_est
    
    y = r_est
    row_idx = 0
    while y <= h - r_est + 1e-9:
        x = r_est + (row_idx % 2) * (dx / 2)
        while x <= w - r_est + 1e-9:
            points.append((x, y))
            x += dx
        y += dy
        row_idx += 1
    
    if len(points) < N:
        side = int(np.ceil(np.sqrt(N)))
        points = [((i + 0.5) / side * w, (j + 0.5) / side * h) for i in range(side) for j in range(side)]
        points = points[:N]
        r_est = 0.5 / side * min(w, h)
    else:
        points = points[:N]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        points = [(p[0] + w/2 - cx, p[1] + h/2 - cy) for p in points]
        points = [(max(r_est*0.9, min(p[0], w - r_est*0.9)), max(r_est*0.9, min(p[1], h - r_est*0.9))) for p in points]
    
    return np.array(points), r_est


def _optimize_packing(w, h, x, y, r, deadline):
    """Optimize circle positions and radii for given rectangle."""
    h = 2.0 - w
    
    lb = np.concatenate([np.zeros(N), np.zeros(N), np.full(N, 1e-8)])
    ub = np.concatenate([np.full(N, w), np.full(N, h), np.full(N, min(w, h)/2)])
    
    def obj_and_grad(z):
        xi = z[:N]
        yi = z[N:2*N]
        ri = z[2*N:3*N]
        
        obj = -np.sum(ri) + 0.005 * np.var(ri)
        
        # Overlap penalty
        overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xi[i]-xi[j])**2 + (yi[i]-yi[j])**2 + 1e-16)
                v = max(0, ri[i] + ri[j] - d)
                overlap += v**2
        obj += 100.0 * overlap
        
        # Boundary penalty
        bpen = 0.0
        for i in range(N):
            bpen += max(0, ri[i] - xi[i])**2
            bpen += max(0, ri[i] - (w - xi[i]))**2
            bpen += max(0, ri[i] - yi[i])**2
            bpen += max(0, ri[i] - (h - yi[i]))**2
        obj += 100.0 * bpen
        
        grad = np.zeros(3*N)
        grad[2*N:3*N] = -1.0 + 0.01/N * (ri - np.mean(ri))
        
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xi[i]-xi[j])**2 + (yi[i]-yi[j])**2 + 1e-16)
                v = max(0, ri[i] + ri[j] - d)
                if v > 0:
                    gx = (xi[i]-xi[j]) / d
                    gy = (yi[i]-yi[j]) / d
                    grad[2*N+i] += 200.0 * v
                    grad[2*N+j] += 200.0 * v
                    grad[i] += 200.0 * v * (-gx)
                    grad[j] += 200.0 * v * gx
                    grad[N+i] += 200.0 * v * (-gy)
                    grad[N+j] += 200.0 * v * gy
            
            # Boundary gradients
            if ri[i] - xi[i] > 0:
                grad[2*N+i] += 200.0 * (ri[i] - xi[i])
                grad[i] -= 200.0 * (ri[i] - xi[i])
            if ri[i] - (w - xi[i]) > 0:
                grad[2*N+i] += 200.0 * (ri[i] - (w - xi[i]))
                grad[i] += 200.0 * (ri[i] - (w - xi[i]))
            if ri[i] - yi[i] > 0:
                grad[2*N+i] += 200.0 * (ri[i] - yi[i])
                grad[N+i] -= 200.0 * (ri[i] - yi[i])
            if ri[i] - (h - yi[i]) > 0:
                grad[2*N+i] += 200.0 * (ri[i] - (h - yi[i]))
                grad[N+i] += 200.0 * (ri[i] - (h - yi[i]))
        
        return obj, grad
    
    z = np.concatenate([x, y, r])
    z = np.clip(z, lb, ub)
    
    for maxiter in [200, 400, 600]:
        if time.time() > deadline:
            break
        try:
            res = minimize(obj_and_grad, z, method='L-BFGS-B', jac=True,
                          bounds=list(zip(lb, ub)),
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            z = res.x
        except Exception:
            break
    
    xi = z[:N]
    yi = z[N:2*N]
    ri = z[2*N:3*N].copy()
    
    # Fix violations
    for _ in range(200):
        changed = False
        for i in range(N):
            ri[i] = min(ri[i], xi[i], w - xi[i], yi[i], h - yi[i])
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xi[i]-xi[j])**2 + (yi[i]-yi[j])**2)
                if d < ri[i] + ri[j]:
                    excess = ri[i] + ri[j] - d
                    ri[i] -= excess/2
                    ri[j] -= excess/2
                    changed = True
        ri = np.maximum(ri, 1e-10)
        if not changed:
            break
    
    # Validate
    valid = True
    for i in range(N):
        if ri[i] > xi[i] + 1e-6 or ri[i] > w - xi[i] + 1e-6 or ri[i] > yi[i] + 1e-6 or ri[i] > h - yi[i] + 1e-6:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xi[i]-xi[j])**2 + (yi[i]-yi[j])**2)
                if d < ri[i] + ri[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        return np.column_stack([xi, yi, ri]), np.sum(ri)
    return None, -np.inf


def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    np.random.seed(42)
    
    best_result = None
    best_score = -np.inf
    
    for w in np.linspace(0.7, 1.3, 11):
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        points, r_est = _hex_init(w, h)
        x, y = points[:, 0], points[:, 1]
        r = np.full(N, r_est)
        
        result, score = _optimize_packing(w, h, x, y, r, deadline)
        if result is not None and score > best_score:
            best_score = score
            best_result = result
        
        for trial in range(2):
            if time.time() > deadline:
                break
            x_pert = x + np.random.normal(0, 0.05 * r_est, N)
            y_pert = y + np.random.normal(0, 0.05 * r_est, N)
            x_pert = np.clip(x_pert, r_est*0.5, w - r_est*0.5)
            y_pert = np.clip(y_pert, r_est*0.5, h - r_est*0.5)
            
            result, score = _optimize_packing(w, h, x_pert, y_pert, r, deadline)
            if result is not None and score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
