import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint, Bounds

N = 21
START = time.time()
DEADLINE = START + 145.0

def _time_ok():
    return time.time() < DEADLINE

def _generate_initial(W, H):
    """Generate a feasible initial layout."""
    # 5x5 grid
    kx, ky = 5, 5
    xs = []
    ys = []
    for j in range(ky):
        for i in range(kx):
            if len(xs) >= N:
                break
            x = (i + 0.5) * W / kx
            y = (j + 0.5) * H / ky
            xs.append(x)
            ys.append(y)
    while len(xs) < N:
        xs.append(W/2)
        ys.append(H/2)
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    # Small initial radii to ensure feasibility
    r = np.ones(N) * min(W, H) / 20.0
    return np.column_stack([xs, ys, r])

def _obj_flat(x_flat, W, H):
    r = x_flat[2::3]
    return -np.sum(r)

def _obj_grad_flat(x_flat, W, H):
    grad = np.zeros_like(x_flat)
    grad[2::3] = -1.0
    return grad

def _boundary_funcs(x_flat, W, H):
    cx = x_flat[0::3]
    cy = x_flat[1::3]
    r = x_flat[2::3]
    c1 = cx - r
    c2 = W - cx - r
    c3 = cy - r
    c4 = H - cy - r
    return np.concatenate([c1, c2, c3, c4])

def _boundary_jac(x_flat, W, H):
    jac = np.zeros((4*N, 3*N))
    idx = np.arange(N)
    jac[idx, idx] = 1.0
    jac[idx, 2 + idx] = -1.0
    jac[N + idx, idx] = -1.0
    jac[N + idx, 2 + idx] = -1.0
    jac[2*N + idx, 1 + idx] = 1.0
    jac[2*N + idx, 2 + idx] = -1.0
    jac[3*N + idx, 1 + idx] = -1.0
    jac[3*N + idx, 2 + idx] = -1.0
    return jac

def _overlap_funcs(x_flat, W, H):
    cx = x_flat[0::3]
    cy = x_flat[1::3]
    r = x_flat[2::3]
    vals = []
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dr = r[i] + r[j]
            vals.append(dx*dx + dy*dy - dr*dr)
    return np.array(vals)

def _overlap_jac(x_flat, W, H):
    cx = x_flat[0::3]
    cy = x_flat[1::3]
    r = x_flat[2::3]
    num_cons = N*(N-1)//2
    jac = np.zeros((num_cons, 3*N))
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dr = r[i] + r[j]
            jac[idx, i] = 2*dx
            jac[idx, 1+i] = 2*dy
            jac[idx, 2+i] = -2*dr
            jac[idx, j] = -2*dx
            jac[idx, 1+j] = -2*dy
            jac[idx, 2+j] = -2*dr
            idx += 1
    return jac

def _optimize_single(W, H, x0_flat, maxiter=200):
    bounds = Bounds(
        lb=np.zeros(3*N),
        ub=np.concatenate([np.full(N, W), np.full(N, H), np.full(N, min(W,H)/2.0)])
    )
    
    con_boundary = NonlinearConstraint(
        _boundary_funcs, 0, np.inf, args=(W, H), jac=_boundary_jac
    )
    con_overlap = NonlinearConstraint(
        _overlap_funcs, 0, np.inf, args=(W, H), jac=_overlap_jac
    )
    
    try:
        res = minimize(
            _obj_flat, x0_flat, method='trust-constr', 
            jac=_obj_grad_flat, 
            bounds=bounds, 
            constraints=[con_boundary, con_overlap],
            options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-8}
        )
        
        x_opt = res.x
        cx = x_opt[0::3]
        cy = x_opt[1::3]
        r = x_opt[2::3]
        
        # Validate
        valid = True
        for i in range(N):
            if r[i] < 0: valid = False; break
            if cx[i] < r[i] - 1e-6 or cx[i] > W - r[i] + 1e-6: valid = False; break
            if cy[i] < r[i] - 1e-6 or cy[i] > H - r[i] + 1e-6: valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d2 = (cx[i]-cx[j])**2 + (cy[i]-cy[j])**2
                    if d2 < (r[i]+r[j])**2 - 1e-6:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            return x_opt, -res.fun
        else:
            # Try to repair by shrinking radii slightly
            # This is a fallback, usually not needed
            return None, 0.0
    except Exception:
        return None, 0.0

def construct():
    best_score = 0.0
    best_out = None
    
    # Try square and near-square
    aspect_ratios = [1.0, 0.95, 1.05]
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        
        # Start 1: Grid
        x0_grid = _generate_initial(W, H)
        x0_flat = x0_grid.flatten()
        out, score = _optimize_single(W, H, x0_flat)
        if out is not None and score > best_score:
            best_score = score
            best_out = np.column_stack([out[0::3], out[1::3], out[2::3]])
            
        # Start 2: Random perturbation
        if _time_ok():
            x0_rand = x0_grid + np.random.randn(*x0_grid.shape) * 0.02
            x0_rand[:, 0] = np.clip(x0_rand[:, 0], 0, W)
            x0_rand[:, 1] = np.clip(x0_rand[:, 1], 0, H)
            x0_rand[:, 2] = np.clip(x0_rand[:, 2], 0.01, min(W,H)/2)
            out, score = _optimize_single(W, H, x0_rand.flatten())
            if out is not None and score > best_score:
                best_score = score
                best_out = np.column_stack([out[0::3], out[1::3], out[2::3]])
                
        # Start 3: Pure random
        if _time_ok():
            x0_rand2 = np.random.rand(N, 3)
            x0_rand2[:, 0] *= W
            x0_rand2[:, 1] *= H
            x0_rand2[:, 2] = np.ones(N) * min(W, H) / 20
            out, score = _optimize_single(W, H, x0_rand2.flatten())
            if out is not None and score > best_score:
                best_score = score
                best_out = np.column_stack([out[0::3], out[1::3], out[2::3]])

    if best_out is None:
        W, H = 1.0, 1.0
        best_out = _generate_initial(W, H)
        
    return best_out
