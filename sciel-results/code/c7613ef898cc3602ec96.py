import numpy as np
import time
from scipy.optimize import minimize

def _keich(n=64):
    """Classical Keich construction (n a power of 2): a good convex start."""
    k = int(round(np.log2(n)))
    j = np.arange(1, k + 1)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = list(bin(i)[2:])
        bits = [0] * (k - len(bits)) + bits
        eps = np.array(bits, dtype=np.float64)
        pos[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return pos

def _smooth_area_grad(x, n, M, a, y, w, eps):
    """
    Compute the smoothed union area and its gradient.
    Uses a smooth approximation for min(d, w).
    """
    L = x[None, :] + a[None, :] * y[:, None]
    order = np.argsort(L, axis=1)
    Ls = np.take_along_axis(L, order, axis=1)
    d = np.diff(Ls, axis=1)
    
    # Smooth min(d, w)
    diff = d - w[:, None]
    denom = np.sqrt(diff**2 + eps**2)
    # Derivative of smooth min w.r.t d
    # phi(d, w) = (d + w - sqrt((d-w)^2 + eps^2)) / 2
    # dphi/dd = 0.5 * (1 - (d-w)/sqrt((d-w)^2 + eps^2))
    grad_d = 0.5 * (1.0 - diff / denom)
    
    # Area is sum of smooth mins + w
    # We don't strictly need the area value for the gradient step if we use minimize, 
    # but minimize needs it.
    # area = (w + sum(phi)).mean()
    # phi = 0.5 * (d + w - denom)
    phi = 0.5 * (d + w[:, None] - denom)
    area = (w + phi.sum(axis=1)).mean()
    
    # Gradient w.r.t Ls
    # dU/dL_p = grad_d_{p-1} - grad_d_p
    gL = np.zeros((M, n), dtype=np.float64)
    gL[:, 1:] += grad_d
    gL[:, :n - 1] -= grad_d
    
    # Scatter back to x
    gx = np.zeros((M, n), dtype=np.float64)
    np.put_along_axis(gx, order, gL, axis=1)
    grad = gx.mean(axis=0)
    
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    t0 = time.time()
    deadline = t0 + 600.0
    n = 64
    M = 20000
    a = (np.arange(n) + 1) / n
    m = np.arange(M)
    y = (m + 0.5) / M
    w = (1.0 - y) / n
    
    # Precompute constants for the objective function closure
    # We will use a sequence of eps values, decreasing from coarse to fine.
    eps_schedule = [0.5, 0.1, 0.01, 0.001, 0.0001]
    
    x_best = _keich(n)
    area_best = float('inf')
    
    # Evaluate initial best
    area_init, _ = _smooth_area_grad(x_best, n, M, a, y, w, eps_schedule[0])
    area_best = area_init
    
    for eps in eps_schedule:
        if time.time() > deadline - 30.0:
            break
            
        def fun_grad(x):
            area, grad = _smooth_area_grad(x, n, M, a, y, w, eps)
            return area, grad
            
        try:
            res = minimize(fun_grad, x_best, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                x_candidate = res.x
                area_candidate = res.fun
                if area_candidate < area_best:
                    area_best = area_candidate
                    x_best = x_candidate
        except Exception:
            # If minimize fails, keep current x_best
            pass
            
    # Final polish with a very small eps or just return the best found
    # The last eps was 1e-4, which is quite small.
    # We can do one more step with eps=1e-6 if time permits.
    if time.time() < deadline - 10.0:
        def fun_grad_final(x):
            area, grad = _smooth_area_grad(x, n, M, a, y, w, 1e-6)
            return area, grad
            
        try:
            res = minimize(fun_grad_final, x_best, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                x_candidate = res.x
                area_candidate = res.fun
                if area_candidate < area_best:
                    area_best = area_candidate
                    x_best = x_candidate
        except Exception:
            pass
            
    # Validate
    # Use the original non-smooth area for validation to ensure it's valid
    # But the smooth area is a lower bound? No, it's an approximation.
    # Actually, smooth min is <= true min?
    # min(a,b) is convex. The smooth approximation is also convex and lies below?
    # Let's check: min(0, 1) = 0. Smooth: (0+1-sqrt(1+eps^2))/2 < 0.5? No.
    # If d < w, min=d. Smooth: (d+w - sqrt((d-w)^2+eps^2))/2.
    # If d=0, w=1, eps=0.1: (1 - sqrt(1.01))/2 = (1-1.005)/2 = -0.0025.
    # So smooth area < true area.
    # So minimizing smooth area might lead to a point where true area is higher?
    # No, the true area is the limit as eps->0.
    # So the smooth area is a lower bound on the true area?
    # Wait, if smooth < true, then minimizing smooth might push us to a place where the gap between smooth and true is large?
    # However, as eps -> 0, they converge.
    # The validation should use the true area to be safe.
    
    def _true_area(x):
        L = x[None, :] + a[None, :] * y[:, None]
        order = np.argsort(L, axis=1)
        Ls = np.take_along_axis(L, order, axis=1)
        d = np.diff(Ls, axis=1)
        mn = np.minimum(d, w[:, None])
        return (w + mn.sum(axis=1)).mean()

    fa = _true_area(x_best)
    if (not np.isfinite(fa)) or fa <= 0 or (not np.all(np.isfinite(x_best))):
        x_best = _keich(n)
        
    return x_best.astype(np.float64)
