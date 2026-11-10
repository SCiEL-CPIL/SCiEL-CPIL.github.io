import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Generate Keich positions as initial guess
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        bits = [int(b) for b in bin(i)[2:]]
        bits = [0] * (k - len(bits)) + bits
        eps_seed = np.array(bits, dtype=float)
        pos = np.sum((1 - j) * eps_seed * (2.0 ** -j) / k)
        positions.append(pos)
    x0 = np.array(positions, dtype=np.float64)
    
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    
    def compute_area_and_grad(x):
        L = x[None, :] + a[None, :] * y[:, None]
        
        order = np.argsort(L, axis=1)
        L_sorted = np.take_along_axis(L, order, axis=1)
        
        diffs = np.diff(L_sorted, axis=1)
        mins = np.minimum(diffs, w[:, None])
        union_lengths = w + np.sum(mins, axis=1)
        area = np.mean(union_lengths)
        
        active = diffs < w[:, None]
        
        grad = np.zeros((M, n))
        m_idx, j_idx = np.where(active)
        if len(m_idx) > 0:
            left_tri = order[m_idx, j_idx]
            right_tri = order[m_idx, j_idx + 1]
            np.add.at(grad, (m_idx, left_tri), -1.0)
            np.add.at(grad, (m_idx, right_tri), 1.0)
        
        grad_x = np.mean(grad, axis=0)
        
        return area, grad_x
    
    start_time = time.time()
    deadline = start_time + 590
    
    best_x = x0.copy()
    best_area = compute_area_and_grad(x0)[0]
    
    try:
        res = minimize(
            lambda x: compute_area_and_grad(x)[0],
            x0,
            jac=lambda x: compute_area_and_grad(x)[1],
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 20,
            }
        )
        
        if res.fun < best_area:
            best_x = res.x
            best_area = res.fun
        
        elapsed = time.time() - start_time
        if elapsed < deadline:
            res2 = minimize(
                lambda x: compute_area_and_grad(x)[0],
                best_x,
                jac=lambda x: compute_area_and_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxls': 10,
                }
            )
            if res2.fun < best_area:
                best_x = res2.x
                best_area = res2.fun
        
    except Exception:
        pass
    
    return best_x

# EVOLVE-BLOCK-END
