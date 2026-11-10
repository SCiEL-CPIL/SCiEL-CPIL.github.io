# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_area_and_grad(x):
    n = 64
    M = 20000
    a = np.arange(1, n + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    # L shape (M, n)
    # L[m, i] = x[i] + a[i] * y[m]
    L = x[None, :] + a[None, :] * y[:, None]
    
    # Sort L along axis 1
    # indices shape (M, n)
    sort_idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, sort_idx, axis=1)
    
    # d shape (M, n-1)
    d = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # Union length U shape (M,)
    # U = w + sum(min(d, w))
    # min(d, w) where w is broadcast
    w_col = w[:, None]
    d_clipped = np.minimum(d, w_col)
    U = w + np.sum(d_clipped, axis=1)
    
    area = np.mean(U)
    
    # Gradient
    # g shape (M, n)
    g = np.zeros_like(L)
    # Condition d < w
    mask = d < w_col
    # For each j in 0..n-2:
    # if mask[m, j]:
    #   g[m, sort_idx[m, j]] -= 1
    #   g[m, sort_idx[m, j+1]] += 1
    
    # We can use np.add.at or scatter
    # Indices for subtraction: sort_idx[:, :-1] where mask is True
    # Indices for addition: sort_idx[:, 1:] where mask is True
    
    # Flatten indices
    m_indices = np.where(mask)[0]
    j_indices = np.where(mask)[1]
    
    # Subtract from sort_idx[m, j]
    # Add to sort_idx[m, j+1]
    
    # np.add.at(g, (m_indices, sort_idx[m_indices, j_indices]), -1)
    # np.add.at(g, (m_indices, sort_idx[m_indices, j_indices + 1]), 1)
    
    # This might be slow in Python loop, but vectorized add.at is okay.
    # However, M=20000, n=64. mask has ~ M*n/2 true values?
    # Actually, if triangles are packed, d is small, so mask is mostly True.
    # Let's try to be efficient.
    
    # Alternative: 
    # grad_L = np.zeros_like(L)
    # np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)
    # np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)
    
    # Then grad_x[i] = sum_m grad_L[m, i] / M
    # grad_x = np.sum(grad_L, axis=0) / M
    
    # Let's implement this carefully.
    grad_L = np.zeros_like(L)
    np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)
    np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)
    
    grad_x = np.sum(grad_L, axis=0) / M
    
    return area, grad_x

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    # Optimize
    start_time = time.time()
    deadline = start_time + 590 # 10s margin
    
    result = minimize(
        lambda x: _compute_area_and_grad(x)[0],
        x0,
        jac=lambda x: _compute_area_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 1000, 'maxfun': 10000, 'disp': False}
    )
    
    # Check time
    if time.time() > deadline:
         # Return best found so far? minimize returns the best.
         pass
         
    return result.x

# EVOLVE-BLOCK-END
