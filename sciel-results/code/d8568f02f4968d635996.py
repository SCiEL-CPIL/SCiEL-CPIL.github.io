# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_area_fast(positions, n=64, M=20000):
    """Compute area of union of triangles using vectorized quadrature."""
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    a = (np.arange(n) + 1) / n     # shape (n,)
    w = (1.0 - y) / n              # shape (M,)
    
    # L[i, m] = x_i + a_i * y_m, shape (n, M)
    L = positions[:, np.newaxis] + a[np.newaxis, :] * y[np.newaxis, :]
    
    # Sort L along axis 0 for each column
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Union length at each height: w + sum of min(gap, w) for consecutive pairs
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    union_len = w[np.newaxis, :] + np.sum(np.minimum(gaps, w[np.newaxis, :]), axis=0)
    
    area = np.mean(union_len)
    return area

def _subgradient(positions, n=64, M=20000):
    """Compute subgradient of area with respect to positions."""
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1.0 - y) / n
    
    L = positions[:, np.newaxis] + a[np.newaxis, :] * y[np.newaxis, :]
    
    # Sort and get permutation
    # perm[k, m] is the index in positions of the k-th smallest L value at height m
    perm = np.argsort(L, axis=0)  # (n, M)
    L_sorted = L[perm, np.arange(M)[np.newaxis, :]] # (n, M)
    
    # Gaps between sorted values
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    
    # Active flags: A[k, m] is 1 if gap k at height m is less than w_m
    A = (gaps < w[np.newaxis, :]).astype(np.float64)  # (n-1, M)
    
    # Construct D[r, m] = A[r, m] - A[r-1, m]
    # D has shape (n, M)
    D = np.zeros((n, M))
    
    # D[0] = -A[0] (since A[-1] is 0)
    D[0, :] = -A[0, :]
    
    # D[n-1] = A[n-2] (since A[n-1] is 0)
    D[n-1, :] = A[n-2, :]
    
    # D[1:n-1] = A[1:n-1] - A[0:n-2]
    if n > 2:
        D[1:-1, :] = A[1:, :] - A[:-1, :]
        
    # Scatter D back to original indices
    # We want grad_orig[i, m] = D[rank_of_i, m]
    # perm[k, m] = i  => i is at rank k
    # So we want to assign D[k, m] to grad_orig[perm[k, m], m]
    grad_orig = np.zeros((n, M))
    
    # Use advanced indexing to scatter
    # grad_orig[perm, np.arange(M)[np.newaxis, :]] = D
    # Check shapes:
    # perm: (n, M)
    # np.arange(M)[np.newaxis, :]: (1, M) -> broadcasts to (n, M)
    # D: (n, M)
    grad_orig[perm, np.arange(M)[np.newaxis, :]] = D
    
    # The subgradient w.r.t. x_i is a_i * mean(grad_orig[i, m])
    subgrad = a * np.mean(grad_orig, axis=1)
    
    return subgrad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    
    # Start with Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    positions = np.array(positions, dtype=np.float64)
    
    start_area = _compute_area_fast(positions, n, M)
    
    t0 = time.time()
    deadline = t0 + 580  # 580s to be safe (630s limit)
    
    best_positions = positions.copy()
    best_area = start_area
    
    # Gradient Descent
    lr = 0.01
    for step in range(2000):
        if time.time() > deadline:
            break
            
        subgrad = _subgradient(positions, n, M)
        positions = positions - lr * subgrad
        
        area = _compute_area_fast(positions, n, M)
        
        if area < best_area:
            best_area = area
            best_positions = positions.copy()
            
        # Adaptive learning rate
        if step % 100 == 0 and step > 0:
            lr *= 0.95
        if lr < 1e-6:
            break
            
    return best_positions

# EVOLVE-BLOCK-END
