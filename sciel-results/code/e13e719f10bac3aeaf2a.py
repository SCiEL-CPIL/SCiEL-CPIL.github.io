"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Compute tensor error (sum of squared errors)
    def tensor_sse(U, V, W):
        rank = U.shape[1]
        sse = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    sse += (total - 1.0) ** 2
        return sse

    # Compute max absolute error
    def tensor_max_err(U, V, W):
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    np.random.seed(42)
    
    # Strategy: For each target rank from 41 down, try to find a valid decomposition
    # Use coordinate descent with continuous values, then snap to half-integers
    
    for target_rank in range(41, 30, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 15:
                break
            
            # Initialize: start from schoolbook and randomly perturb/merge
            # Strategy: take first target_rank columns of schoolbook, then add random perturbation
            if target_rank >= n * m * p - 5:
                # Start close to schoolbook: use first target_rank terms, zero out rest, then perturb
                U_c = np.zeros((n * m, target_rank))
                V_c = np.zeros((m * p, target_rank))
                W_c = np.zeros((p * n, target_rank))
                
                terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                for c in range(target_rank):
                    i, j, k = terms[c]
                    U_c[i * m + j, c] = 1.0
                    V_c[j * p + k, c] = 1.0
                    W_c[k * n + i, c] = 1.0
                
                # Add small random perturbation to all entries
                U_c += np.random.randn(n * m, target_rank) * 0.1
                V_c += np.random.randn(m * p, target_rank) * 0.1
                W_c += np.random.randn(p * n, target_rank) * 0.1
            else:
                # More aggressive: random initialization with small values
                U_c = np.random.randn(n * m, target_rank) * 0.5
                V_c = np.random.randn(m * p, target_rank) * 0.5
                W_c = np.random.randn(p * n, target_rank) * 0.5
            
            # Coordinate descent to minimize SSE
            best_sse = tensor_sse(U_c, V_c, W_c)
            
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Pick a random matrix and entry to update
                mat = np.random.choice(['U', 'V', 'W'])
                
                if mat == 'U':
                    row = np.random.randint(0, n * m)
                    col = np.random.randint(0, target_rank)
                    # Compute gradient for U[row, col]
                    # d/dU[row,col] SSE = 2 * sum_{i,j,k: i*m+j=row} (pred-1) * V[j*p+k,col] * W[k*n+i,col]
                    # For a given row, i and j are fixed: i = row // m, j = row % m
                    i_val = row // m
                    j_val = row % m
                    grad = 0.0
                    for k in range(p):
                        pred = np.sum(U_c[i_val * m + j_val] * V_c[j_val * p + k] * W_c[k * n + i_val])
                        grad += (pred - 1.0) * V_c[j_val * p + k, col] * W_c[k * n + i_val, col]
                    grad *= 2.0
                    
                    # Try a few step sizes
                    for step in [0.01, 0.005, 0.001]:
                        U_c[row, col] -= step * grad
                        new_sse = tensor_sse(U_c, V_c, W_c)
                        if new_sse < best_sse:
                            best_sse = new_sse
                            break
                        U_c[row, col] += step * grad
                
                elif mat == 'V':
                    row = np.random.randint(0, m * p)
                    col = np.random.randint(0, target_rank)
                    # row = j*p + k, so j = row // p, k = row % p
                    j_val = row // p
                    k_val = row % p
                    grad = 0.0
                    for i in range(n):
                        pred = np.sum(U_c[i * m + j_val] * V_c[j_val * p + k_val] * W_c[k_val * n + i])
                        grad += (pred - 1.0) * U_c[i * m + j_val, col] * W_c[k_val * n + i, col]
                    grad *= 2.0
                    
                    for step in [0.01, 0.005, 0.001]:
                        V_c[row, col] -= step * grad
                        new_sse = tensor_sse(U_c, V_c, W_c)
                        if new_sse < best_sse:
                            best_sse = new_sse
                            break
                        V_c[row, col] += step * grad
                
                else:
                    row = np.random.randint(0, p * n)
                    col = np.random.randint(0, target_rank)
                    # row = k*n + i, so k = row // n, i = row % n
                    k_val = row // n
                    i_val = row % n
                    grad = 0.0
                    for j in range(m):
                        pred = np.sum(U_c[i_val * m + j] * V_c[j * p + k_val] * W_c[k_val * n + i_val])
                        grad += (pred - 1.0) * U_c[i_val * m + j, col] * V_c[j * p + k_val, col]
                    grad *= 2.0
                    
                    for step in [0.01, 0.005, 0.001]:
                        W_c[row, col] -= step * grad
                        new_sse = tensor_sse(U_c, V_c, W_c)
                        if new_sse < best_sse:
                            best_sse = new_sse
                            break
                        W_c[row, col] += step * grad
            
            # Check if we found a valid solution
            if best_sse < 1e-12:
                if verify(U_c, V_c, W_c):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
                    found = True
                    break
            
            # Try to snap to half-integers and verify
            U_snap = np.round(U_c * 2.0) / 2.0
            V_snap = np.round(V_c * 2.0) / 2.0
            W_snap = np.round(W_c * 2.0) / 2.0
            if verify(U_snap, V_snap, W_snap):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
        else:
            # If we couldn't find this rank, try next lower rank
            # But if we're struggling, break early
            if target_rank <= 38:
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
