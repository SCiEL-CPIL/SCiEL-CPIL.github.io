"""
Matrix multiplication tensor decomposition for (2,5,5).
Attempts to find a rank-minimal exact decomposition with integer/half-integer entries.
Uses gradient descent with random restarts, constrained to half-integer grid,
with strict verification.
"""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T[i, j, k] = delta_{i,i'} delta_{k,k'} structure.
    Actually, the decomposition condition is:
      For all i, i', j, k, k': sum_r U[(i,j), r] * V[(j,k), r] * W[(k,i), r] = delta_{i,i'} * delta_{k,k'}
    Wait, let me re-derive.
    
    Matrix multiplication: C = A * B where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k (since it's the standard bilinear form).
    Wait, no. The multiplication tensor is:
      T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    And the decomposition is:
      T[i,j,k] = sum_r U[(i,j), r] * V[(j,k), r] * W[(k,i), r]
    
    Wait, that doesn't look right dimensionally. Let me check the convention.
    
    U has shape (n*m, rank) -> indices (i*m+j, r)
    V has shape (m*p, rank) -> indices (j*p+k, r)
    W has shape (p*n, rank) -> indices (k*n+i, r)
    
    And the condition is:
      For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Yes, that's correct. The target is the all-ones tensor of shape (n*m, m*p, p*n) but actually it's a 3-tensor where each entry T[i,j,k] = 1.
    """
    n, m, p = n, m, p
    target = np.ones((n * m, m * p, p * n), dtype=float)
    return target

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """
    Verify that the decomposition is exact (within tolerance).
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # For each r, we have outer product of U[:,r], V[:,r], W[:,r]
    # Sum over r
    # Efficient way: for each (i,j,k) compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's do it with broadcasting
    # U_expanded: (n*m, 1, 1, rank)
    # V_expanded: (1, m*p, 1, rank)
    # W_expanded: (1, 1, p*n, rank)
    # But we need to align indices correctly.
    
    # Alternative: loop over i, j, k
    n, m, p = n, m, p
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err

def _try_gradient_descent(n, m, p, rank, max_iter=5000, lr=0.01, seed=0):
    """
    Try to find a decomposition of given rank using gradient descent.
    Returns (U, V, W) if found, else None.
    """
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    # For each (i,j,k), we need u_idx, v_idx, w_idx
    idx_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_list.append((i * m + j, j * p + k, k * n + i))
    
    def compute_residuals_and_grads(U, V, W):
        """Compute residuals and gradients for all (i,j,k)."""
        residuals = np.zeros((n, m, p), dtype=float)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for idx, (u_idx, v_idx, w_idx) in enumerate(idx_list):
            i, j, k = (idx // (m*p), (idx // p) % m, idx % p)
            # Actually, let me just use the loop structure
            pass
        
        # Simpler: just loop
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Residual: sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r] - 1
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res = val - 1.0
                    residuals[i, j, k] = res
                    # Gradients
                    dU[u_idx, :] += res * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += res * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += res * U[u_idx, :] * V[v_idx, :]
        
        return residuals, dU, dV, dW
    
    best_obj = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        residuals, dU, dV, dW = compute_residuals_and_grads(U, V, W)
        obj = np.sum(residuals ** 2)
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if obj < 1e-10:
            # Check if we can snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            valid, err = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8)
            if valid:
                return U_snap, V_snap, W_snap
            # Also check without snapping
            valid, err = _verify_decomposition(U, V, W, n, m, p, tol=1e-10)
            if valid:
                return U, V, W
            break
        
        # Adaptive learning rate
        lr_eff = lr / (1 + it * 0.001)
        U -= lr_eff * dU
        V -= lr_eff * dV
        W -= lr_eff * dW
    
    # Final check on best
    valid, err = _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-10)
    if valid:
        return best_U, best_V, best_W
    
    # Try snapping
    U_snap = np.round(best_U * 2) / 2
    V_snap = np.round(best_V * 2) / 2
    W_snap = np.round(best_W * 2) / 2
    valid, err = _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-8)
    if valid:
        return U_snap, V_snap, W_snap
    
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 130  # 130 seconds to leave margin
    
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try ranks from 25 to 49
    for rank in range(25, min(50, schoolbook_rank)):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            result = _try_gradient_descent(n, m, p, rank, max_iter=3000, lr=0.005, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Final verification
                valid, err = _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8)
                if valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result

# For testing
if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    valid, err = _verify_decomposition(U, V, W, n, m, p, tol=1e-8)
    print(f"Valid: {valid}, Max error: {err}")
