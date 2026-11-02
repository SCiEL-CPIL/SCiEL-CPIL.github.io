"""Matrix multiplication 3,3,8 - rank-minimal decomposition search."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _is_valid_number(x):
    """Check if x is an integer or half-integer within tolerance."""
    # Check if close to integer or half-integer
    nearest_int = round(x)
    nearest_half = round(x * 2) / 2.0
    return abs(x - nearest_int) < 1e-6 or abs(x - nearest_half) < 1e-6

def _verify(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify that U, V, W form a valid decomposition of the matmul tensor.
    
    Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all valid (i,j,k).
    Also checks that entries are integers or half-integers.
    """
    U_f = np.asarray(U, dtype=np.float64)
    V_f = np.asarray(V, dtype=np.float64)
    W_f = np.asarray(W, dtype=np.float64)
    
    # Check type constraints
    for mat in [U_f, V_f, W_f]:
        if not np.all([_is_valid_number(x) for row in mat for x in row]):
            return False
            
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_factorization(n, m, p, target_rank, max_iter=500, deadline=None):
    """
    Try to find a rank-target_rank decomposition using multiplicative updates.
    
    The constraint is: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    We can view this as: for each (i,j,k), the vector product of the three corresponding
    rows across all r must sum to 1.
    
    Strategy: Initialize randomly, then iteratively update each factor to reduce the error.
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Precompute index mappings
    # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i*m+j, j*p+k, k*n+i))
    
    num_constraints = len(indices)
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n_m, target_rank) * 0.5
    V = np.random.randn(m_p, target_rank) * 0.5
    W = np.random.randn(p_n, target_rank) * 0.5
    
    # Normalize to avoid trivial scaling issues
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-10)
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-10)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute current error
        errors = np.zeros(num_constraints)
        for idx, (ui, vi, wi) in enumerate(indices):
            errors[idx] = np.dot(U[ui, :], V[vi, :]) * 1.0  # This is wrong, need element-wise product sum
            # Actually: sum_r U[ui,r] * V[vi,r] * W[wi,r]
            errors[idx] = np.sum(U[ui, :] * V[vi, :] * W[wi, :])
        
        err = np.max(np.abs(errors - 1.0))
        
        # Snap to nearest half-integer for validity check
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        # Calculate error on snapped values
        errors_snapped = np.zeros(num_constraints)
        for idx, (ui, vi, wi) in enumerate(indices):
            errors_snapped[idx] = np.sum(U_snap[ui, :] * V_snap[vi, :] * W_snap[wi, :])
        err_snapped = np.max(np.abs(errors_snapped - 1.0))
        
        if err_snapped < 1e-6:
            return U_snap.copy(), V_snap.copy(), W_snap.copy(), err_snapped
        
        if err_snapped < best_error:
            best_error = err_snapped
            best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
        
        # Multiplicative update: for each constraint, adjust factors
        # Simple approach: gradient descent on the error
        # E = sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for idx, (ui, vi, wi) in enumerate(indices):
            residual = errors[idx] - 1.0
            # dE/dU[ui,r] = 2 * residual * V[vi,r] * W[wi,r]
            dU[ui, :] += 2 * residual * V[vi, :] * W[wi, :]
            dV[vi, :] += 2 * residual * U[ui, :] * W[wi, :]
            dW[wi, :] += 2 * residual * U[ui, :] * V[vi, :]
        
        # Gradient step with adaptive learning rate
        lr = 0.1 / (1.0 + it * 0.01)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Keep values bounded
        U = np.clip(U, -10, 10)
        V = np.clip(V, -10, 10)
        W = np.clip(W, -10, 10)
    
    if best_error < 1e-6:
        return best_U, best_V, best_W, best_error
    return None

def _try_structured(n, m, p, target_rank, deadline=None):
    """
    Try a structured decomposition approach.
    
    For the matmul tensor T[i,j,k] = 1 (in the flattened sense), we can try to
    exploit the Kronecker product structure.
    
    One idea: if we can write the all-ones pattern as a sum of rank-1 terms
    that respect the index structure, we might find a lower rank.
    
    For 3x3*3x8: n=3, m=3, p=8
    U has shape (9, r), V has shape (24, r), W has shape (24, r)
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Try different structured initializations
    np.random.seed(123)
    
    # Strategy 1: Block diagonal structure
    # Group by i (rows of A/C)
    for strategy in range(5):
        if deadline and time.time() > deadline:
            break
        
        U = np.zeros((n_m, target_rank))
        V = np.zeros((m_p, target_rank))
        W = np.zeros((p_n, target_rank))
        
        if strategy == 0:
            # Random
            U = np.random.randn(n_m, target_rank) * 0.3
            V = np.random.randn(m_p, target_rank) * 0.3
            W = np.random.randn(p_n, target_rank) * 0.3
        elif strategy == 1:
            # Block structure by i
            per_i = target_rank // n
            for i in range(n):
                start = i * per_i
                end = start + per_i if i < n-1 else target_rank
                for j in range(m):
                    U[i*m+j, start:end] = 1.0
                for k in range(p):
                    W[k*n+i, start:end] = 1.0
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, start:end] = 1.0
        elif strategy == 2:
            # Fourier-like structure
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = np.cos(2*np.pi*r*i/n + np.pi*r*j/m)
                for j in range(m):
                    for k in range(p):
                        V[j*p+k, r] = np.cos(2*np.pi*r*j/m + np.pi*r*k/p)
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, r] = np.cos(2*np.pi*r*k/p + np.pi*r*i/n)
        elif strategy == 3:
            # Sparse random
            U = (np.random.randn(n_m, target_rank) > 0.5).astype(float) * 0.5
            V = (np.random.randn(m_p, target_rank) > 0.5).astype(float) * 0.5
            W = (np.random.randn(p_n, target_rank) > 0.5).astype(float) * 0.5
        else:
            # Mixed
            U = np.random.randn(n_m, target_rank) * 0.2
            V = np.random.randn(m_p, target_rank) * 0.2
            W = np.random.randn(p_n, target_rank) * 0.2
            # Add some structure
            for i in range(n):
                U[i*m:(i+1)*m, :target_rank//n] = 1.0
                W[i:(i+1)*p*n//n, :target_rank//n] = 0.5
        
        # Run optimization
        result = _try_factorization(n, m, p, target_rank, max_iter=200, deadline=deadline)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    
    # Try to find better decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But realistically, for matmul, the rank is close to n*m*p
    
    # Try a range of ranks from lower to schoolbook
    ranks_to_try = list(range(30, best_rank + 1, 2))
    # Ensure we try some key values
    for r in [32, 40, 48, 56, 64, 70]:
        if r not in ranks_to_try and r < best_rank:
            ranks_to_try.append(r)
    ranks_to_try.sort()
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try structured approach
        result = _try_structured(n, m, p, target_rank, deadline=deadline)
        
        if result is not None:
            U, V, W, err = result
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, target_rank, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook as fallback
    # Ensure types are integers (0 or 1)
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
