"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check consistent indices: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check inconsistent j indices
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
                    if abs(s) > tol:
                        return False
    
    # Check inconsistent k indices
    for j in range(m):
        for k in range(p):
            for k2 in range(p):
                if k == k2:
                    continue
                for i in range(n):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k2*n+i, :])
                    if abs(s) > tol:
                        return False
    
    # Check inconsistent i indices
    for i in range(n):
        for i2 in range(n):
            if i == i2:
                continue
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i2, :])
                    if abs(s) > tol:
                        return False
    
    return True

def _error(U, V, W, n=3, m=4, p=8):
    """Compute the total squared error of the decomposition."""
    err = 0.0
    rank = U.shape[1]
    # Consistent terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                err += (s - 1.0) ** 2
    # Inconsistent j
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2:
                    continue
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
                    err += s ** 2
    # Inconsistent k
    for j in range(m):
        for k in range(p):
            for k2 in range(p):
                if k == k2:
                    continue
                for i in range(n):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k2*n+i, :])
                    err += s ** 2
    # Inconsistent i
    for i in range(n):
        for i2 in range(n):
            if i == i2:
                continue
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i2, :])
                    err += s ** 2
    return err

def _solve_for_U(U, V, W, n, m, p, target_rank):
    """Update U using least squares with full constraints."""
    for i in range(n):
        for j in range(m):
            rows = []
            rhs = []
            # Consistent: for each k, want 1
            for k in range(p):
                rows.append(V[j*p+k, :] * W[k*n+i, :])
                rhs.append(1.0)
            # Inconsistent j2 != j
            for j2 in range(m):
                if j2 == j:
                    continue
                for k in range(p):
                    rows.append(V[j2*p+k, :] * W[k*n+i, :])
                    rhs.append(0.0)
            # Inconsistent i2 != i
            for i2 in range(n):
                if i2 == i:
                    continue
                for k in range(p):
                    rows.append(V[j*p+k, :] * W[k*n+i2, :])
                    rhs.append(0.0)
            A = np.array(rows)
            b = np.array(rhs)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = sol
            except:
                pass
    return U

def _solve_for_V(U, V, W, n, m, p, target_rank):
    """Update V using least squares with full constraints."""
    for j in range(m):
        for k in range(p):
            rows = []
            rhs = []
            # Consistent: for each i, want 1
            for i in range(n):
                rows.append(U[i*m+j, :] * W[k*n+i, :])
                rhs.append(1.0)
            # Inconsistent j2 != j
            for j2 in range(m):
                if j2 == j:
                    continue
                for i in range(n):
                    rows.append(U[i*m+j2, :] * W[k*n+i, :])
                    rhs.append(0.0)
            # Inconsistent k2 != k
            for k2 in range(p):
                if k2 == k:
                    continue
                for i in range(n):
                    rows.append(U[i*m+j, :] * W[k2*n+i, :])
                    rhs.append(0.0)
            A = np.array(rows)
            b = np.array(rhs)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = sol
            except:
                pass
    return V

def _solve_for_W(U, V, W, n, m, p, target_rank):
    """Update W using least squares with full constraints."""
    for k in range(p):
        for i in range(n):
            rows = []
            rhs = []
            # Consistent: for each j, want 1
            for j in range(m):
                rows.append(U[i*m+j, :] * V[j*p+k, :])
                rhs.append(1.0)
            # Inconsistent k2 != k
            for k2 in range(p):
                if k2 == k:
                    continue
                for j in range(m):
                    rows.append(U[i*m+j, :] * V[j*p+k2, :])
                    rhs.append(0.0)
            # Inconsistent i2 != i
            for i2 in range(n):
                if i2 == i:
                    continue
                for j in range(m):
                    rows.append(U[i2*m+j, :] * V[j*p+k, :])
                    rhs.append(0.0)
            A = np.array(rows)
            b = np.array(rhs)
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = sol
            except:
                pass
    return W

def construct():
    n, m, p = 3, 4, 8
    t_start = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Schoolbook decomposition (rank 96) as fallback
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0), dtype=float)
    V0 = np.zeros((m * p, rank0), dtype=float)
    W0 = np.zeros((p * n, rank0), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    best_rank = rank0
    
    # Try various target ranks
    # Theoretical lower bound: max(n*m, m*p, p*n) = 32
    # But realistically, we can probably get down to around 60-80
    target_ranks = list(range(85, 97)) + list(range(70, 85)) + list(range(55, 70)) + list(range(40, 55))
    
    rng = np.random.RandomState(12345)
    
    for target_rank in target_ranks:
        if time.time() - t_start > time_limit - 10:
            break
        if best_rank <= target_rank:
            continue
        
        # Number of attempts based on rank
        n_attempts = 15 if target_rank >= 80 else 8
        if time.time() - t_start > time_limit - 40:
            n_attempts = max(1, n_attempts // 2)
        
        for attempt in range(n_attempts):
            if time.time() - t_start > time_limit - 10:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = rng.randn(n * m, target_rank) * scale
            V = rng.randn(m * p, target_rank) * scale
            W = rng.randn(p * n, target_rank) * scale
            
            # Alternating least squares iterations
            max_iters = 50
            for it in range(max_iters):
                if time.time() - t_start > time_limit - 10:
                    break
                
                U = _solve_for_U(U, V, W, n, m, p, target_rank)
                V = _solve_for_V(U, V, W, n, m, p, target_rank)
                W = _solve_for_W(U, V, W, n, m, p, target_rank)
                
                # Check error for early stopping
                if it % 5 == 0:
                    err = _error(U, V, W, n, m, p)
                    if err < 1e-8:
                        break
            
            # Final check with tight tolerance
            if _verify(U, V, W, n, m, p, tol=1e-6):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    U, V, W = U_r, V_r, W_r
                
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    print(f"Found valid decomposition with rank {target_rank}")
                    # If we found a low rank, we can break early
                    if target_rank <= 60:
                        break
        
        # If we found a rank, continue to see if we can do better
        # But stop trying if we've used too much time
        if time.time() - t_start > time_limit - 20:
            break
    
    # Convert to list of lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Convert to proper Python types (int where possible)
    def clean_val(v):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        elif abs(v*2 - round(v*2)) < 1e-9:
            return (2*int(round(v*2)))//2
        else:
            return v
    
    U_out = [[clean_val(v) for v in row] for row in U_out]
    V_out = [[clean_val(v) for v in row] for row in V_out]
    W_out = [[clean_val(v) for v in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
