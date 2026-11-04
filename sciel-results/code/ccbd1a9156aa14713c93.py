"""Rewritten program for 3x4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # Target: find min rank r such that exists U(nm x r), V(mp x r), W(pn x r)
    # with for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Schoolbook rank = n*m*p = 60
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        r = schoolbook_rank
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, r
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_numeric_search(rank, max_iters=5000, lr=0.01, seed=42):
        """Try to find a rank-`rank` decomposition via gradient descent."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Precompute index mappings
        indices = [(i, j, k, i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k, ui, vi, wi) in indices:
                # f = sum_c U[ui,c] * V[vi,c] * W[wi,c]
                # df/dU[ui,c] = V[vi,c] * W[wi,c]
                # df/dV[vi,c] = U[ui,c] * W[wi,c]
                # df/dW[wi,c] = U[ui,c] * V[vi,c]
                vw = V[vi, :] * W[wi, :]
                uw = U[ui, :] * W[wi, :]
                uv = U[ui, :] * V[vi, :]
                
                dU[ui, :] += vw
                dV[vi, :] += uw
                dW[wi, :] += uv
            
            # Gradient of error = sum over (i,j,k) of (f-1)^2
            # d(error)/dU = 2*sum (f-1) * df/dU
            # We need to compute f for each term
            # This is expensive; let's do it properly
            
            # Recompute with full gradient
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for (i, j, k, ui, vi, wi) in indices:
                f = np.dot(U[ui, :], V[vi, :] * W[wi, :])
                err = f - 1.0
                vw = V[vi, :] * W[wi, :]
                uw = U[ui, :] * W[wi, :]
                uv = U[ui, :] * V[vi, :]
                
                dU[ui, :] += 2.0 * err * vw
                dV[vi, :] += 2.0 * err * uw
                dW[wi, :] += 2.0 * err * uv
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 500 == 0:
                max_err = max(abs(np.dot(U[i*m+j,:], V[j*p+k,:]*W[k*n+i,:]) - 1.0) 
                             for i in range(n) for j in range(m) for k in range(p))
                if max_err < 1e-8:
                    break
        
        # Check convergence
        max_err = max(abs(np.dot(U[i*m+j,:], V[j*p+k,:]*W[k*n+i,:]) - 1.0) 
                     for i in range(n) for j in range(m) for k in range(p))
        if max_err < 1e-7:
            return U, V, W, max_err
        return None, None, None, max_err
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, r_sb = build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    # Try to find lower rank
    for rank in range(r_sb - 1, 40, -1):
        if time.time() > deadline - 20.0:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            U_t, V_t, W_t, err = try_numeric_search(rank, max_iters=3000, lr=0.005, seed=seed + rank)
            if U_t is not None and err < 1e-7:
                best_U, best_V, best_W, best_rank = U_t, V_t, W_t, rank
                found = True
                break
        if found:
            continue
    
    # Convert to lists with rounding to half-integers if close
    def to_list(mat):
        mat = np.round(mat * 2) / 2.0  # round to half-integers
        return mat.tolist()
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb, r_sb = build_schoolbook()
        best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
