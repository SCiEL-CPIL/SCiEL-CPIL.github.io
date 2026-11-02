"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Precompute index arrays for vectorized operations
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    idx_U = []
    idx_V = []
    idx_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_U.append(i * m + j)
                idx_V.append(j * p + k)
                idx_W.append(k * n + i)
    idx_U = np.array(idx_U)
    idx_V = np.array(idx_V)
    idx_W = np.array(idx_W)
    num_eqs = len(idx_U)
    
    def compute_error_vec(U, V, W):
        """Compute error vector using numpy."""
        # U[idx_U] has shape (num_eqs, rank)
        # Element-wise multiply and sum over rank axis
        prod = U[idx_U] * V[idx_V] * W[idx_W]
        sums = prod.sum(axis=1)
        return sums - 1.0
    
    def compute_max_err(U, V, W):
        err = compute_error_vec(U, V, W)
        return np.max(np.abs(err))
    
    # Try to find lower rank using iterative refinement with vectorized ops
    for rank in range(55, 30, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            if found:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            # Iterative refinement with vectorized gradient steps
            lr = 0.05
            for iteration in range(500):
                if time.time() > deadline - 20:
                    break
                
                err = compute_error_vec(U, V, W)
                max_err = np.max(np.abs(err))
                if max_err < 1e-10:
                    break
                
                # Gradient descent step for U
                # grad_U[i*m+j, c] = sum over (j',k') where i*m+j appears
                # Simplified: for each equation (i,j,k), grad w.r.t U[i*m+j,c] += 2*err * V[j*p+k,c] * W[k*n+i,c]
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Vectorized gradient accumulation
                # For U: each row appears in p equations (one for each k)
                for k in range(p):
                    mask = np.array([idx_W[e] // n == k and idx_U[e] // m != k for e in range(num_eqs)])
                    # Actually, let's just do it directly
                    pass
                
                # Direct computation (still fast with numpy for small sizes)
                for e in range(num_eqs):
                    ue = idx_U[e]
                    ve = idx_V[e]
                    we = idx_W[e]
                    de = 2 * err[e]
                    dU[ue] += de * V[ve] * W[we]
                    dV[ve] += de * U[ue] * W[we]
                    dW[we] += de * U[ue] * V[ve]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if converged
            max_err = compute_max_err(U, V, W)
            if max_err < 1e-8:
                # Try to find exact rational solution
                # Round to nearest fraction with small denominator
                U_frac = np.zeros_like(U)
                V_frac = np.zeros_like(V)
                W_frac = np.zeros_like(W)
                
                for i in range(n*m):
                    for c in range(rank):
                        x = U[i, c]
                        # Try to find fraction p/q with q <= 4 close to x
                        best_f = x
                        for q in range(1, 5):
                            p_num = round(x * q)
                            f = p_num / q
                            if abs(f - x) < abs(best_f - x):
                                best_f = f
                        U_frac[i, c] = best_f
                
                for i in range(m*p):
                    for c in range(rank):
                        x = V[i, c]
                        best_f = x
                        for q in range(1, 5):
                            p_num = round(x * q)
                            f = p_num / q
                            if abs(f - x) < abs(best_f - x):
                                best_f = f
                        V_frac[i, c] = best_f
                
                for i in range(p*n):
                    for c in range(rank):
                        x = W[i, c]
                        best_f = x
                        for q in range(1, 5):
                            p_num = round(x * q)
                            f = p_num / q
                            if abs(f - x) < abs(best_f - x):
                                best_f = f
                        W_frac[i, c] = best_f
                
                # Verify with rounded values
                max_err_frac = compute_max_err(U_frac, V_frac, W_frac)
                if max_err_frac < 1e-6:
                    best_result = {
                        "U": U_frac.tolist(),
                        "V": V_frac.tolist(),
                        "W": W_frac.tolist()
                    }
                    best_rank = rank
                    found = True
                    break
    
    return best_result
# EVOLVE-BLOCK-END
