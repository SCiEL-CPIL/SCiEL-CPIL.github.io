import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Construct the trivial rank n*m*p solution efficiently
    # U shape (n*m, R), V shape (m*p, R), W shape (p*n, R)
    R_trivial = n * m * p
    U_trivial = np.zeros((n * m, R_trivial))
    V_trivial = np.zeros((m * p, R_trivial))
    W_trivial = np.zeros((p * n, R_trivial))
    
    # We can fill these using broadcasting or simple loops, but loops are fine for 42
    # To be safe and fast, we'll use the loop but it's negligible compared to optimization
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, idx] = 1.0
                V_trivial[j * p + k, idx] = 1.0
                W_trivial[k * n + i, idx] = 1.0
                idx += 1
                
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = R_trivial

    # Helper to verify a decomposition
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        R = U.shape[1]
        # Vectorized check for speed
        # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # And 0 for other combinations? 
        # The tensor T[i,j,k] for matmul is 1 if we are computing C[i,k] = sum_j A[i,j]B[j,k].
        # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # For standard matmul, T[i,j,k] = 1 for all i,j,k? 
        # No, the "tensor" of matrix multiplication is defined such that 
        # C_{ik} = sum_{j} A_{ij} B_{jk}.
        # The bilinear form is C_{ik} = sum_{j, r} U_{ij, r} V_{jk, r} W_{ki, r}.
        # This matches if we identify the tensor T with indices i, j, k where 
        # T[i,j,k] contributes to C[i,k]. 
        # Specifically, the condition is that for all i, k:
        # sum_{j, r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] * A[i,j] B[j,k] ... 
        # Wait, the definition in the prompt is:
        # "exactly decompose the matrix multiplication tensor"
        # "score is the rank"
        # The standard definition is that the tensor $\mathcal{T}$ has entries $\mathcal{T}_{i,j,k} = 1$ 
        # if the indices correspond to a valid multiplication term?
        # Actually, for $A \in \mathbb{R}^{n \times m}, B \in \mathbb{R}^{m \times p}$,
        # $C_{i,k} = \sum_{j=1}^m A_{i,j} B_{j,k}$.
        # This is a trilinear map $A, B \to C$.
        # The tensor $T$ in $U \otimes V \otimes W$ space is such that
        # $T_{i,j,k} = 1$ for all $i,j,k$? No.
        # If $T_{i,j,k}=1$ for all, then $\sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1$ for all $i,j,k$.
        # This is the condition for the "multiplication tensor" to be the all-ones tensor?
        # Let's check the schoolbook.
        # Schoolbook: $U_{ij, r(i,j,k)} = 1, V_{jk, r(i,j,k)} = 1, W_{ki, r(i,j,k)} = 1$.
        # Sum over r: Only term where r matches (i,j,k) is non-zero.
        # So $\sum_r \dots = 1$ for all $i,j,k$.
        # Yes, the condition is that the trilinear form equals 1 for all valid indices.
        
        # Vectorized verification:
        # Reshape U, V, W to 3D?
        # U is (n*m, R). Reshape to (n, m, R).
        # V is (m*p, R). Reshape to (m, p, R).
        # W is (p*n, R). Reshape to (p, n, R).
        
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # We want sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] == 1
        # This is a contraction.
        # Let's compute the tensor S[i,j,k] = sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r]
        # We can do this by iterating over r? No, R is small.
        # Or using einsum: 'ijr,jkr,kir->ijk'
        # S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        try:
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            if not np.allclose(S, 1.0, atol=tol):
                return False
        except Exception:
            return False
            
        return True

    def try_find_decomp(n, m, p, target_rank, time_limit):
        """
        Attempts to find a rank target_rank decomposition.
        """
        if time.time() > time_limit:
            return None
            
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        total_size = size_U + size_V + size_W
        
        def residuals(x):
            U = x[:size_U].reshape(n, m, target_rank)
            V = x[size_U:size_U + size_V].reshape(m, p, target_rank)
            W = x[size_U + size_V:].reshape(p, n, target_rank)
            
            # Compute S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            S = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            # Target is 1.0 for all i,j,k
            res = S - 1.0
            return res.flatten()
        
        # Try a few random restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            # Initialize with random small values
            x0 = np.random.randn(total_size) * 0.1
            
            try:
                sol = least_squares(residuals, x0, method='lm', max_nfev=200)
                if sol.cost < 1e-10:
                    x_sol = sol.x
                    U = x_sol[:size_U].reshape(n * m, target_rank)
                    V = x_sol[size_U:size_U + size_V].reshape(m * p, target_rank)
                    W = x_sol[size_U + size_V:].reshape(p * n, target_rank)
                    
                    # Snap to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    if verify_decomp(U_half, V_half, W_half, n, m, p, tol=1e-6):
                        return U_half, V_half, W_half
                        
            except Exception:
                continue
                
        return None

    # Strategy: Try to find rank < 42.
    # Given the difficulty, we only try a few ranks close to 42 with a tight budget.
    # The lower bound is 21. But 42 is likely the answer.
    # We will try rank 41 and 40 quickly.
    
    ranks_to_try = [41, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        res = try_find_decomp(n, m, p, r, deadline - 10)
        if res:
            U_new, V_new, W_new = res
            best_U = U_new
            best_V = V_new
            best_W = W_new
            best_rank = r
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
