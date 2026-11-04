"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]
_DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
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


def _rationalize(M, tol=1e-8, max_den=100):
    """Try to find rational entries close to float matrix M."""
    from fractions import Fraction
    rows, cols = M.shape
    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = M[r, c]
            if abs(v) < tol:
                out[r][c] = 0
            else:
                # try small denominators
                found = False
                for den in range(1, max_den + 1):
                    num = round(v * den)
                    if abs(v - num / den) < tol:
                        if num % den == 0:
                            out[r][c] = num // den
                        else:
                            out[r][c] = num / den
                        found = True
                        break
                if not found:
                    out[r][c] = round(v)
    return out


def _try_rank(rank, n, m, p, time_budget=10.0):
    """Attempt to find a rank-r decomposition via alternating least squares."""
    start = time.time()
    # Tensor T[i,j,k] = delta_{i,k} in flattened form: T[(i*m+j)*p + k] = 1 if i==k else 0
    # Actually the matmul tensor: T[i, j, k] = 1 if (i,j,k) corresponds to C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 if the triple (i,j,k) is "active", i.e., always 1 for valid indices
    # Wait - the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor entries are all 1. The rank is the minimum r such that T = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t[i,j], v_t[j,k], w_t[k,i] (with the trace convention W[k*n+i])
    
    # Flatten: T is a 3-tensor of shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Actually no. The tensor for matmul: T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
    # In the factorization: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # For the schoolbook: U[i*m+j, i*m+j] = 1, V[j*p+k, j*p+k] = 1, W[k*n+i, k*n+i] = 1
    # and the index t = i*m*p + j*p + k
    
    # Let's use a different approach: random initialization + ALS
    np.random.seed(42 + rank)
    
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Build the target tensor in a computable form
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We'll use a random test matrix approach to verify
    
    best_err = float('inf')
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        
        # Update U: for each (i,j), U[i*m+j, :] = argmin sum_{k} (T[i,j,k] - sum_t V[j*p+k,t]*W[k*n+i,t]*U[i*m+j,t])^2
        # This is: for each (i,j), U[i*m+j,:] = (V_j^T W_i)^+ T_{i,j,:}
        # where V_j is the m*p x rank submatrix for fixed j... no.
        
        # Actually: T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # For fixed (i,j): T[i,j,k] = sum_t U[i*m+j,t] * (V[j*p+k,t] * W[k*n+i,t])
        # Let A_t(k) = V[j*p+k,t] * W[k*n+i,t] for each t
        # Then T[i,j,:] = U[i*m+j,:] @ A  where A is (p, rank) with A[k,t] = V[j*p+k,t]*W[k*n+i,t]
        # So U[i*m+j,:] = T[i,j,:] @ A^+ (pseudoinverse)
        
        for i in range(n):
            for j in range(m):
                # Build A: (p, rank)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Target: T[i,j,k] = 1 for all k
                target = np.ones(p)
                # Solve: target = A^T @ U[i*m+j,:]  => U[i*m+j,:] = (A A^T)^{-1} A target
                # i.e., U[i*m+j,:] = pinv(A) @ target
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A.T, target, rcond=None)[0]
                except:
                    pass
        
        # Update V: for fixed j, T[i,j,k] = sum_t V[j*p+k,t] * (U[i*m+j,t] * W[k*n+i,t])
        # For fixed j: T[i,j,k] over all i,k
        # V[j*p+k,t]: for each k, V[j*p+k,:] is determined by:
        # T[i,j,k] = sum_t V[j*p+k,t] * U[i*m+j,t] * W[k*n+i,t] for all i
        # So for fixed (j,k): vector over i: T[:,j,k] = V[j*p+k,:] @ (U_col * W_row)
        # where U_col[i,t] = U[i*m+j,t], W_row[i,t] = W[k*n+i,t]
        # Let B[i,t] = U[i*m+j,t] * W[k*n+i,t], shape (n, rank)
        # T_col[i] = 1 for all i (since T[i,j,k]=1)
        # V[j*p+k,:] = pinv(B) @ ones(n)
        
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                target = np.ones(n)
                try:
                    V[j * p + k, :] = np.linalg.lstsq(B.T, target, rcond=None)[0]
                except:
                    pass
        
        # Update W: for fixed k, T[i,j,k] = sum_t W[k*n+i,t] * (U[i*m+j,t] * V[j*p+k,t])
        # For fixed (k,i): W[k*n+i,:] determined by:
        # T[i,j,k] = sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] for all j
        # Let C[j,t] = U[i*m+j,t] * V[j*p+k,t], shape (m, rank)
        # T_row[j] = 1 for all j
        # W[k*n+i,:] = pinv(C) @ ones(m)
        
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                target = np.ones(m)
                try:
                    W[k * n + i, :] = np.linalg.lstsq(C.T, target, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        best_err = min(best_err, err)
    
    if best_err < 1e-6:
        return _rationalize(U), _rationalize(V), _rationalize(W)
    return None


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    max_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # For 2x5x6, known results suggest rank might be around 30-60
    # Try ranks from lower bound upward
    
    min_possible = max(n * m, m * p, p * n)
    
    for rank in range(min_possible, max_rank + 1):
        if time.time() - start > _DEADLINE - 5:
            break
        time_budget = min(15.0, _DEADLINE - 5 - (time.time() - start))
        if time_budget < 2.0:
            break
        result = _try_rank(rank, n, m, p, time_budget=time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
