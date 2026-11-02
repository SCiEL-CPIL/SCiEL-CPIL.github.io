import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Strategy: Fix U to a structured basis, solve for V and W
    # The matrix multiplication tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need T[i,j,k] = delta_{i,j,k} (the identity in the appropriate sense)
    
    # For matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] where the "output" index is (i,k) and input is (j)
    # Actually, the standard decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # But we need to match the trace convention W[k*n+i, r]
    
    # Let's use the approach: fix U to have a specific structure
    # U[i*m+j, r] - we can fix U to be a permutation of basis vectors
    
    # Try different rank values
    for rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_structured(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            continue
    
    return best_result

def _try_rank_structured(n, m, p, rank, deadline):
    """
    Try to find a rank decomposition by fixing U to a structured form
    and solving for V and W.
    """
    # The constraint is: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) valid else 0
    
    # Actually for matrix multiplication, we want:
    # (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] represents the coefficient such that
    # (AB)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    
    # Wait, let me re-read the problem. The output factor follows W[k*n+i, r]
    # This suggests the tensor is T[i,j,k] with indices:
    # - i: row of A (0..n-1)
    # - j: column of A / row of B (0..m-1)  
    # - k: column of B (0..p-1)
    # And T[i,j,k] = 1 for all valid combinations
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Strategy: Fix U to be a basis-like matrix
    # Let U[i*m+j, r] = delta_{(i,j), sigma(r)} for some permutation sigma
    # Then for each r, U selects a specific (i,j) pair
    
    # Better approach: Use the fact that we can absorb permutations
    # Let's try: U[i*m+j, r] = 1 if (i,j) = (r mod n, r div n) else 0
    # But this only works if rank >= n*m
    
    # Alternative: Use random initialization with structured constraints
    
    np.random.seed(int(time.time() * 1000) % 100000)
    
    # Generate random U with entries in {-1, 0, 1}
    U = np.zeros((n * m, rank))
    for row in range(n * m):
        # Each row has exactly one 1 (or -1)
        col = np.random.randint(rank)
        U[row, col] = np.random.choice([-1, 1])
    
    # Now solve for V and W
    # The constraint becomes: for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is still nonlinear. Let's use a different approach.
    
    # Fix both U and V to be structured, solve for W
    # U[i*m+j, r] = delta_{i, r mod n} * delta_{j, r div n}  (if rank >= n*m)
    # V[j*p+k, r] = delta_{j, r mod m} * delta_{k, r div m}  (if rank >= m*p)
    
    # But rank might be less than n*m or m*p
    
    # Let's try a gradient-based approach with better initialization
    
    # Initialize U, V, W with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Define the residual
    def compute_residual(U, V, W):
        """Compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1 for all i,j,k"""
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = val - 1.0
        return residual
    
    # Use scipy's least_squares
    from scipy.optimize import least_squares
    
    def flatten_params(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return U, V, W
    
    def objective(params):
        U, V, W = flatten_params(params)
        res = compute_residual(U, V, W)
        return res.flatten()
    
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = least_squares(objective, params0, method='trf', max_nfev=300)
    except Exception:
        return None
    
    if result.cost < 1e-8:
        U_opt, V_opt, W_opt = flatten_params(result.x)
        
        # Try to snap to half-integers
        for denom in [1, 2, 3, 4, 6]:
            U_snap = np.round(U_opt * denom) / denom
            V_snap = np.round(V_opt * denom) / denom
            W_snap = np.round(W_opt * denom) / denom
            
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_snap[i*m+j, r] * V_snap[j*p+k, r] * W_snap[k*n+i, r]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-9:
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
    
    return None
