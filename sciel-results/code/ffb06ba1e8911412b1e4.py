"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    # The tensor T[i,j,k] = delta (A[i,j] * B[j,k] = C[i,k])
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a valid mult
    # Actually, the tensor for M_{n,m,p} has T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # Wait no. The matrix multiplication tensor M_{n,m,p} is defined as:
    # T[i,j,k] such that sum_{j} A[i,j] * B[j,k] = sum_{j} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]) * A[i,j]*B[j,k]
    # Actually the standard form: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T_{i,j,k} where the bilinear form is sum_{r} U_{i,j}^{r} V_{j,k}^{r} W_{i,k}^{r}
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta_{i,k?} No...
    
    # Let me think again. The matrix multiplication tensor M_{n,m,p} has entries:
    # T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # The rank-r decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right either. Let me use the standard convention.
    
    # Standard: M_{n,m,p} tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # Rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U is n*m x r, V is m*p x r, W is p*n x r (using the W[k*n+i,r] convention)
    
    # So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the full tensor and check
    T_expected = np.ones((n, m, p))
    T_computed = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        # U[:, r] is a vector of length n*m
        # V[:, r] is a vector of length m*p  
        # W[:, r] is a vector of length p*n
        # We need to compute the outer product appropriately
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        # T[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_computed[i, j, k] += U_r[i, j] * V_r[j, k] * W_r[k, i]
    
    return np.allclose(T_computed, T_expected, atol=tol)

def _try_rank_fixed_U(U, n, m, p, max_iter=1000):
    """Given a fixed U matrix, try to find V and W that complete the decomposition.
    
    The constraint is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is bilinear in V and W. We can alternate: fix W, solve for V (linear), fix V, solve for W (linear).
    """
    rank = U.shape[1]
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    for it in range(max_iter):
        # Fix W, solve for V
        # For each (j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
        # This is: for each (j,k), we have a system: sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
        # Let A_{i,r} = U[i*m+j, r] * W[k*n+i, r]
        # Then: sum_r A_{i,r} * V[j*p+k, r] = 1 for all i
        # This is a linear system: A * v = 1, where v = V[j*p+k, :]
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Build the n x rank matrix A
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                # Solve A * v = 1
                try:
                    v, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return None, None
        
        # Fix V, solve for W
        # For each (k,i): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
        # Let B_{j,r} = U[i*m+j, r] * V[j*p+k, r]
        # Then: sum_r B_{j,r} * W[k*n+i, r] = 1 for all j
        # This is: B * w = 1, where w = W[k*n+i, :]
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # Build the m x rank matrix B
                B = np.zeros((m, rank))
                for j in range(m):
                    B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve B * w = 1
                try:
                    w, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return None, None
        
        V = V_new
        W = W_new
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return V, W
    return None, None

def _random_U(rank, n, m, seed=42):
    """Generate a random U matrix with structure that might help."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    # Scale to have reasonable magnitude
    U /= np.sqrt(rank)
    return U

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Try ranks from 24 upward
    
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random U matrices
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            U = _random_U(target_rank, n, m, seed=42 + target_rank * 100 + attempt)
            
            V, W = _try_rank_fixed_U(U, n, m, p, max_iter=500)
            
            if V is not None and W is not None:
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break  # Found a valid decomposition for this rank, move to next rank
        
        if best_rank == target_rank:
            # Found a valid decomposition for this rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
