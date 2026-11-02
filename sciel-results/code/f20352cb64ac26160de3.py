import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k
    # And for non-matching indices, it should be 0.
    # Actually, the correct condition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is because the tensor component T[(i,j),(j,k),(k,i)] = 1 (diagonal)
    # and all other components are 0.
    
    # Let's verify by checking the bilinear identity:
    # For random A (n x m) and B (m x p), check that A*B = sum_r (U_r^T A) * (V_r^T B) * W_r^T
    # where U_r is reshaped to n x m, V_r to m x p, W_r to p x n (then transposed to n x p)
    
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)  # trace convention: W[k*n+i] is at position (k,i)
        # The contribution is: (A * u_r) * (B * v_r) * w_r^T
        # But this doesn't make sense dimensionally.
        
        # Let me reconsider. The decomposition should satisfy:
        # (A*B)[i,k] = sum_r sum_{i',j} U[i'*m+j, r] * A[i',j] * sum_{j',k'} V[j'*p+k', r] * B[j',k'] * W[k*n+i, r]
        # This doesn't work because we'd be summing over i' and k' which shouldn't be there.
        
        pass
    
    # Alternative verification: directly check the tensor components
    # T[(i,j),(j',k),(k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i']
    # This should equal delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += U[i * m + j, r] * V[j2 * p + k, r] * W[k2 * n + i2, r]
                            err = abs(actual - expected)
                            if err > max_err:
                                max_err = err
                            if err > tol:
                                return False, max_err
    return True, max_err

def _try_random_search(n, m, p, max_rank, n_trials=100, deadline=None):
    """Try to find a low-rank decomposition using random initialization and verification."""
    best_rank = None
    best_decomp = None
    
    for rank in range(max_rank, 0, -1):
        for trial in range(n_trials):
            if deadline and time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to improve conditioning
            for r in range(rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    scale = (norm_u * norm_v * norm_w) ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            # Verify
            valid, err = _verify(U, V, W, n, m, p, tol=1e-6)
            if valid:
                best_rank = rank
                best_decomp = (U, V, W)
                break
        if best_rank is not None:
            break
        if deadline and time.time() > deadline:
            break
    
    if best_decomp is not None:
        return best_rank, best_decomp
    return None, None

def construct():
    n, m, p = 3, 4, 8
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition
    # Known lower bound for 3x4x8 is around 41
    # Try ranks from 41 to 95
    for rank in range(41, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank
        found_rank, decomp = _try_random_search(n, m, p, rank, n_trials=50, deadline=deadline - 10)
        if found_rank is not None:
            best_rank = found_rank
            best_U, best_V, best_W = decomp
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
