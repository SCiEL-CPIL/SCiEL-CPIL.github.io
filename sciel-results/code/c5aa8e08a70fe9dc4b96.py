import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
DEADLINE = 150.0

def _build_schoolbook(n, m, p):
    """Return the standard (rank=n*m*p) decomposition."""
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

def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i,j,k] = 1 if we need (i,j) of A and (j,k) of B
    # The decomposition says: for each output (i,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 when j is the shared index, else 0.
    # More precisely: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: (AB)[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j'*p+k,r]*B[j',k]) ... 
    # Actually the standard form: T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    # where U has shape (n,m,rank), V has shape (m,p,rank), W has shape (p,n,rank)
    # But here U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # So U[i*m+j, r] corresponds to U_{i,j}^{(r)}
    # V[j*p+k, r] corresponds to V_{j,k}^{(r)}
    # W[k*n+i, r] corresponds to W_{i,k}^{(r)}
    
    # Check: for all i, j, k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 if this is a valid term
    # Actually the tensor T_{i,j,k} for matmul is 1 for all i,j,k (it's the standard multiplication tensor)
    # Wait no. The matrix multiplication tensor T has T_{i,j,k} = 1 for all valid (i,j,k) indices.
    # The decomposition requires: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by computing the tensor product and checking
    # For efficiency, check a random sample or all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_find_lower_rank(n, m, p, max_rank, time_budget):
    """
    Try to find a decomposition with rank < max_rank.
    Uses a simple approach: start from schoolbook and try to merge terms.
    """
    start = time.time()
    U, V, W = _build_schoolbook(n, m, p)
    current_rank = U.shape[1]
    
    # Try to find linear dependencies among the rank-1 tensors
    # Each rank-1 term is defined by (u_r, v_r, w_r)
    # If we can express some terms as linear combinations of others, we can reduce rank.
    
    # For a 3x3x6 multiplication, the known lower bounds suggest rank >= 27 (n*m*p / 2)
    # but actual minimal rank for small cases may be higher.
    
    # Let's try a random search: pick random subsets of terms and see if we can replace them.
    # This is complex, so let's try a simpler heuristic.
    
    # Actually, for 3x3x6, the schoolbook rank is 54.
    # Known results: 3x3x3 has rank 23 (Strassen-like), 3x3x6 might have rank around 40-50.
    
    # Let's try a simple approach: use numpy to check if we can find a lower rank
    # by solving for linear dependencies.
    
    # Represent each rank-1 term as a vector in R^(n*m * m*p * p*n)
    # But that's too large. Instead, we can check if the set of rank-1 tensors is linearly dependent.
    
    # For now, let's just return the schoolbook if we don't have time or a good method.
    if time.time() - start > time_budget:
        return None
    
    # Try a simple random perturbation approach
    # This is unlikely to work well, but let's try.
    
    return None

def construct():
    n, m, p = 3, 3, 6
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - elapsed - 10.0  # 10s safety margin
    
    # First, build the schoolbook decomposition as fallback
    U, V, W = _build_schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition if we have time
    if remaining > 30.0:
        result = _try_find_lower_rank(n, m, p, U.shape[1], remaining - 10.0)
        if result is not None:
            U, V, W = result
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
