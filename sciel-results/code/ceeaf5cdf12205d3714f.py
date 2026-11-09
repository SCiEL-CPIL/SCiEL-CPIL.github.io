"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 if the standard matrix multiplication has a 1 at that position, else 0.
    For standard matrix multiplication, T[i,j,k] = delta_{i,j,k} in the appropriate sense.
    Actually, the tensor for matrix multiplication C = AB where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T[i,j,k] has T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor).
    Wait, no. The tensor is defined such that:
    C[i,k] = sum_r sum_j U[i,j,r] * V[j,k,r] * W[r] ... 
    Actually the standard form is:
    C[i,k] = sum_r W[r,i,k] * (sum_j U[r,i,j] * A[i,j]) * (sum_k V[r,j,k] * B[j,k])
    Let me re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
    
    The standard tensor decomposition for matrix multiplication:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where the sum is over r (rank dimension) AND j (inner dimension).
    
    Wait, that's not right either. Let me think more carefully.
    
    In the standard formulation, we have:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-1 decomposition would be:
    C[i,k] = sum_r (sum_j U[r,i,j] * A[i,j]) * (sum_k V[r,j,k] * B[j,k]) * W[r,i,k]
    
    But the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    So U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    
    The tensor T[i,j,k] for matrix multiplication is 1 for all (i,j,k) since
    C[i,k] = sum_j A[i,j] B[j,k] means the tensor has T[i,j,k] = 1.
    
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    rank = U_arr.shape[1]
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # This is equivalent to checking that the tensor product equals the all-ones tensor.
    
    # Efficient check: for each r, form the outer product of U[:,r], V[:,r], W[:,r]
    # and sum over r. Then check if the result is all ones.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # The tensor T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    # We can compute this using einsum
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's compute this step by step
    # First, for each r, we have U_reshaped[:,:,r] is (n,m), V_reshaped[:,:,r] is (m,p), W_reshaped[:,:,r] is (p,n)
    # The contribution to T from rank r is:
    # T_r[i,j,k] = U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
    
    # This is a 3D tensor of shape (n, m, p) for each r, summed over r.
    
    # Using einsum: 'ijr,jkr,kr->ijk' but W is indexed as [k,i,r] so it's 'ijr,jkr,kr->ijk'
    # Wait, W_reshaped is (p, n, rank) so W_reshaped[k,i,r]
    # So the einsum is: U[i,j,r] * V[j,k,r] * W[k,i,r] -> T[i,j,k]
    # In einsum notation: 'ijr,jkr,kr->ijk'
    
    try:
        T = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    
    # T should be all ones
    return np.allclose(T, 1.0, atol=tol)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W, rank

def _try_kronecker(n, m, p, n1, m1, p1):
    """
    Try Kronecker product decomposition with factors (n1,m1,p1) and (n2,m2,p2).
    Returns (U, V, W, rank) or None if not a valid factorization.
    """
    n2 = n // n1
    m2 = m // m1
    p2 = p // p1
    
    if n1 * n2 != n or m1 * m2 != m or p1 * p2 != p:
        return None
    
    rank1 = n1 * m1 * p1
    rank2 = n2 * m2 * p2
    total_rank = rank1 * rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                c1 = i1 * m1 * p1 + j1 * p1 + k1
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            
                            c2 = i2 * m2 * p2 + j2 * p2 + k2
                            c = c1 * rank2 + c2
                            
                            U[i * m + j][c] = 1
                            V[j * p + k][c] = 1
                            W[k * n + i][c] = 1
    
    return U, V, W, total_rank

def _search_random(n, m, p, max_rank, time_limit, seed=42):
    """
    Random search for low-rank decompositions.
    """
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    best_U, best_V, best_W = None, None, None
    best_rank = max_rank + 1
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try Kronecker decompositions
    for n1 in range(1, n + 1):
        if n % n1 != 0:
            continue
        n2 = n // n1
        for m1 in range(1, m + 1):
            if m % m1 != 0:
                continue
            m2 = m // m1
            for p1 in range(1, p + 1):
                if p % p1 != 0:
                    continue
                p2 = p // p1
                result = _try_kronecker(n, m, p, n1, m1, p1)
                if result is not None:
                    U_k, V_k, W_k, rank_k = result
                    if rank_k < best_rank:
                        # Verify
                        if _verify_decomposition(U_k, V_k, W_k, n, m, p):
                            best_U, best_V, best_W = U_k, V_k, W_k
                            best_rank = rank_k
    
    # Random search: try to find lower rank decompositions
    # Strategy: start with a random sparse initialization and use iterative methods
    # For matrix multiplication, we can use the "tensor rank" approach
    
    # Try various random initializations with low rank
    for rank_attempt in range(best_rank - 1, 1, -1):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Try multiple random initializations
        for trial in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Random sparse initialization
            U_rand = np.zeros((n * m, rank_attempt))
            V_rand = np.zeros((m * p, rank_attempt))
            W_rand = np.zeros((p * n, rank_attempt))
            
            # Randomly assign 1s
            for r in range(rank_attempt):
                # Each rank-1 term should have some structure
                # For matrix multiplication, a rank-1 term corresponds to
                # choosing specific i, j, k indices
                # But we can also have more general terms
                
                # Randomly choose non-zero entries
                idx_u = rng.randint(0, n * m)
                idx_v = rng.randint(0, m * p)
                idx_w = rng.randint(0, p * n)
                
                U_rand[idx_u, r] = 1.0
                V_rand[idx_v, r] = 1.0
                W_rand[idx_w, r] = 1.0
            
            # Verify
            if _verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p):
                best_U, best_V, best_W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                best_rank = rank_attempt
                break  # Found valid decomposition at this rank
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Deadline guard: leave 10 seconds margin
    deadline = 140
    
    # First, get schoolbook decomposition
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to improve with Kronecker and random search
    U, V, W, rank = _search_random(n, m, p, rank, deadline - (time.time() - start_time))
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
