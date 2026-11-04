"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The decomposition satisfies: for all i in [0,n), j in [0,m), k in [0,p)
    sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    
    Actually, the standard matrix multiplication tensor decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, let me think about this more carefully. The matrix multiplication
    tensor T has indices (i, j, k) where i is row of A (n), j is column of A / 
    row of B (m), k is column of B (p).
    
    T[i,j,k] = sum_j A[i,j] * B[j,k]
    
    In a rank-r decomposition, we write:
    T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right either. Let me reconsider.
    
    Actually, the standard way to decompose matrix multiplication is:
    T_{i,j,k} = sum_{r} U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    
    But the interface says U has shape (n*m, rank), V has shape (m*p, rank), 
    W has shape (p*n, rank).
    
    So U[i*m + j, r] corresponds to the (i,j) entry of U
    V[j*p + k, r] corresponds to the (j,k) entry of V
    W[k*n + i, r] corresponds to the (i,k) entry of W (note: k*n + i, not i*n + k)
    
    The decomposition should satisfy:
    For all i, j, k:
        1 = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    Wait, that's for the identity tensor. For matrix multiplication, we have:
    (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T_{i,j,k} = delta (it's 1 if we're computing the product).
    
    Actually, the matrix multiplication tensor is defined such that:
    sum_{i,j,k} T_{i,j,k} * A_{i,j} * B_{j,k} = sum_{i,k} (A*B)_{i,k} * C_{k,i}
    
    Hmm, let me think about this differently. The standard formulation is:
    
    We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] * B[j, k]
    
    The tensor T has entries T[i, j, k] such that:
    sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
    
    where C[i,k] = sum_j A[i,j] * B[j,k].
    
    So T[i,j,k] = 1 for all valid i,j,k? No, that's not right.
    
    Actually, the matrix multiplication tensor is:
    T_{i,j,k} = 1 if we're looking at the term A[i,j] * B[j,k] contributing to C[i,k].
    
    In a rank-r decomposition:
    T_{i,j,k} = sum_{r=1}^r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    
    But with the given indexing:
    U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    So the decomposition is:
    For all i in [0,n), j in [0,m), k in [0,p):
        1 = sum_{r=0}^{rank-1} U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    Wait, but that would mean T[i,j,k] = 1 for all i,j,k, which is the all-ones tensor, 
    not the matrix multiplication tensor.
    
    Let me reconsider. The matrix multiplication tensor T is such that:
    T[i, j, k] is the coefficient of A[i,j] * B[j,k] in the expression for C.
    
    Since C[i,k] = sum_j A[i,j] * B[j,k], the tensor T has:
    T[i, j, k] = 1 for all valid i, j, k.
    
    No wait, that's still the all-ones tensor. Let me look at this more carefully.
    
    Actually, I think the correct formulation is:
    The matrix multiplication tensor T_{i,j,k} is defined such that the bilinear
    form sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} equals sum_{i,k} (AB)_{i,k} C_{i,k}
    where C is the output.
    
    But in the standard tensor decomposition for matrix multiplication, we have:
    T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}
    
    And the schoolbook decomposition has rank n*m*p with:
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for the term (i,j,k)
    
    So for the schoolbook, each term r corresponds to a specific (i,j,k) triple,
    and U[i*m+j, r] = 1 if the row index matches i*m+j, etc.
    
    Let me just verify the schoolbook solution works:
    For a given (i,j,k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1.
    
    In the schoolbook, only the term r corresponding to (i,j,k) has all three factors
    equal to 1, and all other terms have at least one factor equal to 0.
    So the sum is indeed 1.
    
    So the verification is: for all i, j, k, the sum should be 1.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_rank_reduction(U, V, W, n, m, p, target_rank, rng, time_budget=30):
    """
    Try to find a valid decomposition at target_rank by starting from the
    schoolbook and attempting to merge/eliminate terms.
    
    Strategy: Randomly perturb the matrices and check validity.
    This is a heuristic approach.
    """
    start_time = time.time()
    
    # Initialize with random values in {-1, 0, 1}
    U_t = np.zeros((n * m, target_rank), dtype=float)
    V_t = np.zeros((m * p, target_rank), dtype=float)
    W_t = np.zeros((p * n, target_rank), dtype=float)
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with values in {-1, 0, 1}
        U_t[:] = rng.choice([-1, 0, 1], size=(n * m, target_rank))
        V_t[:] = rng.choice([-1, 0, 1], size=(m * p, target_rank))
        W_t[:] = rng.choice([-1, 0, 1], size=(p * n, target_rank))
        
        # Check if valid
        if _verify_decomposition(U_t, V_t, W_t, n, m, p):
            return U_t, V_t, W_t, True
    
    return None, None, None, False

def _local_search(n, m, p, initial_rank, time_budget=60, seed=42):
    """
    Perform local search to find a valid decomposition at a lower rank.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # Start from the lowest plausible rank
    # For matrix multiplication, the rank is at least n*m + m*p + p*n - (n-1)*(m-1)*(p-1) or similar bounds
    # A simple lower bound is max(n*m, m*p, p*n) = 16 for 3x4x4
    # But the actual minimal rank is unknown. Let's try from a few values below initial_rank.
    
    best_U, best_V, best_W = None, None, None
    best_rank = initial_rank
    
    # Get schoolbook as baseline
    U_sb, V_sb, W_sb, _ = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from initial_rank - 1 down to some minimum
    min_rank = max(n * m, m * p, p * n)  # Simple lower bound
    
    for target_rank in range(initial_rank - 1, min_rank - 1, -1):
        if time.time() - start_time > time_budget:
            break
        
        U_t, V_t, W_t, found = _try_rank_reduction(U_sb, V_sb, W_sb, n, m, p, target_rank, rng, 
                                                     time_budget=min(10, time_budget - (time.time() - start_time)))
        if found:
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = target_rank
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Construct a rank-minimal (or as low as possible) tensor decomposition
    for 3x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook solution as a guaranteed fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    
    # Try local search to find a better solution
    try:
        remaining_time = 120 - (time.time() - start_time)
        if remaining_time > 10:
            U, V, W, rank = _local_search(n, m, p, rank_sb, time_budget=remaining_time - 10, seed=42)
            
            # Verify the solution is valid
            if _verify_decomposition(U, V, W, n, m, p):
                # Convert to integer/half-integer if needed
                U = np.rint(U * 2) / 2
                V = np.rint(V * 2) / 2
                W = np.rint(W * 2) / 2
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
