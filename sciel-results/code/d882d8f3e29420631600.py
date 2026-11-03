import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 5, 5]
DEADLINE = 145.0  # Leave 5s safety margin before 150s kill

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
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

def _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Check if U, V, W form a valid tensor decomposition for matrix multiplication.
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if i,j,k are valid indices for (i,j) in A, (j,k) in B,
    and the product C[i,k] gets contribution from A[i,j]*B[j,k].
    
    Actually, the correct check: For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition should satisfy:
    For each term r, U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    This should reconstruct the multiplication tensor.
    
    The multiplication tensor has entries T[i,j,k] = 1 if the indices correspond to a valid
    multiplication step (i.e., for each i in [0,n), j in [0,m), k in [0,p), T[i,j,k] = 1
    because A[i,j] * B[j,k] contributes to C[i,k]).
    
    Wait, let me reconsider. The standard matrix multiplication tensor M_{n,m,p} has:
    M[i,j,k] = 1 for all valid i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    
    No, that's not right either. Let me think about this more carefully.
    
    The tensor for matrix multiplication is defined such that:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as a tensor contraction. The "multiplication tensor" T has indices
    (i, j, k) and T[i,j,k] = 1 for all valid combinations, meaning each term A[i,j]*B[j,k]
    contributes to C[i,k].
    
    A rank-r decomposition gives:
    T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
    
    So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Let's verify with schoolbook: each term r corresponds to one (i,j,k) triple, with
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1, and all other entries 0.
    Then sum_r ... = 1 for the matching (i,j,k), and 0 otherwise? No, that's wrong.
    
    Actually, for a fixed (i,j,k), only the term where r corresponds to that (i,j,k) has
    all three values equal to 1, so the sum is 1. For other (i',j',k'), the sum is also 1
    because there's exactly one term that matches. So T[i,j,k] = 1 for all valid (i,j,k).
    
    But wait, is T[i,j,k] supposed to be 1 for all (i,j,k)? Let's verify with a simple case.
    
    For 1x1x1: A is 1x1, B is 1x1, C is 1x1. C[0,0] = A[0,0]*B[0,0].
    The tensor T[0,0,0] should be 1, and the decomposition should give sum_r U[0,r]*V[0,r]*W[0,r] = 1.
    Schoolbook gives rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Sum = 1. Correct.
    
    For 2x1x1: A is 2x1, B is 1x1, C is 2x1.
    C[0,0] = A[0,0]*B[0,0]
    C[1,0] = A[1,0]*B[0,0]
    The tensor T has entries T[i,0,0] for i=0,1. Each should be 1.
    Schoolbook rank = 2*1*1 = 2.
    Term r=0 (i=0,j=0,k=0): U[0,0]=1, V[0,0]=1, W[0,0]=1
    Term r=1 (i=1,j=0,k=0): U[1,1]=1, V[0,1]=1, W[0,1]=1
    For (i=0,j=0,k=0): sum = U[0,0]*V[0,0]*W[0,0] + U[0,1]*V[0,1]*W[0,1] = 1*1*1 + 0*1*1 = 1. Correct.
    For (i=1,j=0,k=0): sum = U[1,0]*V[0,0]*W[0,0] + U[1,1]*V[0,1]*W[0,1] = 0*1*1 + 1*1*1 = 1. Correct.
    
    So the check is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_reduce_rank(n, m, p, target_rank, time_limit=30.0, seed=None):
    """
    Try to find a rank-target_rank decomposition using random search.
    Returns (U, V, W) as lists of lists with integer/half-integer entries, or None.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    max_iters = 10000
    
    # Use a simpler approach: start from schoolbook and try to merge terms
    # or use a heuristic search
    
    # For now, try random sparse decompositions
    for iter_num in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Random sparse initialization
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Initialize with random ±1 values in a sparse pattern
        for r in range(target_rank):
            # Each term r corresponds to some pattern
            # Randomly assign non-zero entries
            num_nonzero_u = random.randint(1, min(n*m, target_rank // 2 + 1))
            num_nonzero_v = random.randint(1, min(m*p, target_rank // 2 + 1))
            num_nonzero_w = random.randint(1, min(p*n, target_rank // 2 + 1))
            
            u_indices = random.sample(range(n*m), num_nonzero_u)
            v_indices = random.sample(range(m*p), num_nonzero_v)
            w_indices = random.sample(range(p*n), num_nonzero_w)
            
            for idx in u_indices:
                U[idx][r] = random.choice([-1, 1])
            for idx in v_indices:
                V[idx][r] = random.choice([-1, 1])
            for idx in w_indices:
                W[idx][r] = random.choice([-1, 1])
        
        # Check validity
        if _validate_decomposition(U, V, W, n, m, p, tol=1e-10):
            # Convert to half-integers if needed
            U_frac = [[_to_fraction(x) for x in row] for row in U]
            V_frac = [[_to_fraction(x) for x in row] for row in V]
            W_frac = [[_to_fraction(x) for x in row] for row in W]
            return U_frac, V_frac, W_frac
    
    return None

def _to_fraction(x):
    """Convert a float to an integer or half-integer if possible, else return as is."""
    # Check if close to an integer
    if abs(x - round(x)) < 1e-10:
        return int(round(x))
    # Check if close to a half-integer
    half = round(2 * x) / 2.0
    if abs(x - half) < 1e-10:
        if half == int(half):
            return int(half)
        return half
    # Return as float (should not happen for valid decompositions we construct)
    return x

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Try ranks from best_rank-1 down to some minimum
    min_rank_to_try = max(1, best_rank - 15)  # Try up to 15 below schoolbook
    
    for target_rank in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Time budget per rank attempt
        time_budget = min(10.0, max(2.0, (DEADLINE - (time.time() - start_time)) / (best_rank - target_rank)))
        
        result = _try_reduce_rank(n, m, p, target_rank, time_limit=time_budget)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            # Continue trying to find even lower rank
            continue
    
    return best_result
