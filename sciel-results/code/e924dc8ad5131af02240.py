"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def _verify_factors_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast vectorized verification using NumPy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Precompute all required dot products
    # For each (i,j,k), check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    all_ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
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
    return U, V, W

def _try_random_search(n, m, p, target_rank, trials=10000, seed=42, time_limit=100):
    """Random search for a low-rank decomposition with small integer entries."""
    rng = random.Random(seed)
    start_time = time.time()
    
    for trial in range(trials):
        if time.time() - start_time > time_limit:
            break
        
        # Use smaller range for better chance of exact solutions
        val_range = rng.choice([1, 2, 3])
        U = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rng.randint(-val_range, val_range) for _ in range(target_rank)] for _ in range(p * n)]
        
        if _verify_factors_fast(U, V, W, n, m, p):
            return U, V, W, target_rank
    return None

def _try_structured_rank14(n, m, p):
    """
    Try to construct a rank-14 decomposition for 2x3x3 matrix multiplication.
    
    Strategy: For 2x3x3, we can think of it as two 1x3x3 multiplications combined.
    A 1x3x3 multiplication has rank 9 (schoolbook). But we might do better by sharing terms.
    
    Let's try a specific construction based on the idea that we can merge some terms.
    """
    # For 2x3x3, indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # Tensor T[i,j,k] = 1 if the product (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Schoolbook: 18 terms, one for each (i,j,k)
    # Try to find 14 terms that still cover all required products
    
    # One approach: use the fact that for fixed i, the 3x3 matrix product can be done
    # but we share the j dimension between the two i values.
    
    # Let's try a specific pattern. We'll use 14 basis tensors.
    # This is a heuristic construction; if it doesn't work, return None.
    
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 9 rows
    W = [[0] * rank for _ in range(p * n)]  # 6 rows
    
    # We'll try to assign values such that the tensor decomposes correctly.
    # This requires solving a system of equations. For now, try a simple pattern.
    
    # Pattern: use the first 9 terms for i=0 (j=0,1,2; k=0,1,2) and last 5 for i=1
    # But this won't work because we need to share some terms.
    
    # Alternative: try a random search with the constraint that entries are in {-1, 0, 1}
    # and see if we can find a rank-14 solution.
    
    return None

def _try_gaussian_elimination_rank(n, m, p, target_rank, time_limit=20):
    """
    Try to find a rank-target decomposition by solving a linear system.
    
    The tensor T has entries T[i,j,k] = 1.
    We want to find U, V, W such that for all (i,j,k):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a system of n*m*p equations in rank*(n*m + m*p + p*n) unknowns.
    For n=2,m=3,p=3: 18 equations in 14*(6+9+6)=378 unknowns (for rank=14).
    This is underdetermined, so solutions likely exist.
    """
    start_time = time.time()
    
    # Number of equations
    num_eq = n * m * p  # 18
    
    # For each equation (i,j,k), we have:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a multilinear system. We can try to fix some variables and solve.
    
    # Strategy: fix U and V to random small integers, then solve for W.
    # But that's not quite right either.
    
    # Better: use a simple iterative approach. Start with random U, V, W and try to
    # adjust them to satisfy the equations.
    
    # For now, this is a placeholder. The random search in _try_random_search is the main approach.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # First, try random search with decreasing rank, starting from lower ranks
    # Use small integer entries for better chance of exact solutions
    
    for target_rank in [11, 12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        remaining_time = deadline - elapsed
        if remaining_time < 10:
            break
        
        # Allocate time proportional to how likely this rank is to succeed
        # Lower ranks are harder, so give them less time
        time_budget = min(remaining_time - 5, 15)
        
        result = _try_random_search(n, m, p, target_rank, trials=20000, seed=target_rank * 1000, time_limit=time_budget)
        if result is not None:
            U, V, W, rank = result
            if _verify_factors_fast(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
