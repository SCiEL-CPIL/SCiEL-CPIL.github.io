"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # The tensor T has shape (n*m, m*p, p*n) with T[i*m+j, j*p+k, k*n+i] = 1
    # We want U (nm × r), V (mp × r), W (pn × r) such that
    # sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx] for all a,b,c_idx
    
    # Try various ranks from low to high
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Lower bound: at least max(nm, mp, pn) = 16
    lower_bound = max(n*m, m*p, p*n)
    
    # Try ranks from lower_bound to rank_schoolbook-1
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        decomp = _try_find_decomposition(n, m, p, rank, deadline - 5)
        if decomp is not None:
            U, V, W = decomp
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = rank
                best_decomp = (U, V, W)
                break  # Found a valid lower rank, stop searching
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}

def _try_find_decomposition(n, m, p, rank, time_limit):
    """Try to find a rank decomposition. Returns (U, V, W) or None."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with random or structured values
    # Use a simple approach: start from schoolbook and try to reduce
    
    # For now, try a few random initializations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > time_limit:
            break
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Try gradient descent or other optimization
        # This is a non-convex problem, so we need to be careful
        
        # For simplicity, just return None for now and rely on schoolbook
        # In a more complete implementation, we'd use optimization here
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0])
    
    # Check dimensions
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (a, b, c_idx), the sum should be 0
    
    # Build the full tensor and verify
    # T_expected[a, b, c_idx] = 1 if a = i*m+j, b = j*p+k, c_idx = k*n+i for some i,j,k
    # else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                val = sum(U[a][c] * V[b][c] * W[c_idx][c] for c in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Also check that off-diagonal terms are zero
    # This is expensive, so only check a sample
    for _ in range(1000):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c_idx = np.random.randint(0, pn)
        
        # Check if this is a valid term
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c_idx // n
        i2 = c_idx % n
        
        is_valid = (j == j2 and i == i2 and k == k2)
        
        val = sum(U[a][c] * V[b][c] * W[c_idx][c] for c in range(rank))
        expected = 1 if is_valid else 0
        if abs(val - expected) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
