"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dict of non-zero entries."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify(U, V, W, n, m, p, T):
    """Verify that the decomposition exactly matches the tensor."""
    rank = len(U[0]) if U else 0
    # Check all non-zero entries
    for (a, b, c), val in T.items():
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if s != val:
            return False
    # Check all zero entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in T:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
    return True

def _try_rank(n, m, p, rank, timeout=10):
    """Try to find a decomposition of given rank. Returns (U,V,W) or None."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # For small ranks, try systematic search
    # The key constraint: for each (a,b,c) not in T, sum_r U[a,r]V[b,r]W[c,r] = 0
    # And for each (a,b,c) in T, sum_r U[a,r]V[b,r]W[c,r] = 1
    
    # Use randomized search with constraint propagation
    # Initialize with random values and try to satisfy constraints
    
    # Actually, for this small problem, let's try a different approach:
    # Use the fact that we can set some entries to 1 and others to 0
    # and use the structure of the tensor
    
    # Strategy: try to find a decomposition where each rank-1 term
    # corresponds to a "slice" of the tensor
    
    # For rank r, we need r rank-1 tensors that sum to T
    # A rank-1 tensor T_r(a,b,c) = U[a,r] * V[b,r] * W[c,r]
    
    # Let's try: for each rank-1 term, it should be non-zero only on
    # a structured subset of the (a,b,c) space
    
    # Simple approach: try all-ones in certain patterns
    # For the all-ones pattern on a "fiber": fix one index, vary others
    
    # Actually, let me try a greedy approach:
    # Start with T, and for each rank-1 term, find the best match and subtract
    
    # But this won't give exact results easily. Let me try integer search.
    
    # For very small ranks (like 1, 2, 3), we can enumerate
    if rank <= 3:
        # Try all combinations of small integers
        values = [-1, 0, 1]
        # This is too large for full enumeration, so use heuristics
        pass
    
    # Let's try a structured approach:
    # The tensor has the property that T[i*m+j, j*p+k, k*n+i] = 1
    # We can try to factor this using the index structure
    
    # For rank = n (2): 
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # If we set U[i*m+j, r] = 1 if i==r, V[j*p+k, r] = 1, W[k*n+i, r] = 1 if i==r
    # Then sum_r [i==r] * 1 * [i==r] = [i==i] = 1. 
    # But we need to check zero entries too.
    
    # Let me verify rank 2:
    # U[i*m+j, r] = 1 if i==r else 0
    # V[j*p+k, r] = 1 for all j,k,r
    # W[k*n+i, r] = 1 if i==r else 0
    # 
    # For (a,b,c) = (i*m+j, j*p+k, k*n+i):
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = sum_r [i==r]*1*[i==r] = 1 ✓
    #
    # For (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    # We need sum_r U[a,r] V[b,r] W[c,r] = 0
    # a = i1*m+j1, b = j2*p+k2, c = k3*n+i3
    # sum_r [i1==r] * 1 * [i3==r] = [i1==i3]
    # So if i1 != i3, it's 0. But if i1 == i3 and the indices are otherwise inconsistent, it's 1. ✗
    
    # So rank 2 doesn't work in general. We need to check when i1==i3 but the (a,b,c) is not a valid tensor entry.
    
    # A valid entry requires: a=i*m+j, b=j*p+k, c=k*n+i for the SAME i,j,k.
    # So if a=i1*m+j1, b=j2*p+k2, c=k3*n+i3, it's valid iff i1==i3 AND j1==j2 AND k2==k3.
    
    # With my rank-2 scheme, sum_r = [i1==i3]. This is 1 whenever i1==i3, regardless of j1,j2,k2,k3.
    # But we need it to be 0 when (j1,j2,k2,k3) are not all consistent.
    # So rank 2 fails.
    
    # What about a smarter rank-2 scheme?
    # U[i*m+j, r] = f_i(j) * [i==r]
    # V[j*p+k, r] = g_j(k) * [i==r] ... but V doesn't depend on i.
    
    # This is getting complex. Let me just do a computational search.
    
    # Given time constraints, let me try a randomized approach for small ranks
    random.seed(42)
    
    for attempt in range(100):
        if time.time() - start_time > timeout:
            break
        
        # Initialize with random small values
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        # Try to fix up using simple iterations
        for iteration in range(50):
            if time.time() - start_time > timeout:
                break
            if _verify(U, V, W, n, m, p, T):
                return U, V, W
            # Perturb randomly
            mat = random.choice([U, V, W])
            ri = random.randint(0, len(mat) - 1)
            rj = random.randint(0, rank - 1)
            mat[ri][rj] = random.choice([-1, 0, 1])
    
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Try ranks from low to high
    # Lower bound: at least 1
    # Upper bound: n*m*p = 30
    
    # Try to find minimum rank
    for rank in range(1, 31):
        result = _try_rank(n, m, p, rank, timeout=3)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, T):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
