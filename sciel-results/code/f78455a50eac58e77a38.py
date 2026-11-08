"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - START_TIME) > TIME_LIMIT

def _verify_exact(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For exactness, check that for all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # and for all other combinations it should be 0... actually the tensor is defined as:
    # T_{a,b,c} where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # T_{(i,j),(j',k),(k',i')} = 1 if j==j' and i==i' and k==k', else 0
    # So we need: for each r, U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] sums to 1 when j=j', i=i', k=k'
    
    # More precisely: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all valid (i,j,k)
    # and the tensor structure means we only care about these 60 entries being 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification: the 3-tensor must match exactly."""
    rank = len(U[0]) if U else 0
    # The matrix multiplication tensor has T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # i.e., T[(i,j),(j,k),(k,i)] = 1, and 0 otherwise.
    # We check: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c
    
    for a in range(n*m):
        i_a = a // m
        j_a = a % m
        for b in range(m*p):
            j_b = b // p
            k_b = b % p
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                actual = 0.0
                for r in range(rank):
                    actual += U[a][r] * V[b][r] * W[c][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, max_iters=50):
    """Try to find an exact decomposition of given rank."""
    if _check_deadline():
        return None
    
    np.random.seed(rank * 1000 + 42)
    
    for attempt in range(max_iters):
        if _check_deadline():
            return None
        
        # Initialize with random small integers/half-integers
        U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
        V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
        W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
        
        # Scale to try to get closer
        # Compute current tensor and target
        # Use simple iterative refinement: adjust one factor at a time
        
        # Actually, let's try a different approach: start from schoolbook and try to merge terms
        if attempt == 0:
            # Start from schoolbook and try to reduce
            if rank >= n*m*p:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                # Now try to zero out some columns by linear combinations
                # This is complex; let's just return schoolbook if rank >= 60
                if rank == n*m*p:
                    return [U.tolist(), V.tolist(), W.tolist()]
        
        # Random search with verification
        # This is unlikely to work for rank < 60 with random init
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 60
    
    # Try to find lower rank decompositions
    # Known results: for 2x3x10, the rank might be 60 or possibly lower
    # Let's try a structured approach
    
    # Approach: Use the fact that n=2 to potentially reduce rank
    # For each k (0..9), we need to compute 2 outputs (i=0,1) from 3 inputs (j=0,1,2)
    # This is like 10 independent 2x3x3 sub-problems, but they share the j dimension
    
    # Actually, let me try a rank-55 approach with structured initialization
    # and verify exactness
    
    best_result = None
    
    # First, ensure we have a valid fallback
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
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve by searching for lower rank
    # Given time constraints, do a limited search
    for try_rank in range(50, 60):
        if _check_deadline():
            break
        result = _try_rank(try_rank, n, m, p, max_iters=10)
        if result is not None:
            U_list, V_list, W_list = result
            if _verify_full(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result
