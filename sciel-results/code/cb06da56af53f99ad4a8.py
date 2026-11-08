"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
_DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook matrix multiplication decomposition with rank n*m*p."""
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

def _strassen_style(n, m, p):
    """
    Try a structured decomposition. For rectangular matrices, we can use
    a block-based approach. Here we try to find a rank < n*m*p decomposition.
    
    Key insight: We can group terms. For each pair (j,k), the contribution to
    all (i) is A[i,j]*B[j,k]. We need U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] to
    sum to 1 for each (i,j,k).
    
    A better approach: Use the fact that for fixed j, we have n*p terms involving
    that j. We can try to share structure across i or k.
    
    For now, let's try a simple improvement: if any dimension is 1, we can do better.
    But for 3,4,5, let's just return schoolbook as baseline.
    """
    return None

def _try_structured_search(n, m, p, max_rank=50):
    """
    Try to find a decomposition with rank < n*m*p using a greedy/structured approach.
    This is heuristic and may not find optimal solutions.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try ranks from lower bound up to best_rank - 1
    # Lower bound: rank >= max(n*m, m*p, p*n) / something... actually for tensor rank,
    # the lower bound is more complex. Let's just try a range.
    
    for rank in range(max(n*m, m*p, p*n), min(max_rank + 1, best_rank)):
        if time.time() - start_time > 30:  # 30s per rank attempt
            break
        
        # Try a simple construction: assign random 0/1/-1 patterns
        rng = np.random.RandomState(rank * 1000 + n * m * p)
        
        for attempt in range(100):
            if time.time() - start_time > 30:
                break
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Initialize with sparse random values
            for r in range(rank):
                # Each column has a few non-zero entries
                u_nz = max(1, n*m // 4)
                v_nz = max(1, m*p // 4)
                w_nz = max(1, p*n // 4)
                
                u_idx = rng.choice(n*m, size=u_nz, replace=False)
                v_idx = rng.choice(m*p, size=v_nz, replace=False)
                w_idx = rng.choice(p*n, size=w_nz, replace=False)
                
                U[u_idx, r] = rng.choice([-1, 1], size=u_nz)
                V[v_idx, r] = rng.choice([-1, 1], size=v_nz)
                W[w_idx, r] = rng.choice([-1, 1], size=w_nz)
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
                return U_list, V_list, W_list, rank
    
    # Return best found
    U, V, W = best_result
    return U, V, W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get a valid schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try to find a better solution
    result = _try_structured_search(n, m, p, max_rank=50)
    
    if result is not None:
        U, V, W, rank = result
        if rank < n * m * p:
            # Verify before returning
            if _verify(U, V, W, n, m, p, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
