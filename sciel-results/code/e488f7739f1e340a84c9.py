"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
from collections import defaultdict

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check against identity: T[i,j,k] should be 1 if this is the (i,j,k) entry of C = A@B
    # Actually the tensor for matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # Wait, no. The tensor decomposition means:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should equal 1 for the matmul tensor (since C_ik = sum_j A_ij B_jk)
    # Actually, the matmul tensor has T[i,j,k] = 1 for all valid i,j,k.
    
    # Let me reconsider. The standard formulation:
    # We want: for all i,j,k, sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta (the tensor)
    # For matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k in ranges.
    # No wait, that's not right either.
    
    # Let me think again. The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_integer_halfinteger(U, V, W):
    """Check all entries are integer or half-integer (multiples of 0.5)."""
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _try_greedy_decomposition(n, m, p, target_rank, max_iters=200, seed=42):
    """Try to find a decomposition of given rank using greedy/randomized approach."""
    rng = random.Random(seed)
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: start with schoolbook and try to merge/eliminate terms
    # Or: try random rank-r decompositions and check
    
    # Let's try a different approach: use the fact that we can combine terms
    # Start with schoolbook (rank = n*m*p) and try to reduce
    
    # Alternative: try to find a rank-r decomposition by solving a system
    # This is complex, so let's try heuristic approaches
    
    # Approach 1: Try to use Strassen-like block decomposition
    # For 4x4 * 4x5, we can think of blocks
    
    # Let's try a randomized search: pick random U, V, W with small integer/half-integer
    # entries and check if they work. This is very unlikely to work directly.
    
    # Better approach: Start from schoolbook and try to eliminate rank-1 terms
    # by finding linear dependencies
    
    # For now, let's try a few known constructions and random perturbations
    
    # Try rank values from a reasonable lower bound upward
    # Lower bound for matmul(n,m,p) is max(n*m, m*p, n*p) = max(16, 20, 20) = 20
    # But actual lower bounds are higher. Let's try from 30 upward.
    
    for r in range(30, target_rank + 1):
        # Try to construct a rank-r decomposition
        # Use a structured approach: try to find U, V, W
        
        # Method: Try random matrices with entries from {-1, 0, 1, 0.5, -0.5}
        # and check. Very unlikely but let's try a few.
        
        for attempt in range(max_iters // 10):
            U = [[rng.choice([-1, -0.5, 0, 0, 0, 0.5, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[rng.choice([-1, -0.5, 0, 0, 0, 0.5, 1]) for _ in range(r)] for _ in range(m * p)]
            W = [[rng.choice([-1, -0.5, 0, 0, 0, 0.5, 1]) for _ in range(r)] for _ in range(p * n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                if _verify_integer_halfinteger(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        # Also try a structured approach: use the schoolbook basis but with linear combinations
        # This is more complex, so skip for now in this function
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, try to find a better decomposition than schoolbook
    # Start with schoolbook as fallback
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p  # 80
    
    # Try a few heuristic approaches with time budget
    attempts = [
        ("greedy_50", lambda: _try_greedy_decomposition(n, m, p, 50, max_iters=100, seed=1)),
        ("greedy_45", lambda: _try_greedy_decomposition(n, m, p, 45, max_iters=100, seed=2)),
        ("greedy_40", lambda: _try_greedy_decomposition(n, m, p, 40, max_iters=100, seed=3)),
    ]
    
    for name, func in attempts:
        if time.time() > deadline:
            break
        try:
            result = func()
            if result is not None:
                rank = len(result["U"][0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    return best_result
# EVOLVE-BLOCK-END
