"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import numpy as np
import random
import time

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build tensors as numpy arrays
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # The decomposition says: for each (i,j,k), the coefficient should be 1 if we're multiplying
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should be 1 for all (i,j,k)
    # Wait, let me re-read. The matrix multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor entry T[i,j,k] = 1 for all i,j,k.
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_early_exit(U, V, W, n, m, p):
    """Quick check without full numpy conversion for speed."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random candidate decomposition and check if it works."""
    rng = random.Random(seed)
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                U[i*m+j][c] = rng.choice([-1, 0, 1])
        for j in range(m):
            for k in range(p):
                V[j*p+k][c] = rng.choice([-1, 0, 1])
        for k in range(p):
            for i in range(n):
                W[k*n+i][c] = rng.choice([-1, 0, 1])
    
    return U, V, W

def _build_decomposition(U, V, W):
    """Return the dict interface."""
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin before 150s limit
    
    # Best known: schoolbook rank = n*m*p = 63
    best_rank = n * m * p
    best_result = None
    
    # Try schoolbook as baseline
    U_sch = [[0] * best_rank for _ in range(n * m)]
    V_sch = [[0] * best_rank for _ in range(m * p)]
    W_sch = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j][column] = 1
                V_sch[j * p + k][column] = 1
                W_sch[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_sch, "V": V_sch, "W": W_sch}
    
    # Structured approach: Use known rank-23 decomposition for 3x3x3 MM
    # and extend to 3x3x7 by adding rank-1 terms for columns 3-6 of B.
    
    # Known rank-23 decomposition for 3x3x3 matrix multiplication (from literature)
    # This uses the "border rank" or specific algebraic construction.
    # For now, let's use a simpler approach: 
    # The 3x3x7 tensor can be seen as 7 copies of 3x3x1, but we can share structure.
    
    # Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # For each k, this is a 3x3 * 3x1 multiplication.
    # We can use the same A-factor structure for all k, and vary B-factor.
    
    # Let's try: rank = 9 (for A part) + 21 (for B part) = 30? No, that's not right.
    # Actually, the tensor is T[i,j,k] = 1 for all i,j,k.
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1.
    
    # Strategy: Use rank = 21 + 4 = 25? Or try to find a better structure.
    
    # Let's try a parametric approach:
    # For the first 3 columns of B (k=0,1,2), use a rank-23 3x3x3 decomposition.
    # For the remaining 4 columns (k=3,4,5,6), add rank-1 terms.
    # Total rank: 23 + 4 = 27.
    
    # Actually, let's be more clever. The 3x3x7 tensor has a special structure.
    # We can write it as a sum of rank-1 tensors.
    
    # Let's try to construct a decomposition with rank = 24.
    # Use the fact that for each fixed k, we need a 3x3x1 decomposition.
    # The minimal rank for 3x3x1 is 3 (trivially).
    # But we can share the A-factor across different k values.
    
    # Approach: Use a common set of "A-terms" and vary the B and W terms.
    # Let's say we have r_A common A-terms, and for each k, we have r_k B-terms.
    # Total rank = r_A + sum_k r_k.
    
    # For simplicity, let's try: r_A = 9, and for each k, r_k = 3. Total = 9 + 7*3 = 30.
    # That's still worse than schoolbook.
    
    # Better: Use the structure of matrix multiplication more directly.
    # C = A * B, where A is 3x3, B is 3x7.
    # We can think of this as 7 independent 3x3 * 3x1 products, but with shared A.
    
    # Let's try a different approach: use the known result that the tensor rank
    # of 3x3x7 MM is at most 24 (from literature or by construction).
    
    # Construct a rank-24 decomposition:
    # Use 21 rank-1 terms for the "main" part and 3 for corrections.
    
    # Actually, let's just do a smarter random search with structure.
    # Initialize with schoolbook and try to merge terms.
    
    # For now, let's try ranks from 21 to 35 with structured random search.
    lower_bound = max(n*m, m*p, p*n)
    
    # Try structured random search for ranks from lower_bound to 40
    for rank in range(lower_bound, min(lower_bound + 20, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # For each rank, try structured random candidates
        max_trials = 500
        found = False
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Generate structured random decomposition
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            rng = random.Random(trial * 7919 + rank * 104729)
            
            # For each rank term, assign random patterns
            for c in range(rank):
                # U: 3x3 grid, each entry is -1, 0, or 1
                for idx in range(n * m):
                    U[idx][c] = rng.choice([-1, 0, 1])
                # V: 3x7 grid
                for idx in range(m * p):
                    V[idx][c] = rng.choice([-1, 0, 1])
                # W: 7x3 grid
                for idx in range(p * n):
                    W[idx][c] = rng.choice([-1, 0, 1])
            
            if _try_early_exit(U, V, W, n, m, p):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
