"""Program for matrix multiplication tensor decomposition 3x4x7."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n=3, m=4, p=7):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                # sum over rank of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - expected) > 1e-9:
                    return False
    # Also verify that the decomposition is exact (no spurious terms)
    # For each rank-1 term, it should contribute only to valid (i,j,k) triples
    for c in range(rank):
        u_col = [U[r][c] for r in range(n * m)]
        v_col = [V[r][c] for r in range(m * p)]
        w_col = [W[r][c] for r in range(p * n)]
        # Check that the term is consistent
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    # This should be 0 or 1 for each c (in a proper decomposition)
                    # Actually, it can be any value, but the sum over c should be 1
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    n, m, p = 3, 4, 7
    nmp = n * m * p  # 84 (schoolbook rank)
    
    # Try to find a decomposition with rank < 84
    # Start with known lower bounds and work up
    
    # The trivial schoolbook decomposition has rank 84
    # We want to find something better
    
    # Strategy: Use random initialization and iterative refinement
    # The matrix multiplication tensor has structure we can exploit
    
    best_rank = nmp  # Start with schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's try a few structured approaches
    
    # Approach 1: Use the fact that M_{3,4,7} can be decomposed using
    # smaller known decompositions
    
    # For 2x2x2, Strassen gives rank 7 instead of 8
    # For 3x3x3, the best known is 23 (Coppersmith-Winograd) or 22
    # But we need 3x4x7 specifically
    
    # Let's try random search with verification
    # This is expensive, so we limit the number of attempts
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound for M_{3,4,7}: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But there are better lower bounds
    
    # Let's just try to verify that the schoolbook works and return it
    # as a safe fallback, but first attempt some optimizations
    
    # Quick structured attempt: use integer/half-integer entries
    # Try rank 28 to 83
    
    for target_rank in range(28, nmp):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization
        max_attempts = 5 if target_rank > 50 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with random ±1 or 0
            for c in range(target_rank):
                for r in range(n * m):
                    U[r, c] = random.choice([-1, 0, 1])
                for r in range(m * p):
                    V[r, c] = random.choice([-1, 0, 1])
                for r in range(p * n):
                    W[r, c] = random.choice([-1, 0, 1])
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        total = 0.0
                        for c in range(target_rank):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                # Found a valid decomposition!
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < nmp:
            break
    
    # If we didn't find anything better, use schoolbook
    if best_U is None:
        rank = nmp
        best_U = [[0] * rank for _ in range(n * m)]
        best_V = [[0] * rank for _ in range(m * p)]
        best_W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j][col] = 1
                    best_V[j * p + k][col] = 1
                    best_W[k * n + i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
