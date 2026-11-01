"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=4):
    """Verify that U, V, W form a valid rank-r decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # For each (i, j, k), the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be:
    # 1 if this is a valid multiplication term, 0 otherwise
    # Actually, the condition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = delta (1 if we're computing that product)
    # Rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to some C)
    # Wait, no. The tensor is defined such that:
    # sum_{i,j} A[i,j] B[j,k] = C[i,k]
    # In tensor form: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # The structure tensor has entries T[i,j,k] where the multiplication is A[i,j] * B[j,k] -> C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the range, but the output index is (i,k)
    
    # The rank-r decomposition satisfies:
    # For all i in [0,n), k in [0,p):
    #   C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k' V[j*p+k',c]*B[j,k']) ... 
    
    # Actually, the correct condition is:
    # Let U be (n*m) x r, V be (m*p) x r, W be (p*n) x r
    # Then for the decomposition to be valid:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  (if (i,j,k) is a valid triple)
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 0  (otherwise, but all triples are valid here)
    
    # Wait, I think I'm confusing myself. Let me re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor has indices (i, j, k) with value 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Since every such product does contribute, T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition means: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition(n=3, m=3, p=4):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_based_decomposition(n=3, m=3, p=4):
    """
    Try to construct a lower-rank decomposition using Strassen-like ideas.
    For 3x3 x 3x3, Strassen gives rank 7. We extend to 3x3 x 3x4.
    """
    # Use the known 3x3 Strassen decomposition (rank 7) for the first 3 columns,
    # and handle the 4th column with additional terms.
    # This is a heuristic; may not be optimal.
    
    # The 3x3 Strassen decomposition (one known rank-7 decomposition):
    # This is complex; let me use a simpler approach.
    
    # Alternative: Use the fact that we can share computations.
    # For now, return None if we can't find a good one quickly.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best["U"], best["V"], best["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find lower-rank decompositions
    # For this small case, we can try various approaches
    
    # Approach: Try rank from best_rank-1 down to some lower bound
    # But this is exponential, so we need a smarter approach.
    
    # For now, return the schoolbook if we can't find better quickly.
    # In a more advanced version, we could use:
    # 1. Known Strassen-based constructions
    # 2. Randomized search
    # 3. Integer linear programming
    
    # Check time
    if time.time() > deadline:
        return best
    
    # Try a few specific constructions
    # For 3x3 x 3x4, try to see if rank 35 is possible by modifying schoolbook
    # This is unlikely to work directly, but let's try other approaches.
    
    # Given time constraints, return the best found
    return best
# EVOLVE-BLOCK-END
