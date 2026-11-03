"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build the multiplication tensor T with modes (n*m), (m*p), (p*n)
    # T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 iff i==i', j==j', k==k'
    # Actually: T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    # T[a,b,c] = 1 iff j==j' and i==i' and k==k'
    
    # For rank computation, flatten to matrix: rows = mode1, cols = mode2*mode3
    # But tensor rank != matrix rank in general.
    
    # Alternative: use the "standard" approach for matrix multiplication tensors.
    # The rank of M_{n,m,p} can be found by solving for minimal rank decomposition.
    
    # Let's try to find a rank-20 decomposition explicitly.
    # For (2,3,6), known results suggest rank 20 might be achievable.
    
    # Construction approach: 
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # (indices: i in [0,1], j in [0,2], k in [0,5])
    
    # Let's use a structured approach. For each k, we have a 2x3 subproblem (A is 2x3, B is 3x1 column).
    # The product C[:,k] = A @ B[:,k] is a 2-vector.
    # We need to decompose this using shared terms across k values.
    
    # Idea: Use 20 terms. Let's try:
    # - 12 terms for "diagonal" contributions: for each (i,k), one term where U[i,*,c]=1, V[*,k,c]=1, W[k,i,c]=1
    #   But that doesn't work directly because V needs to match j.
    
    # Let me try a different construction. 
    # Terms 0-11: For each (i, j) pair (6 pairs), create terms that handle all k.
    #   For fixed (i,j): U[i,j,c]=1, V[j,k,c]=1 for all k, W[k,i,c]=1 for all k
    #   This gives contribution 1 to C[i,k] for all k, but we only want it for specific k.
    
    # Better: Let's use the following known construction for small matrix multiplication.
    # For (2,3,6), I'll attempt rank 20 with a systematic search.
    
    # Actually, let me just compute the rank by building the tensor and using rank finding.
    
    # Build tensor as 3D array: shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # To find tensor rank, we can try: for each candidate rank r, check if there exist
    # rank-1 tensors that sum to T. This is NP-hard in general, but for small sizes we can use
    # the flattening trick.
    
    # One flattening: matrix M of size (n*m) x (m*p * p*n)
    # M[a, (b, c)] = T[a, b, c]
    # The tensor rank is at least the matrix rank of any flattening.
    
    M_flat = T.reshape(n * m, m * p * p * n)
    # Compute rank over rationals using fraction arithmetic
    # Convert to Fraction matrix
    M_frac = np.array([[Fraction(x) for x in row] for row in M_flat.tolist()], dtype=object)
    
    # Gaussian elimination over Q
    def matrix_rank_Q(M):
        """Compute rank of matrix over rationals using Gaussian elimination."""
        M = [list(row) for row in M]
        m, n = len(M), len(M[0])
        rank = 0
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(rank, m):
                if M[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                continue
            # Swap rows
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
            # Eliminate below
            for row in range(m):
                if row != rank and M[row][col] != 0:
                    factor = M[row][col] / M[rank][col]
                    for j in range(col, n):
                        M[row][j] = M[row][j] - factor * M[rank][j]
            rank += 1
        return rank
    
    # This flattening gives a lower bound, but not the exact tensor rank.
    # For matrix multiplication tensors, the tensor rank is known for many small cases.
    # For (2,3,6), let me try to construct a rank-20 decomposition.
    
    # Known: rank(M_{2,3,6}) <= 20 via specific constructions.
    # Let me try to construct it.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0] * rank for _ in range(m * p)]  # 18 x 20
    W = [[0] * rank for _ in range(p * n)]  # 12 x 20
    
    # Construction for rank 20:
    # We'll use the following pattern:
    # For k in 0..5, j in 0..2, i in 0..1:
    # We need to cover all (i,j,k) triples.
    # 
    # Let's use 20 terms with the following structure:
    # Terms 0-5: For each k, a term that handles j=0, i=0
    # Terms 6-11: For each k, a term that handles j=0, i=1  
    # Terms 12-17: For each k, a term that handles j=1, both i's (using difference trick)
    # Terms 18-19: Handle j=2
    # 
    # This is getting complicated. Let me use a verified construction.
    
    # After analysis, I'll use the schoolbook rank 36 as a safe fallback,
    # but first let me try to verify if rank 20 works by construction.
    
    # Let me try a specific rank-20 construction:
    # Use the fact that for each k, the 2x3 matrix multiplication can be done in 7 terms (Strassen-like),
    # but we need to share across k.
    
    # Simpler: just output schoolbook for correctness.
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
