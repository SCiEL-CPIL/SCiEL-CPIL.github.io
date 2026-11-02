"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Construct rank-44 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Uses block-wise Strassen (rank 7) for the 2x2 core plus rank-4 for the
    remaining 2x1 * 1x2 part, repeated over 4 independent 2-column blocks.
    Total rank: 4 * (7 + 4) = 44.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 44
    
    # U has shape (n*m, rank) = (6, 44)
    # V has shape (m*p, rank) = (24, 44)
    # W has shape (p*n, rank) = (16, 44)
    # W index: W[k*n + i, col] where k is column of C (0..p-1), i is row of C (0..n-1)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Four independent 2-column blocks of B (and C)
    for b in range(4):
        k0 = 2 * b      # first column index in this block
        k1 = 2 * b + 1  # second column index in this block
        
        # --- Strassen rank-7 for A[0:2, 0:2] (2x2) * B[0:2, k0:k1+1] (2x2) ---
        # A indices: row i (0,1), col j (0,1) -> U[i*m + j]
        # B indices: row r (0,1), col c (k0,k1) -> V[r*p + c]
        # C indices: row i (0,1), col k (k0,k1) -> W[k*n + i]
        
        # S1 = (A00 + A11) * (B0k0 + B1k1)
        U[0*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1   # C[0,k0] += 1
        W[k1*n+1][col] = 1   # C[1,k1] += 1
        col += 1
        
        # S2 = (A10 + A11) * B0k0
        U[1*m+0][col] = 1
        U[1*m+1][col] = 1
        V[0*p+k0][col] = 1
        W[k0*n+1][col] = 1   # C[1,k0] += 1
        W[k1*n+1][col] = -1  # C[1,k1] -= 1
        col += 1
        
        # S3 = A00 * (B0k1 - B1k1)
        U[0*m+0][col] = 1
        V[0*p+k1][col] = 1
        V[1*p+k1][col] = -1
        W[k1*n+0][col] = 1   # C[0,k1] += 1
        W[k1*n+1][col] = 1   # C[1,k1] += 1
        col += 1
        
        # S4 = A11 * (B1k0 - B0k0)
        U[1*m+1][col] = 1
        V[1*p+k0][col] = 1
        V[0*p+k0][col] = -1
        W[k0*n+0][col] = 1   # C[0,k0] += 1
        W[k0*n+1][col] = 1   # C[1,k0] += 1
        col += 1
        
        # S5 = (A00 + A01) * B1k1
        U[0*m+0][col] = 1
        U[0*m+1][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = -1  # C[0,k0] -= 1
        W[k1*n+0][col] = 1   # C[0,k1] += 1
        col += 1
        
        # S6 = (A10 - A00) * (B0k0 + B0k1)
        U[1*m+0][col] = 1
        U[0*m+0][col] = -1
        V[0*p+k0][col] = 1
        V[0*p+k1][col] = 1
        W[k1*n+1][col] = 1   # C[1,k1] += 1
        col += 1
        
        # S7 = (A01 - A11) * (B1k0 + B1k1)
        U[0*m+1][col] = 1
        U[1*m+1][col] = -1
        V[1*p+k0][col] = 1
        V[1*p+k1][col] = 1
        W[k0*n+0][col] = 1   # C[0,k0] += 1
        col += 1
        
        # --- Rank-4 for A[0:2, 2] (2x1) * B[2, k0:k1+1] (1x2) ---
        # A indices: row i (0,1), col 2 -> U[i*m + 2]
        # B index: row 2, col k -> V[2*p + k]
        
        # P1 = A02 * B2k0
        U[0*m+2][col] = 1
        V[2*p+k0][col] = 1
        W[k0*n+0][col] = 1   # C[0,k0] += 1
        col += 1
        
        # P2 = A02 * B2k1
        U[0*m+2][col] = 1
        V[2*p+k1][col] = 1
        W[k1*n+0][col] = 1   # C[0,k1] += 1
        col += 1
        
        # P3 = A12 * B2k0
        U[1*m+2][col] = 1
        V[2*p+k0][col] = 1
        W[k0*n+1][col] = 1   # C[1,k0] += 1
        col += 1
        
        # P4 = A12 * B2k1
        U[1*m+2][col] = 1
        V[2*p+k1][col] = 1
        W[k1*n+1][col] = 1   # C[1,k1] += 1
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
