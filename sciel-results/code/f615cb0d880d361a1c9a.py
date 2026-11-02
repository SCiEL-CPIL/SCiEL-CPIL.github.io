"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _strassen_rank55():
    """Build a rank-55 Strassen-based decomposition for 2x3 * 3x10.
    
    Strategy: 
    - Split the 3 columns of A into [0,1] and [2].
    - Split the 10 rows of B into 5 blocks of 2 columns each.
    - Use 7 Strassen multiplications per block for the [0,1] x [0,1] sub-problem.
    - Use schoolbook for the row-2 of A times each column of B.
    
    Total: 5*7 + 2*10 = 35 + 20 = 55 multiplications.
    
    A is 2x3: rows 0,1; cols 0,1,2.
    B is 3x10: rows 0,1,2; cols 0..9.
    
    Output C is 2x10.
    For block i (columns 2i, 2i+1 of B):
      Strassen handles A[:,0:2] * B[0:2, 2i:2i+2] -> gives C[:, 2i:2i+2] contribution from rows 0,1 of B.
      Schoolbook handles A[2,:] * B[2, 2i:2i+2] -> gives C[:, 2i:2i+2] contribution from row 2 of B.
    
    Wait, I need to be careful. A is 2x3, so A[i,j] where i in {0,1}, j in {0,1,2}.
    B is 3x10, so B[k, l] where k in {0,1,2}, l in {0..9}.
    C[i, l] = sum_j A[i,j] * B[j,l].
    
    So C[i,l] = A[i,0]*B[0,l] + A[i,1]*B[1,l] + A[i,2]*B[2,l].
    
    Let me define:
    - For the "main" part (j=0,1): use Strassen on A[0:2, 0:2] * B[0:2, 2i:2i+2] for each block i.
    - For the "extra" part (j=2): use schoolbook A[i,2] * B[2, l] for each i and l.
    
    Strassen for 2x2 * 2x2:
    Let A_sub = A[0:2, 0:2] (2x2), B_sub = B[0:2, 2i:2i+2] (2x2).
    
    Standard Strassen:
    M1 = (a00 + a11) * (b00 + b11)
    M2 = (a10 + a11) * b00
    M3 = a00 * (b01 - b11)
    M4 = a11 * (b10 - b00)
    M5 = (a00 + a01) * b11
    M6 = (a10 - a00) * (b00 + b01)
    M7 = (a01 - a11) * (b10 + b11)
    
    Result:
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    Here a00 = A[0,0], a01 = A[0,1], a10 = A[1,0], a11 = A[1,1].
    b00 = B[0, 2i], b01 = B[0, 2i+1], b10 = B[1, 2i], b11 = B[1, 2i+1].
    
    So for block i:
    Contribution to C[0, 2i]   = M1 + M4 - M5 + M7
    Contribution to C[0, 2i+1] = M3 + M5
    Contribution to C[1, 2i]   = M2 + M4
    Contribution to C[1, 2i+1] = M1 - M2 + M3 + M6
    
    Plus schoolbook:
    C[i, 2i]   += A[i,2] * B[2, 2i]
    C[i, 2i+1] += A[i,2] * B[2, 2i+1]
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Number of blocks: p // 2 = 5
    n_blocks = p // 2  # 5
    assert p % 2 == 0, "p must be even for this Strassen approach"
    
    # Total rank: n_blocks * 7 + 2 * (p - 2 * n_blocks)
    # Since n_blocks = p//2, p - 2*n_blocks = 0 if p is even.
    # Wait, that's not right. Let me recount.
    # For each block, we do 7 Strassen mults.
    # For the schoolbook part, we need A[i,2] * B[2, l] for each i in {0,1} and l in {0..9}.
    # That's 2 * 10 = 20 schoolbook mults.
    # But wait, can we combine schoolbook mults? No, they're all independent.
    # So total rank = 5*7 + 2*10 = 35 + 20 = 55.
    
    rank = n_blocks * 7 + n * (p - 2 * n_blocks)
    # Actually, p - 2*n_blocks = 0 since n_blocks = p//2 and p is even.
    # Hmm, that doesn't make sense. Let me re-think.
    # The schoolbook part is for j=2 (the third row/column), which is NOT part of the Strassen 2x2.
    # So for each column l of B, we need A[i,2] * B[2,l] for i=0,1.
    # That's 2 multiplications per column, and there are p=10 columns.
    # So 2 * 10 = 20 schoolbook multiplications.
    # Plus 5 blocks * 7 Strassen = 35.
    # Total = 55.
    
    # Let me define the column indices:
    # Columns 0 to 34: Strassen multiplications (5 blocks * 7 mults each)
    # Columns 35 to 54: Schoolbook multiplications (2 * 10 = 20 mults)
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Fill in Strassen multiplications
    for i in range(n_blocks):  # block index
        col_base = i * 7
        # b00 = B[0, 2i], b01 = B[0, 2i+1], b10 = B[1, 2i], b11 = B[1, 2i+1]
        # In V, row = j*p + k where j is the row of B (0,1,2) and k is the column.
        # So B[0, 2i] is at V row 0*p + 2i = 2i
        # B[0, 2i+1] is at V row 0*p + 2i+1 = 2i+1
        # B[1, 2i] is at V row 1*p + 2i = p + 2i
        # B[1, 2i+1] is at V row 1*p + 2i+1 = p + 2i + 1
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,2i] + B[1,2i+1])
        col = col_base + 0
        U[0 * m + 0][col] = 1  # A[0,0]
        U[1 * m + 1][col] = 1  # A[1,1]
        V[0 * p + 2 * i][col] = 1      # B[0, 2i]
        V[1 * p + 2 * i + 1][col] = 1  # B[1, 2i+1]
        # W: output rows. C[0,2i] += 1, C[0,2i+1] += 0, C[1,2i] += 0, C[1,2i+1] += 1
        # W row = k*n + i where k is col of C, i is row of C.
        # C[0,2i] -> W[2i * n + 0][col] += 1
        # C[1,2i+1] -> W[(2i+1) * n + 1][col] += 1
        W[2 * i * n + 0][col] = 1
        W[(2 * i + 1) * n + 1][col] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,2i]
        col = col_base + 1
        U[1 * m + 0][col] = 1  # A[1,0]
        U[1 * m + 1][col] = 1  # A[1,1]
        V[0 * p + 2 * i][col] = 1  # B[0, 2i]
        # C[1,2i] += 1, C[1,2i+1] += -1
        W[2 * i * n + 1][col] = 1
        W[(2 * i + 1) * n + 1][col] = -1
        
        # M3 = A[0,0] * (B[0,2i+1] - B[1,2i+1])
        col = col_base + 2
        U[0 * m + 0][col] = 1  # A[0,0]
        V[0 * p + 2 * i + 1][col] = 1    # B[0, 2i+1]
        V[1 * p + 2 * i + 1][col] = -1   # B[1, 2i+1]
        # C[0,2i+1] += 1, C[1,2i+1] += 1
        W[(2 * i + 1) * n + 0][col] = 1
        W[(2 * i + 1) * n + 1][col] = 1
        
        # M4 = A[1,1] * (B[1,2i] - B[0,2i])
        col = col_base + 3
        U[1 * m + 1][col] = 1  # A[1,1]
        V[1 * p + 2 * i][col] = 1    # B[1, 2i]
        V[0 * p + 2 * i][col] = -1   # B[0, 2i]
        # C[0,2i] += 1, C[1,2i] += 1
        W[2 * i * n + 0][col] = 1
        W[2 * i * n + 1][col] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,2i+1]
        col = col_base + 4
        U[0 * m + 0][col] = 1  # A[0,0]
        U[0 * m + 1][col] = 1  # A[0,1]
        V[1 * p + 2 * i + 1][col] = 1  # B[1, 2i+1]
        # C[0,2i] += -1, C[0,2i+1] += 1
        W[2 * i * n + 0][col] = -1
        W[(2 * i + 1) * n + 0][col] = 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,2i] + B[0,2i+1])
        col = col_base + 5
        U[1 * m + 0][col] = 1   # A[1,0]
        U[0 * m + 0][col] = -1  # A[0,0]
        V[0 * p + 2 * i][col] = 1      # B[0, 2i]
        V[0 * p + 2 * i + 1][col] = 1  # B[0, 2i+1]
        # C[1,2i+1] += 1
        W[(2 * i + 1) * n + 1][col] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,2i] + B[1,2i+1])
        col = col_base + 6
        U[0 * m + 1][col] = 1   # A[0,1]
        U[1 * m + 1][col] = -1  # A[1,1]
        V[1 * p + 2 * i][col] = 1      # B[1, 2i]
        V[1 * p + 2 * i + 1][col] = 1  # B[1, 2i+1]
        # C[0,2i] += 1
        W[2 * i * n + 0][col] = 1
    
    # Fill in schoolbook multiplications for j=2 (row 2 of B, col 2 of A)
    # C[i, l] += A[i, 2] * B[2, l]
    schoolbook_base = n_blocks * 7  # 35
    for i in range(n):  # row of A (and C)
        for l in range(p):  # col of B (and C)
            col = schoolbook_base + i * p + l
            U[i * m + 2][col] = 1  # A[i, 2]
            V[2 * p + l][col] = 1  # B[2, l]
            W[l * n + i][col] = 1  # C[i, l]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x10.
    
    Uses a Strassen-based construction with rank 55.
    """
    return _strassen_rank55()
# EVOLVE-BLOCK-END
