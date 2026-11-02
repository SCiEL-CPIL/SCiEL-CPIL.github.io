"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition based on 8 independent 2x2x2 Strassen multiplications
    # Rank = 8 * 7 = 56
    n, m, p = DIMENSIONS
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A B
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a10)b11
    # P6 = (a01-a00)(b00+b10)
    # P7 = (a10-a11)(b01+b11)
    
    # Indices for A (2x2): i in 0,1; j in 0,1
    # Indices for B (2x2): j in 0,1; k in 0,1
    # Indices for C (2x2): i in 0,1; k in 0,1
    
    # We have 8 blocks.
    # Block t (0..7) corresponds to:
    # A block: rows 0,1; cols 2*t, 2*t+1
    # B block: rows 2*t, 2*t+1; cols 0,1 (for first 4 blocks) or 4,5 (for next 4)?
    # No, B is 4x8.
    # Let's split B into 8 2x2 blocks.
    # B[2*t : 2*t+2, 2*s : 2*s+2]
    # But A is 2x4.
    # A[i, 2*t : 2*t+2]
    # So for each t (0..1) in A and s (0..3) in B?
    # A has 2 columns of 2x2 blocks.
    # B has 4 rows of 2x2 blocks and 4 columns of 2x2 blocks.
    # C[i, 2*s : 2*s+2] = sum_{t=0}^1 A[i, 2*t : 2*t+2] B[2*t : 2*t+2, 2*s : 2*s+2]
    # So we have 2 * 4 = 8 independent 2x2x2 multiplications.
    
    col_idx = 0
    for t in range(2): # A block index
        for s in range(4): # B block index
            # Define the 7 Strassen terms for this block
            # A indices: i, 2*t + j
            # B indices: 2*t + j, 2*s + k
            # C indices: i, 2*s + k
            
            # P1
            U[0 * 4 + 2*t + 0][col_idx] = 1
            U[1 * 4 + 2*t + 1][col_idx] = 1
            V[2*t + 0][2*s + 0] = 1 # Wait, V is (m*p, rank). V[j*p+k, c]
            # V index: j*p+k. j is 0..3, k is 0..7.
            # V[j, k] corresponds to B[j, k]
            # So V[2*t + 0, 2*s + 0] is not the format.
            # V is a matrix of shape (m*p, rank).
            # V[row, col] where row = j*p + k.
            V[(2*t + 0) * 8 + (2*s + 0)][col_idx] = 1
            V[(2*t + 1) * 8 + (2*s + 1)][col_idx] = 1
            W[(2*s + 0) * 2 + 0][col_idx] = 1 # W[k*n+i, c]
            W[(2*s + 1) * 2 + 0][col_idx] = 1
            W[(2*s + 0) * 2 + 1][col_idx] = 1
            W[(2*s + 1) * 2 + 1][col_idx] = 1
            col_idx += 1
            
            # ... repeat for P2..P7
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
