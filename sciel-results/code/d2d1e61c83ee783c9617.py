import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Strategy: Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x10) and B2 (1x10).
    # C = A1 B1 + A2 B2.
    # Rank(A1 B1) = Rank(M_{2,2,10}) = 5 * Rank(M_{2,2,2}) = 5 * 7 = 35.
    # Rank(A2 B2) = Rank(M_{2,1,10}) = 2 * 10 = 20.
    # Total Rank = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Part 1: M_{2,2,10} using Strassen
    # A indices: 0,1 (row 0), 2,3 (row 1) for first 2 cols.
    # B indices: 0..9 (row 0), 10..19 (row 1) for first 2 rows.
    # C indices: 0..9 (row 0), 10..19 (row 1).
    
    # Strassen coefficients for M_{2,2,2}
    # Mult k corresponds to:
    # U_A (coeffs for a11, a12, a21, a22)
    # U_B (coeffs for b11, b12, b21, b22)
    # W_C (coeffs for c11, c12, c21, c22)
    
    # Standard Strassen:
    # M1: (a11+a22)(b11+b22) -> c11 += M1, c22 += M1
    # M2: (a21+a22)b11       -> c21 += M2, c22 -= M2
    # M3: a11(b12-b22)       -> c12 += M3
    # M4: a22(b21-b11)       -> c11 += M4, c21 += M4
    # M5: (a11+a12)b22       -> c12 += M5, c11 -= M5
    # M6: (a21-a11)(b11+b12) -> c22 += M6
    # M7: (a12-a22)(b21+b22) -> c22 -= M7
    
    # Note: The standard Strassen formula varies. Let's use a consistent one.
    # c11 = M1 - M5 + M4
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M6 - M7 (Wait, M7 is usually (a12-a22)(b21+b22)?)
    # Let's verify M7.
    # a12 b21 + a12 b22 - a22 b21 - a22 b22.
    # c22 needs a12 b22 + a21 b21 + ...
    # Actually, let's just use the known rank 7 decomposition.
    
    # Indices for A (2x2 block):
    # a00: U[0], a01: U[1], a10: U[2], a11: U[3]
    # Indices for B (2x2 block):
    # b00: V[0], b01: V[1], b10: V[10], b11: V[11] (Note: V is 3x10, so row 1 is 10..19)
    # Indices for C (2x2 block):
    # c00: W[0], c01: W[1], c10: W[10], c11: W[11]
    
    # We have 5 blocks of B (cols 0-1, 2-3, 4-5, 6-7, 8-9).
    # For each block, we apply the 7 Strassen multiplications.
    
    col_offset = 0
    for block in range(5):
        # B indices for this block:
        # b00: V[col_offset]
        # b01: V[col_offset + 1]
        # b10: V[10 + col_offset]
        # b11: V[10 + col_offset + 1]
        
        # C indices for this block:
        # c00: W[col_offset]
        # c01: W[col_offset + 1]
        # c10: W[10 + col_offset]
        # c11: W[10 + col_offset + 1]
        
        # 7 Multiplications
        # 1: (a00+a11)(b00+b11)
        U[0][col_offset] = 1; U[3][col_offset] = 1
        V[col_offset][col_offset] = 1; V[10+col_offset][col_offset] = 1
        W[col_offset][col_offset] = 1; W[10+col_offset+1][col_offset] = 1
        col_offset += 1
        
        # 2: (a10+a11)b00
        U[2][col_offset] = 1; U[3][col_offset] = 1
        V[col_offset][col_offset] = 1
        W[10+col_offset][col_offset] = 1; W[10+col_offset+1][col_offset] = -1
        col_offset += 1
        
        # 3: a00(b01-b11)
        U[0][col_offset] = 1
        V[col_offset+1][col_offset] = 1; V[10+col_offset+1][col_offset] = -1
        W[col_offset+1][col_offset] = 1
        col_offset += 1
        
        # 4: a11(b10-b00)
        U[3][col_offset] = 1
        V[10+col_offset][col_offset] = 1; V[col_offset][col_offset] = -1
        W[col_offset][col_offset] = 1; W[10+col_offset][col_offset] = 1
        col_offset += 1
        
        # 5: (a00+a01)b11
        U[0][col_offset] = 1; U[1][col_offset] = 1
        V[10+col_offset+1][col_offset] = 1
        W[col_offset+1][col_offset] = 1; W[col_offset][col_offset] = -1
        col_offset += 1
        
        # 6: (a10-a00)(b00+b01)
        U[2][col_offset] = 1; U[0][col_offset] = -1
        V[col_offset][col_offset] = 1; V[col_offset+1][col_offset] = 1
        W[10+col_offset+1][col_offset] = 1
        col_offset += 1
        
        # 7: (a01-a11)(b10+b11)
        U[1][col_offset] = 1; U[3][col_offset] = -1
        V[10+col_offset][col_offset] = 1; V[10+col_offset+1][col_offset] = 1
        W[10+col_offset+1][col_offset] = -1
        col_offset += 1

    # Part 2: M_{2,1,10}
    # A col 2: a02 (U[4]), a12 (U[5])
    # B row 2: b20..b29 (V[20]..V[29])
    # C: c00..c09 (W[0]..W[9]), c10..c19 (W[10]..W[19])
    
    # 10 mults for a02 * b2k
    for k in range(10):
        U[4][col_offset] = 1
        V[20+k][col_offset] = 1
        W[k][col_offset] = 1
        col_offset += 1
        
    # 10 mults for a12 * b2k
    for k in range(10):
        U[5][col_offset] = 1
        V[20+k][col_offset] = 1
        W[10+k][col_offset] = 1
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
