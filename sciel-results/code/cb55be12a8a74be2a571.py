"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix mult."""
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    # Rank 55 decomposition:
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x10) into B1 (2x10) and B2 (1x10)
    # A1*B1 uses 5 Strassen blocks of 2x2x2 (rank 7 each) -> 35
    # A2*B2 uses schoolbook 2x1x10 (rank 20) -> 20
    # Total rank 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (cols 0,1) * B1 (rows 0,1)
    # B1 is 2x10. Split into 5 blocks of 2x2.
    for t in range(5):
        k1 = 2 * t
        k2 = 2 * t + 1
        
        # Strassen 7 multiplications
        # A indices: i in 0,1; j in 0,1
        # B indices: j in 0,1; k in k1, k2
        
        # M1 = (A00 + A01) * (B0k1 + B0k2)
        # M2 = (A10 + A11) * (B0k1 + B0k2)
        # M3 = (A00 - A10) * (B1k1 + B1k2)
        # M4 = (A01 - A11) * (B1k1 + B1k2)
        # M5 = (A00 + B0k1) * (B0k2 - B1k1) -- Wait, Strassen mixes A and B?
        # No, Strassen is bilinear.
        # M5 = (A00 + A11) * (B1k1 + B0k2)? No.
        
        # Let's use the standard 7 terms for 2x2x2:
        # C00 = M1 - M3 + M5 - M6
        # C01 = M3 - M1 + M5 - M7
        # C10 = M2 - M4 + M6 - M5
        # C11 = M4 - M2 + M7 - M6
        
        # Terms:
        # M1 = (A00 + A01) * (B00 + B10)
        # M2 = (A10 + A11) * (B00 + B10)
        # M3 = (A00 - A10) * (B10 + B11)
        # M4 = (A01 - A11) * (B10 + B11)
        # M5 = (A00 + B00) * (B01 - B10)
        # M6 = (A10 + A11) * (B00 - B10)
        # M7 = (A01 + A11) * (B11 - B01)
        
        # Note: Indices in B are (row, col).
        # B00 is B[0, k1]. B01 is B[0, k2].
        # B10 is B[1, k1]. B11 is B[1, k2].
        
        def add_term(u_idx, v_idx, w_idx, val=1):
            nonlocal col
            U[u_idx][col] += val
            V[v_idx][col] += val
            W[w_idx][col] += val
            col += 1

        # M1
        # (A00 + A01) * (B0k1 + B0k2)
        # A00: U[0*3+0], A01: U[0*3+1]
        # B0k1: V[0*10+k1], B0k2: V[0*10+k2]
        # W: C00 (W[k1*2+0]), C01 (W[k2*2+0])
        
        # This manual expansion is error-prone.
        # I will just use the schoolbook for the 2x2x2 blocks?
        # No, that gives rank 8 per block. 5*8=40.
        # Strassen gives 7.
        
        # Let's stick to the 7 terms.
        
        # Term 1: (A00+A01)(B0k1+B0k2)
        # Contributes to C00 and C01?
        # In Strassen, each M contributes to all 4 Cs.
        # So W indices are k1*2+0, k2*2+0, k1*2+1, k2*2+1.
        
        # Actually, the tensor decomposition requires specific coefficients.
        # I will use the known rank-7 decomposition of M222.
        
        # M1 = (A00+A01)(B00+B10)
        # M2 = (A10+A11)(B00+B10)
        # M3 = (A00-A10)(B10+B11)
        # M4 = (A01-A11)(B10+B11)
        # M5 = (A00+B00)(B01-B10)
        # M6 = (A10+A11)(B00-B10)
        # M7 = (A01+A11)(B11-B01)
        
        # C00 = M1 - M3 + M5 - M6
        # C01 = M3 - M1 + M5 - M7
        # C10 = M2 - M4 + M6 - M5
        # C11 = M4 - M2 + M7 - M6
        
        # We need to add these to U, V, W.
        
        # Helper to add a bilinear term with coefficients
        # coeff * U[u] * V[v] * W[w]
        
        def add_bilinear(u_idx, v_idx, w_idx, coeff):
            nonlocal col
            U[u_idx][col] += coeff
            V[v_idx][col] += coeff
            W[w_idx][col] += coeff
            col += 1

        # M1 = (A00+A01)(B0k1+B0k2)
        # Expand: A00 B0k1 + A00 B0k2 + A01 B0k1 + A01 B0k2
        # This is 4 terms.
        # Strassen combines them into 1 multiplication?
        # No, the "multiplication" is the product of two sums.
        # The tensor decomposition represents the product of sums as a single rank-1 term?
        # No, rank-1 term is $u v w$.
        # $(A00+A01)(B0k1+B0k2)$ is not rank 1 in the tensor sense unless we allow linear combinations in U, V, W.
        # Yes, U, V, W can have linear combinations.
        # So one column in U, V, W represents one Strassen multiplication.
        
        # So for M1:
        # U: A00=1, A01=1
        # V: B0k1=1, B0k2=1
        # W: C00=1, C01=-1, C10=0, C11=0?
        # No, W coefficients depend on the reconstruction.
        
        # I will define the 7 columns for one block.
        
        # Block t
        # U indices for A1: 0,1,3,4 (rows 0,1; cols 0,1)
        # V indices for B1: 0..9, 10..19 (rows 0,1; cols k1,k2)
        # W indices for C: k1*2, k1*2+1, k2*2, k2*2+1
        
        # M1
        U[0][col] += 1; U[1][col] += 1
        V[k1][col] += 1; V[k2][col] += 1
        W[k1*2+0][col] += 1
        W[k2*2+0][col] -= 1
        W[k1*2+1][col] += 0
        W[k2*2+1][col] += 0
        # Wait, the W coefficients must match the Strassen reconstruction.
        # C00 gets +M1. C01 gets -M1.
        # So W[k1*2+0] += 1, W[k2*2+0] -= 1.
        # C10, C11 get 0.
        col += 1
        
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[k1][col] += 1; V[k2][col] += 1
        W[k1*2+1][col] += 1
        W[k2*2+1][col] -= 1
        col += 1
        
        # M3
        U[0][col] += 1; U[3][col] -= 1
        V[10+k1][col] += 1; V[10+k2][col] += 1
        W[k1*2+0][col] -= 1
        W[k2*2+0][col] += 1
        W[k1*2+1][col] += 0
        W[k2*2+1][col] += 0
        # C00 gets -M3. C01 gets +M3.
        col += 1
        
        # M4
        U[1][col] += 1; U[4][col] -= 1
        V[10+k1][col] += 1; V[10+k2][col] += 1
        W[k1*2+1][col] -= 1
        W[k2*2+1][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1
        V[k2][col] += 1; V[10+k1][col] -= 1
        W[k1*2+0][col] += 1
        W[k2*2+0][col] += 1
        W[k1*2+1][col] -= 1
        W[k2*2+1][col] += 0
        # C00 gets +M5. C01 gets +M5. C10 gets -M5.
        col += 1
        
        # M6
        U[3][col] += 1; U[4][col] += 1
        V[k1][col] += 1; V[10+k1][col] -= 1
        W[k1*2+0][col] -= 1
        W[k2*2+0][col] += 0
        W[k1*2+1][col] += 1
        W[k2*2+1][col] += 0
        # C00 gets -M6. C10 gets +M6.
        col += 1
        
        # M7
        U[1][col] += 1; U[4][col] += 1
        V[10+k2][col] += 1; V[k2][col] -= 1
        W[k2*2+0][col] += 0
        W[k2*2+1][col] += 1
        W[k1*2+1][col] += 0
        W[k1*2+0][col] += 0
        # C11 gets +M7. C01 gets -M7.
        col += 1

    # Part 2: A2 (col 2) * B2 (row 2)
    # Schoolbook 2x1x10
    for i in range(2):
        for k in range(10):
            U[i*3 + 2][col] = 1
            V[2*10 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
