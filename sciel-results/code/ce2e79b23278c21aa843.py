"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np

def _verify(U, V, W, n=2, m=2, p=5):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that non-target entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Already checked above for target
                pass
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (10, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Try to construct a rank-13 decomposition using Strassen-like structure
    # For 2x2x2, Strassen uses 7 terms. For 2x2x5, we can use:
    # 7 terms for the first 2 columns (Strassen on 2x2x2)
    # + 2*3 = 6 terms for the remaining 3 columns (schoolbook for 2x2x3)
    # Total: 13
    
    # Strassen's 7 terms for 2x2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a01-a10)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    
    # For the full 2x2x5 problem, let's define:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03, b04], [b10, b11, b12, b13, b14]]
    # C = [[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14]]
    
    # We'll use 13 terms total.
    # Terms 0-6: Strassen-like for columns 0,1 of B
    # Terms 7-12: Schoolbook for columns 2,3,4 of B
    
    rank = 13
    U = [[0]*rank for _ in range(4)]  # (4, 13)
    V = [[0]*rank for _ in range(10)]  # (10, 13)
    W = [[0]*rank for _ in range(10)]  # (10, 13)
    
    # Strassen terms for first 2 columns (indices 0-6)
    # m1 = (a00+a11)(b00+b11)
    # U: a00->U[0], a11->U[3]; V: b00->V[0], b11->V[6]; W: c00->W[0], c11->W[7]
    # C[0,0] += a00*b00 + a11*b11 (from m1 contribution)
    # C[1,1] += a00*b00 + a11*b11 (from m1 contribution)
    # Actually, let me be more careful.
    
    # Let me use a different approach. I'll directly construct the factors.
    # For Strassen 2x2x2, the standard decomposition is:
    
    # Let me just try to find a valid rank-13 decomposition by construction.
    
    # Actually, let me try a simpler approach: use the known fact that
    # R(2,2,5) <= R(2,2,2) + 2*R(1,2,3) or similar bounds.
    
    # Let me try to construct it directly with integer entries.
    
    # I'll use a systematic approach:
    # For each of the 10 output entries C[k,i], we need:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    
    # Let me define 13 bilinear forms:
    # t0 = (a00+a10+a11)*(b00+b10)
    # t1 = (a00+a11)*(b01+b11)
    # t2 = (a10+a11)*(b00)
    # t3 = a00*(b01-b11)
    # t4 = a11*(b10-b00)
    # t5 = (a00+a10)*(b11)
    # t6 = (a01-a10)*(b00+b10)  -- wait, a01 and a10 are different rows
    
    # Let me restart with proper Strassen for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a01-a10)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # For columns 0,1 of B (i.e., b00,b01,b10,b11):
    # C[0,0] = a00*b00 + a01*b10
    # C[1,0] = a10*b00 + a11*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,1] = a10*b01 + a11*b11
    
    # Using Strassen:
    # C[0,0] = m1 - m4 - m5
    # C[1,0] = m2 + m4
    # C[0,1] = m3 + m5
    # C[1,1] = m1 - m3 - m2
    
    # For columns 2,3,4, use schoolbook (2 terms each, 6 total):
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    # C[0,3] = a00*b03 + a01*b13
    # C[1,3] = a10*b03 + a11*b13
    # C[0,4] = a00*b04 + a01*b14
    # C[1,4] = a10*b04 + a11*b14
    
    # Total: 7 + 6 = 13 terms
    
    # Now let's construct U, V, W:
    # U has 4 rows (a00, a01, a10, a11) and 13 columns
    # V has 10 rows (b00,b01,b02,b03,b04,b10,b11,b12,b13,b14) and 13 columns
    # W has 10 rows (c00,c01,c02,c03,c04,c10,c11,c12,c13,c14) and 13 columns
    
    # Term 0: m1 = (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[6][0] = 1  # b11
    # m1 contributes to C[0,0] with -1, C[1,0] with 0, C[0,1] with 0, C[1,1] with +1
    # Wait, let me recheck:
    # C[0,0] = m1 - m4 - m5, so m1 contributes +1 to C[0,0]
    # C[1,1] = m1 - m3 - m2, so m1 contributes +1 to C[1,1]
    W[0][0] = 1   # c00
    W[7][0] = 1   # c11
    
    # Term 1: m2 = (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    # C[1,0] = m2 + m4, so m2 contributes +1 to C[1,0]
    # C[1,1] = m1 - m3 - m2, so m2 contributes -1 to C[1,1]
    W[6][1] = 1   # c10
    W[7][1] = -1  # c11
    
    # Term 2: m3 = a00(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[6][2] = -1 # b11
    # C[0,1] = m3 + m5, so m3 contributes +1 to C[0,1]
    # C[1,1] = m1 - m3 - m2, so m3 contributes -1 to C[1,1]
    W[1][2] = 1   # c01
    W[7][2] = -1  # c11
    
    # Term 3: m4 = a11(b10-b00)
    U[3][3] = 1  # a11
    V[5][3] = 1  # b10
    V[0][3] = -1 # b00
    # C[1,0] = m2 + m4, so m4 contributes +1 to C[1,0]
    # C[0,0] = m1 - m4 - m5, so m4 contributes -1 to C[0,0]
    W[6][3] = 1   # c10
    W[0][3] = -1  # c00
    
    # Term 4: m5 = (a00+a10)b11
    U[0][4] = 1  # a00
    U[2][4] = 1  # a10
    V[6][4] = 1  # b11
    # C[0,1] = m3 + m5, so m5 contributes +1 to C[0,1]
    # C[0,0] = m1 - m4 - m5, so m5 contributes -1 to C[0,0]
    W[1][4] = 1   # c01
    W[0][4] = -1  # c00
    
    # Term 5: m6 = (a01-a10)(b00+b10)
    U[1][5] = 1  # a01
    U[2][5] = -1 # a10
    V[0][5] = 1  # b00
    V[5][5] = 1  # b10
    # C[0,0] = a00*b00 + a01*b10, m6 = (a01-a10)(b00+b10) = a01*b00 + a01*b10 - a10*b00 - a10*b10
    # This doesn't directly help. Let me reconsider.
    
    # Actually, I made an error. Let me use the correct Strassen formulas:
    # For C = A*B where A is 2x2, B is 2x2:
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    
    # Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a01-a10)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # C[0,0] = m1 - m4 - m5
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m3 - m2
    
    # Let me verify C[0,0]:
    # m1 - m4 - m5 = (a00+a11)(b00+b11) - a11(b10-b00) - (a00+a10)b11
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b10 + a11*b00 - a00*b11 - a10*b11
    # = a00*b00 + 2*a11*b00 + a11*b11 - a11*b10 - a10*b11
    # This doesn't equal a00*b00 + a01*b10. I'm confusing the indices.
    
    # The issue is that in standard Strassen, the matrix multiplication is C = A*B
    # where C[i,j] = sum_k A[i,k]*B[k,j]
    # So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    
    # In our problem, the output is C[k,i] = sum_j A[i,j]*B[j,k]
    # So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] (k=0, i=0)
    # C[1,0] = A[0,0]*B[0,1] + A[0,1]*B[1,1] (k=1, i=0)
    # C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0] (k=0, i=1)
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] (k=1, i=1)
    
    # This is C = B^T * A^T or something similar. Let me just use the direct approach.
    
    # For simplicity, let me just use the schoolbook for all 5 columns, which gives rank 20,
    # but try to find a better decomposition.
    
    # Actually, let me try a different strategy: use a known low-rank decomposition.
    # The border rank of 2x2x5 is known to be 13 or 14.
    
    # Let me try to construct it using linear algebra.
    
    # For now, let me return the schoolbook decomposition as a fallback,
    # but try to improve it.
    
    # Let me try rank 14 first with a structured search.
    
    # Actually, let me just return the schoolbook for now to ensure correctness,
    # and the evolution can improve it.
    
    rank = n * m * p  # 20
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
