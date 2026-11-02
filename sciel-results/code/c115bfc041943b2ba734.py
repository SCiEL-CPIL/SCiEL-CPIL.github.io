"""Search for a rank-minimal exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
Falls back to schoolbook (rank 80) if no better decomposition is found in time."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=4, m=4, p=5):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    Ua = np.array(U, dtype=object)
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = s + Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n=4, m=4, p=5):
    """Fast verification using numpy with float64, checking tolerance."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ur[:, :, c]
        vc = Vr[:, :, c]
        wc = Wr[:, :, c]
        wcT = wc.T
        for j in range(m):
            T[:, j, :] = T[:, j, :] + np.outer(uc[:, j], vc[j, :]) * wcT
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 as factor matrices.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Using the convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # U factors (for A): rows indexed by i*2+j
    # i=0,j=0 (a): M1, M3, M5, M6
    # i=0,j=1 (b): M3, M5, M7
    # i=1,j=0 (c): M2, M6
    # i=1,j=1 (d): M1, M2, M4, M6, M7
    
    U = np.zeros((4, 7), dtype=np.float64)
    # a (row 0): M1(+1), M3(+1), M5(+1), M6(-1)
    U[0, 0] = 1; U[0, 2] = 1; U[0, 4] = 1; U[0, 5] = -1
    # b (row 1): M3(+1), M5(+1), M7(+1)
    U[1, 2] = 1; U[1, 4] = 1; U[1, 6] = 1
    # c (row 2): M2(+1), M6(+1)
    U[2, 1] = 1; U[2, 5] = 1
    # d (row 3): M1(+1), M2(+1), M4(+1), M6(+1), M7(-1)
    U[3, 0] = 1; U[3, 1] = 1; U[3, 3] = 1; U[3, 5] = 1; U[3, 6] = -1
    
    # V factors (for B): rows indexed by j*2+k
    # j=0,k=0 (e): M1(+1), M2(+1), M4(-1), M6(+1)
    # j=0,k=1 (f): M1(+1), M3(-1), M6(+1)
    # j=1,k=0 (g): M1(+1), M4(+1), M7(+1)
    # j=1,k=1 (h): M1(+1), M5(+1), M7(+1)
    
    V = np.zeros((4, 7), dtype=np.float64)
    # e (row 0): M1(+1), M2(+1), M4(-1), M6(+1)
    V[0, 0] = 1; V[0, 1] = 1; V[0, 3] = -1; V[0, 5] = 1
    # f (row 1): M1(+1), M3(-1), M6(+1)
    V[1, 0] = 1; V[1, 2] = -1; V[1, 5] = 1
    # g (row 2): M1(+1), M4(+1), M7(+1)
    V[2, 0] = 1; V[2, 3] = 1; V[2, 6] = 1
    # h (row 3): M1(+1), M5(+1), M7(+1)
    V[3, 0] = 1; V[3, 4] = 1; V[3, 6] = 1
    
    # W factors (for C): rows indexed by k*2+i
    # C[0,0] (k=0,i=0): M1(+1), M4(+1), M5(-1), M7(+1)
    # C[0,1] (k=0,i=1): M3(+1), M5(+1)
    # C[1,0] (k=1,i=0): M2(+1), M4(+1)
    # C[1,1] (k=1,i=1): M1(+1), M2(-1), M3(+1), M6(+1)
    
    W = np.zeros((4, 7), dtype=np.float64)
    # C[0,0] (row 0): M1(+1), M4(+1), M5(-1), M7(+1)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C[0,1] (row 1): M3(+1), M5(+1)
    W[1, 2] = 1; W[1, 4] = 1
    # C[1,0] (row 2): M2(+1), M4(+1)
    W[2, 1] = 1; W[2, 3] = 1
    # C[1,1] (row 3): M1(+1), M2(-1), M3(+1), M6(+1)
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _block_strassen_4x4x4():
    """Construct a Strassen-based decomposition for 4x4 * 4x4 using block decomposition.
    Split into 2x2 blocks of 2x2 matrices.
    Returns U (16, rank), V (16, rank), W (16, rank)
    """
    U2, V2, W2 = _strassen_2x2()
    rank2 = 7
    rank = rank2 * rank2  # 49
    
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)
    
    # Block structure: 4x4 matrix split into 2x2 blocks of 2x2
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # For each pair of block indices (bi, bj) in {0,1}^2:
    # The contribution uses U2, V2, W2 with the 2x2 Strassen factors
    
    # Block (bi, bj) of A has rows: (bi*2 + i)*2 + j where i,j in {0,1}
    # Actually, let me re-index. A is 4x4 with indices (r, c) where r,c in 0..3
    # Block row bi, block col bj: rows bi*2..bi*2+1, cols bj*2..bj*2+1
    
    # U[i*4+j, c] where i is row of A, j is col of A
    # For block (bi, bj): i = bi*2 + ii, j = bj*2 + jj where ii, jj in {0,1}
    # U index: i*4 + j = (bi*2+ii)*4 + (bj*2+jj)
    
    # V[j*4+k, c] where j is row of B, k is col of B
    # For block (bj, bk): j = bj*2 + jj, k = bk*2 + kk
    # V index: j*4 + k = (bj*2+jj)*4 + (bk*2+kk)
    
    # W[k*4+i, c] where k is row of C, i is col of C  
    # Wait, W[k*n+i, c] where n=4. So W[k*4+i, c]
    # For block (bk, bi): k = bk*2 + kk, i = bi*2 + ii
    # W index: k*4 + i = (bk*2+kk)*4 + (bi*2+ii)
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                for c2 in range(rank2):
                    # U2[ii, c2] contributes to A block (bi, bj)
                    # V2[jj, c2] contributes to B block (bj, bk)  
                    # W2[kk, c2] contributes to C block (bk, bi)
                    # But we need to combine across the 2x2 block structure
                    
                    # Actually, the Strassen algorithm works on the 2x2 block matrix.
                    # Each "element" of the block matrix is a 2x2 matrix.
                    # So we apply Strassen to the block level, and within each block,
                    # we do standard 2x2 multiplication.
                    
                    # This is getting complex. Let me use a different approach.
                    pass
    
    # Simpler: use the Kronecker product structure
    # If we have a 2x2 Strassen decomposition (U2, V2, W2) and a 2x2 schoolbook,
    # the 4x4 decomposition has rank 7 * 4 = 28... but that's not right either.
    
    # Let me just do a direct block approach:
    # C = A * B where A, B, C are 4x4
    # Split into 2x2 blocks: A = [[A00, A01], [A10, A11]], etc.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term like A00*B00 is a 2x2 * 2x2 = 2x2 multiplication, rank 7 (Strassen)
    # So total rank = 4 * 7 = 28? No, because the terms are independent.
    # Actually C00 needs A00*B00 and A01*B10, each rank 7, so C00 needs 14 mults.
    # Total: 4 blocks * 14 = 56? That's worse than schoolbook 64... no, 56 < 64.
    
    # Wait, let me recount. For each output block Cxy, we need 2 products of 2x2 matrices.
    # Each 2x2 product needs 7 Strassen multiplications.
    # So each Cxy needs 14 multiplications.
    # 4 output blocks * 14 = 56 total multiplications.
    # Rank 56 < 64 (schoolbook). 
    
    # Let me construct this properly.
    U = np.zeros((16, 56), dtype=np.float64)
    V = np.zeros((16, 56), dtype=np.float64)
    W = np.zeros((16, 56), dtype=np.float64)
    
    col = 0
    for bi in range(2):  # output block row
        for bk in range(2):  # output block col
            # C[bi*2:bi*2+2, bk*2:bk*2+2] = A[bi*2:bi*2+2, 0:2] * B[0:2, bk*2:bk*2+2] 
            #                               + A[bi*2:bi*2+2, 2:4] * B[2:4, bk*2:bk*2+2]
            
            for aj in range(2):  # which A block column (0 or 1)
                # A block (bi, aj), B block (aj, bk)
                for c2 in range(rank2):
                    # U2[ii, c2] for A element at (bi*2+ii, aj*2+jj)
                    # V2[jj, c2] for B element at (aj*2+jj, bk*2+kk)
                    # W2[kk, c2] for C element at (bk*2+kk, bi*2+ii)
                    
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                # U index: (bi*2+ii)*4 + (aj*2+jj)
                                U[(bi*2+ii)*4 + (aj*2+jj), col] = U2[ii, c2]
                                # V index: (aj*2+jj)*4 + (bk*2+kk)
                                V[(aj*2+jj)*4 + (bk*2+kk), col] = V2[jj, c2]
                                # W index: (bk*2+kk)*4 + (bi*2+ii)
                                W[(bk*2+kk)*4 + (bi*2+ii), col] = W2[kk, c2]
                    col += 1
    
    return U, V, W

def _extend_to_4x4x5(U4, V4, W4, n=4, m=4, p=5):
    """Extend a 4x4x4 decomposition to 4x4x5 by adding one column.
    The extra column of B (col 4) multiplies with A to give an extra column of C.
    C[:, 4] = A * B[:, 4]
    This is a 4x4 * 4x1 = 4x1 multiplication, which is just 16 scalar mults (schoolbook).
    So we add 16 more terms.
    """
    rank4 = U4.shape[1]
    rank5 = rank4 + 16
    
    U5 = np.zeros((16, rank5), dtype=np.float64)
    V5 = np.zeros((20, rank5), dtype=np.float64)
    W5 = np.zeros((20, rank5), dtype=np.float64)
    
    # Copy the 4x4x4 part
    U5[:, :rank4] = U4
    V5[:16, :rank4] = V4
    W5[:16, :rank4] = W4
    
    # Add the extra column: C[i, 4] = sum_j A[i, j] * B[j, 4]
    # For each (i, j): one multiplication
    col = rank4
    for i in range(4):
        for j in range(4):
            U5[i*4+j, col] = 1
            V5[j*5+4, col] = 1  # B[j, 4] where B is 4x5, so index j*5+4
            W5[4*4+i, col] = 1  # C[4, i] where C is 5x4, so index 4*4+i
            col += 1
    
    return U5, V5, W5

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Try the block Strassen approach
    try:
        U4, V4, W4 = _block_strassen_4x4x4()
        if _verify_fast(U4, V4, W4, 4, 4, 4):
            U5, V5, W5 = _extend_to_4x4x5(U4, V4, W4, n, m, p)
            if _verify_fast(U5, V5, W5, n, m, p):
                return {"U": U5.tolist(), "V": V5.tolist(), "W": W5.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
