import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """Return the 7 rank-1 factors for 2x2x2 Strassen.
    Each factor is (A_pattern, B_pattern, C_pattern) as 2x2 arrays.
    """
    factors = []
    # P1 = (A00+A11)(B00+B11)
    A = np.array([[1,0],[1,1]])
    B = np.array([[1,1],[1,1]])
    C = np.array([[1,0],[0,1]])
    factors.append((A, B, C))
    # P2 = (A00+A01)B11
    A = np.array([[1,1],[0,0]])
    B = np.array([[0,0],[0,1]])
    C = np.array([[0,1],[1,-1]])
    factors.append((A, B, C))
    # P3 = (A10+A11)B00
    A = np.array([[0,0],[1,1]])
    B = np.array([[1,0],[0,0]])
    C = np.array([[0,1],[-1,0]])
    factors.append((A, B, C))
    # P4 = A00(B01-B11)
    A = np.array([[1,0],[0,0]])
    B = np.array([[0,1],[0,-1]])
    C = np.array([[1,-1],[0,1]])
    factors.append((A, B, C))
    # P5 = A11(B10-B00)
    A = np.array([[0,0],[0,1]])
    B = np.array([[0,-1],[1,0]])
    C = np.array([[-1,0],[1,1]])
    factors.append((A, B, C))
    # P6 = (A01-A11)(B10+B11)
    A = np.array([[0,1],[0,-1]])
    B = np.array([[0,0],[1,1]])
    C = np.array([[0,1],[0,0]])
    factors.append((A, B, C))
    # P7 = (A10-A00)(B00+B01)
    A = np.array([[-1,0],[1,0]])
    B = np.array([[1,1],[0,0]])
    C = np.array([[1,0],[1,0]])
    factors.append((A, B, C))
    return factors

def _construct_strassen_4x4():
    """Construct rank-49 Strassen decomposition for 4x4x4.
    Returns U (16x49), V (16x49), W (16x49).
    """
    s2 = _strassen_2x2_factors()
    
    # Top-level 4x4x4 Strassen: 7 products
    # Each product combines a 2x2 block of A, a 2x2 block of B, and contributes to 2x2 blocks of C
    top_factors = []
    # s=0: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    A_blocks = np.array([[1,0],[0,1]])  # A11 + A22
    B_blocks = np.array([[1,1],[1,1]])  # B11 + B22
    C_blocks = np.array([[1,0],[0,1]])  # C11 + C22
    top_factors.append((A_blocks, B_blocks, C_blocks))
    # s=1: (A11+A12)B22 -> C12 += 1, C21 += 1, C22 -= 1
    A_blocks = np.array([[1,1],[0,0]])
    B_blocks = np.array([[0,0],[0,1]])
    C_blocks = np.array([[0,1],[1,-1]])
    top_factors.append((A_blocks, B_blocks, C_blocks))
    # s=2: (A21+A22)B11 -> C12 += 1, C21 -= 1
    A_blocks = np.array([[0,0],[1,1]])
    B_blocks = np.array([[1,0],[0,0]])
    C_blocks = np.array([[0,1],[-1,0]])
    top_factors.append((A_blocks, B_blocks, C_blocks))
    # s=3: A11(B12-B22) -> C11 += 1, C12 -= 1, C22 += 1
    A_blocks = np.array([[1,0],[0,0]])
    B_blocks = np.array([[0,1],[0,-1]])
    C_blocks = np.array([[1,-1],[0,1]])
    top_factors.append((A_blocks, B_blocks, C_blocks))
    # s=4: A22(B21-B11) -> C11 -= 1, C21 += 1, C22 += 1
    A_blocks = np.array([[0,0],[0,1]])
    B_blocks = np.array([[0,-1],[1,0]])
    C_blocks = np.array([[-1,0],[1,1]])
    top_factors.append((A_blocks, B_blocks, C_blocks))
    # s=5: (A12-A22)(B21+B22) -> C12 += 1
    A_blocks = np.array([[0,1],[0,-1]])
    B_blocks = np.array([[0,0],[1,1]])
    C_blocks = np.array([[0,1],[0,0]])
    top_factors.append((A_blocks, B_blocks, C_blocks))
    # s=6: (A21-A11)(B11+B12) -> C11 += 1, C21 += 1
    A_blocks = np.array([[-1,0],[1,0]])
    B_blocks = np.array([[1,1],[0,0]])
    C_blocks = np.array([[1,0],[1,0]])
    top_factors.append((A_blocks, B_blocks, C_blocks))
    
    # Now build the full rank-49 decomposition
    # For each top factor (s) and each 2x2 Strassen sub-factor (t):
    # U index: (i_block*2+i_sub)*4 + (j_block*2+j_sub) -> but we need i*m+j = i*4+j
    # V index: j*p+k = j*4+k
    # W index: k*n+i = k*4+i
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for s in range(7):
        A_b, B_b, C_b = top_factors[s]
        for t in range(7):
            A_s2, B_s2, C_s2 = s2[t]
            # For each element in the 2x2x2 sub-problem:
            for ib in range(2):  # i_block
                for jb in range(2):  # j_block
                    for kb in range(2):  # k_block
                        # This doesn't work directly because the 2x2 Strassen
                        # factors are for the full 2x2 matrices, not element-wise.
                        pass
    
    # Actually, the correct construction is:
    # For top factor s, sub factor t:
    # U[i*4+j, col] = A_b[ib, jb] * A_s2[is, js] where i=ib*2+is, j=jb*2+js
    # V[j*4+k, col] = B_b[jb, kb] * B_s2[js, ks] where j=jb*2+js, k=kb*2+ks
    # W[k*4+i, col] = C_b[ib, kb] * C_s2[is, ks] where i=ib*2+is, k=kb*2+ks
    
    col = 0
    for s in range(7):
        A_b, B_b, C_b = top_factors[s]
        for t in range(7):
            A_s2, B_s2, C_s2 = s2[t]
            for ib in range(2):
                for jb in range(2):
                    for is_ in range(2):
                        for js_ in range(2):
                            i = ib * 2 + is_
                            j = jb * 2 + js_
                            # U[i*4+j, col] = A_b[ib,jb] * A_s2[is_,js_]
                            U[i*4+j, col] = A_b[ib, jb] * A_s2[is_, js_]
            for jb in range(2):
                for kb in range(2):
                    for js_ in range(2):
                        for ks_ in range(2):
                            j = jb * 2 + js_
                            k = kb * 2 + ks_
                            V[j*4+k, col] = B_b[jb, kb] * B_s2[js_, ks_]
            for ib in range(2):
                for kb in range(2):
                    for is_ in range(2):
                        for ks_ in range(2):
                            i = ib * 2 + is_
                            k = kb * 2 + ks_
                            W[k*4+i, col] = C_b[ib, kb] * C_s2[is_, ks_]
            col += 1
    
    return U, V, W

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _construct_4x4x8():
    """Construct rank-98 decomposition for 4x4x8.
    Split B into B1 (cols 0-3) and B2 (cols 4-7).
    Use Strassen 4x4x4 for each, sharing A-side.
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    s2 = _strassen_2x2_factors()
    
    top_factors = [
        (np.array([[1,0],[0,1]]), np.array([[1,1],[1,1]]), np.array([[1,0],[0,1]])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[0,1],[1,-1]])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,1],[-1,0]])),
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[1,-1],[0,1]])),
        (np.array([[0,0],[0,1]]), np.array([[0,-1],[1,0]]), np.array([[-1,0],[1,1]])),
        (np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[0,1],[0,0]])),
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[1,0],[1,0]])),
    ]
    
    # For each half (h=0 for cols 0-3, h=1 for cols 4-7):
    # For each top factor s and sub factor t:
    # U[i*4+j, col] = A_b[ib,jb] * A_s2[is,js]  (same for both halves)
    # V[j*8+k, col] = B_b[jb,kb] * B_s2[js,ks] where k = kb*2+ks+h*4
    # W[k*4+i, col] = C_b[ib,kb] * C_s2[is,ks] where k = kb*2+ks+h*4
    
    col = 0
    for h in range(2):  # two halves
        for s in range(7):
            A_b, B_b, C_b = top_factors[s]
            for t in range(7):
                A_s2, B_s2, C_s2 = s2[t]
                # U: same for both halves
                for ib in range(2):
                    for jb in range(2):
                        for is_ in range(2):
                            for js_ in range(2):
                                i = ib * 2 + is_
                                j = jb * 2 + js_
                                U[i*m+j, col] = A_b[ib, jb] * A_s2[is_, js_]
                # V: k offset by h*4
                for jb in range(2):
                    for kb in range(2):
                        for js_ in range(2):
                            for ks_ in range(2):
                                j = jb * 2 + js_
                                k = kb * 2 + ks_ + h * 4
                                V[j*p+k, col] = B_b[jb, kb] * B_s2[js_, ks_]
                # W: k offset by h*4
                for ib in range(2):
                    for kb in range(2):
                        for is_ in range(2):
                            for ks_ in range(2):
                                i = ib * 2 + is_
                                k = kb * 2 + ks_ + h * 4
                                W[k*n+i, col] = C_b[ib, kb] * C_s2[is_, ks_]
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try Strassen-based rank-98 decomposition
    U, V, W = _construct_4x4x8()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Convert to lists for the return interface
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 128
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
