"""Rewritten program exploiting algebraic structure for 2x2 * 2x13 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    """Fallback: schoolbook decomposition of rank 52."""
    n, m, p = DIMENSIONS
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


def _strassen_2x2_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7) with proper indexing.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    Standard Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    c00 = M1+M4-M5+M7
    c01 = M3+M5
    c10 = M2+M4
    c11 = M1-M2+M3+M6
    """
    U = [
        # a00, a01, a10, a11
        [1, 0, 0, 0, 1, -1, 0],   # a00: M1, M3, M5, M6
        [0, 1, 0, 0, 1, 0, -1],   # a01: M3, M5, M7
        [0, 0, 1, 1, 0, -1, 0],   # a10: M2, M6
        [1, 0, 1, 1, 0, 0, -1],   # a11: M1, M2, M4, M7
    ]
    V = [
        # b00, b01, b10, b11
        [1, 0, 0, 0, 0, 1, 0],    # b00: M1, M2, M6
        [0, 1, 0, 0, 0, 1, 0],    # b01: M3, M6
        [0, 0, 1, 0, 0, 0, 1],    # b10: M4, M7
        [1, -1, 0, 1, 1, 0, 1],   # b11: M1, M3, M5, M7
    ]
    W = [
        # c00, c01, c10, c11
        [1, 0, 0, -1, 1, 0, 1],   # c00: M1, M4, M5, M7
        [0, 1, 0, 0, 1, 0, 0],    # c01: M3, M5
        [0, 0, 1, 1, 0, 0, 0],    # c10: M2, M4
        [1, 0, 0, 0, 0, 1, 0],    # c11: M1, M2, M3, M6
    ]
    
    # Verify all conditions
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Check that for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # and for all other (i,j,k,i2) with i2 != i, the sum is 0
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a = i * 2 + j
                b = j * 2 + k
                for i2 in range(2):
                    c = k * 2 + i2
                    val = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                    expected = 1.0 if i2 == i else 0.0
                    if abs(val - expected) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook rank-8 for 2x2*2x2
        U_fb = [[0]*8 for _ in range(4)]
        V_fb = [[0]*8 for _ in range(4)]
        W_fb = [[0]*8 for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_fb[i*2+j][col] = 1
                    V_fb[j*2+k][col] = 1
                    W_fb[k*2+i][col] = 1
                    col += 1
        return U_fb, V_fb, W_fb
    
    return U, V, W


def _schoolbook_2x2_2x1():
    """Schoolbook decomposition for 2x2 * 2x1 -> 2x1.
    Returns U (4x4), V (2x4), W (2x4) with proper indexing.
    U[i*2+j, c], V[j*1+k, c] = V[j, c], W[k*2+i, c]
    """
    U = [
        [1, 0, 0, 0],  # a00
        [0, 1, 0, 0],  # a01
        [0, 0, 1, 0],  # a10
        [0, 0, 0, 1],  # a11
    ]
    V = [
        [1, 0, 1, 0],  # b0
        [0, 1, 0, 1],  # b1
    ]
    W = [
        [1, 1, 0, 0],  # c0
        [0, 0, 1, 1],  # c1
    ]
    return U, V, W


def _rank10_2x2_2x3():
    """
    Rank-10 decomposition for 2x2 * 2x3 -> 2x3.
    Returns U (4x10), V (6x10), W (6x10).
    U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]
    """
    # Systematic construction: combine Strassen (rank 7 for cols 0,1) with 
    # 3 additional terms for col 2 that share structure with Strassen.
    # Total rank: 7 + 3 = 10
    
    # Strassen for 2x2 * 2x2 (cols 0,1):
    U7 = [
        [1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [0, 1, 0, 1, 1, -1, 0],
    ]
    V7 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 1, 0, 1],
    ]
    W7 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # For col 2, we need 3 additional rank-1 terms.
    # The col-2 part of the tensor: for each (i,j), we need sum over k=2 of 
    # U[i,j,c]*V[j,2,c]*W[2,i,c] = 1 for all i,j.
    # We add 3 columns (indices 7,8,9) with values:
    # Try: U_extra = [[0],[0],[1],[0]], V_extra = [[1],[1],[0],[0]], W_extra = [[0],[1],[0],[1]]
    # This gives: c00: U[0,0]*V[0,2]*W[2,0] + U[0,0]*V[0,2]*W[2,0]... 
    # Let me just try a simple schoolbook-like extension for col 2 with 4 terms, 
    # then see if we can reduce to 3.
    
    # Actually, let's use a different strategy: rank 11 (Strassen + schoolbook col 2)
    # and see if we can shave off one rank.
    
    # For now, return None to use the rank-11 fallback in construct()
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Strategy: Use rank-10 for a 2x2*2x3 block and rank-7 Strassen for 2x2*2x2 blocks
    # 13 = 3 + 2*5, so one block of 3 (rank 10) + five blocks of 2 (rank 7 each)
    # Total: 10 + 5*7 = 45
    
    # Or: 13 = 2*6 + 1, six blocks of 2 (rank 7) + one block of 1 (rank 4) = 46
    # Or: 13 = 3 + 2*5, one block of 3 (rank 10) + five blocks of 2 (rank 7) = 45
    
    # Let's try the rank-45 approach if the rank-10 decomposition is valid
    
    def _strassen_plus_schoolbook_col():
        """Build rank-11 decomposition: Strassen for cols 0,1 + schoolbook for col 2."""
        U7, V7, W7 = _strassen_2x2_2x2()
        rank = 11
        U11 = [[0] * rank for _ in range(4)]
        V11 = [[0] * rank for _ in range(6)]
        W11 = [[0] * rank for _ in range(6)]
        
        # First 7 columns: Strassen for cols 0,1
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U11[i * 2 + j][c] = U7[i * 2 + j][c]
            for j in range(2):
                for kl in range(2):
                    V11[j * 3 + kl][c] = V7[j * 2 + kl][c]
            for kl in range(2):
                for i in range(2):
                    W11[kl * 2 + i][c] = W7[kl * 2 + i][c]
        
        # Last 4 columns: schoolbook for col 2
        # c00: U[0,0]=1, V[0,2]=1, W[2,0]=1
        # c01: U[0,1]=1, V[0,2]=1, W[2,1]=1  -- wait, this doesn't work
        # Schoolbook for 2x2 * 2x1 -> 2x1:
        # (a00*b0, a01*b1) -> c0 = a00*b0 + a01*b1
        # (a10*b0, a11*b1) -> c1 = a10*b0 + a11*b1
        # So for col 2 (k=2):
        # c[2,0] = a00*b[0,2] + a01*b[1,2]
        # c[2,1] = a10*b[0,2] + a11*b[1,2]
        
        # Term 7: a00 * b[0,2] -> c[2,0]
        U11[0][7] = 1  # a00
        V11[0*3+2][7] = 1  # b[0,2]
        W11[2*2+0][7] = 1  # c[2,0]
        
        # Term 8: a01 * b[1,2] -> c[2,0]
        U11[1][8] = 1  # a01
        V11[1*3+2][8] = 1  # b[1,2]
        W11[2*2+0][8] = 1  # c[2,0]
        
        # Term 9: a10 * b[0,2] -> c[2,1]
        U11[2][9] = 1  # a10
        V11[0*3+2][9] = 1  # b[0,2]
        W11[2*2+1][9] = 1  # c[2,1]
        
        # Term 10: a11 * b[1,2] -> c[2,1]
        U11[3][10] = 1  # a11
        V11[1*3+2][10] = 1  # b[1,2]
        W11[2*2+1][10] = 1  # c[2,1]
        
        return U11, V11, W11
    
    def _try_rank45():
        """Try to build a rank-45 decomposition using one 2x3 block (rank 10) + five 2x2 blocks (rank 7 each)."""
        if time.time() > deadline - 30:
            return None
        
        r10 = _rank10_2x2_2x3()
        if r10 is None:
            return None
        
        U10, V10, W10 = r10
        U7, V7, W7 = _strassen_2x2_2x2()
        
        rank = 10 + 5 * 7  # 45
        U_full = [[0] * rank for _ in range(n * m)]
        V_full = [[0] * rank for _ in range(m * p)]
        W_full = [[0] * rank for _ in range(p * n)]
        
        col_offset = 0
        
        # First block: columns 0, 1, 2 using rank-10 decomposition
        for c in range(10):
            for i in range(2):
                for j in range(2):
                    U_full[i * m + j][col_offset + c] = U10[i * 2 + j][c]
            for j in range(2):
                for kl in range(3):
                    k = kl
                    V_full[j * p + k][col_offset + c] = V10[j * 3 + kl][c]
            for kl in range(3):
                k = kl
                for i in range(2):
                    W_full[k * n + i][col_offset + c] = W10[kl * 2 + i][c]
        col_offset += 10
        
        # Next five blocks: columns 3-4, 5-6, 7-8, 9-10, 11-12 using Strassen
        for block in range(5):
            k0 = 3 + block * 2
            k1 = k0 + 1
            for c in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[i * m + j][col_offset + c] = U7[i * 2 + j][c]
                for j in range(2):
                    for kl in range(2):
                        k = k0 + kl
                        V_full[j * p + k][col_offset + c] = V7[j * 2 + kl][c]
                for kl in range(2):
                    k = k0 + kl
                    for i in range(2):
                        W_full[k * n + i][col_offset + c] = W7[kl * 2 + i][c]
            col_offset += 7
        
        # Verify
        Ua = np.array(U_full, dtype=float)
        Va = np.array(V_full, dtype=float)
        Wa = np.array(W_full, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                    if abs(val - 1) > 1e-9:
                        return None
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    def _try_rank47():
        """Build rank-47 decomposition: one 2x3 block (rank 11) + five 2x2 blocks (rank 7 each)."""
        U11, V11, W11 = _strassen_plus_schoolbook_col()
        U7, V7, W7 = _strassen_2x2_2x2()
        r7 = len(U7[0])
        
        rank = 11 + 5 * r7
        U_full = [[0] * rank for _ in range(n * m)]
        V_full = [[0] * rank for _ in range(m * p)]
        W_full = [[0] * rank for _ in range(p * n)]
        
        col_offset = 0
        
        # First block: columns 0, 1, 2 using rank-11 decomposition
        for c in range(11):
            for i in range(2):
                for j in range(2):
                    U_full[i * m + j][col_offset + c] = U11[i * 2 + j][c]
            for j in range(2):
                for kl in range(3):
                    k = kl
                    V_full[j * p + k][col_offset + c] = V11[j * 3 + kl][c]
            for kl in range(3):
                k = kl
                for i in range(2):
                    W_full[k * n + i][col_offset + c] = W11[kl * 2 + i][c]
        col_offset += 11
        
        # Next five blocks: columns 3-4, 5-6, 7-8, 9-10, 11-12 using Strassen
        for block in range(5):
            k0 = 3 + block * 2
            k1 = k0 + 1
            for c in range(r7):
                for i in range(2):
                    for j in range(2):
                        U_full[i * m + j][col_offset + c] = U7[i * 2 + j][c]
                for j in range(2):
                    for kl in range(2):
                        k = k0 + kl
                        V_full[j * p + k][col_offset + c] = V7[j * 2 + kl][c]
                for kl in range(2):
                    k = k0 + kl
                    for i in range(2):
                        W_full[k * n + i][col_offset + c] = W7[kl * 2 + i][c]
            col_offset += r7
        
        # Verify
        Ua = np.array(U_full, dtype=float)
        Va = np.array(V_full, dtype=float)
        Wa = np.array(W_full, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                    if abs(val - 1) > 1e-9:
                        return None
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    def _try_rank46():
        """Build rank-46 decomposition: six 2x2 Strassen blocks + one 2x1 schoolbook block."""
        U7, V7, W7 = _strassen_2x2_2x2()
        r7 = len(U7[0])  # rank of the 2x2 block (7 or 4)
        U4, V4, W4 = _schoolbook_2x2_2x1()
        
        rank = 6 * r7 + 4
        U_full = [[0] * rank for _ in range(n * m)]
        V_full = [[0] * rank for _ in range(m * p)]
        W_full = [[0] * rank for _ in range(p * n)]
        
        col_offset = 0
        for t in range(6):
            k0 = 2 * t
            k1 = 2 * t + 1
            for c in range(r7):
                for i in range(2):
                    for j in range(2):
                        U_full[i * m + j][col_offset + c] = U7[i * 2 + j][c]
                for j in range(2):
                    for kl in range(2):
                        k = k0 + kl
                        V_full[j * p + k][col_offset + c] = V7[j * 2 + kl][c]
                for kl in range(2):
                    k = k0 + kl
                    for i in range(2):
                        W_full[k * n + i][col_offset + c] = W7[kl * 2 + i][c]
            col_offset += r7
        
        k0 = 12
        for c in range(4):
            for i in range(2):
                for j in range(2):
                    U_full[i * m + j][col_offset + c] = U4[i * 2 + j][c]
            for j in range(2):
                V_full[j * p + k0][col_offset + c] = V4[j][c]
            for i in range(2):
                W_full[k0 * n + i][col_offset + c] = W4[i][c]
        
        # Verify
        Ua = np.array(U_full, dtype=float)
        Va = np.array(V_full, dtype=float)
        Wa = np.array(W_full, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(Ua[a, :] * Va[b, :] * Wa[c, :])
                    if abs(val - 1) > 1e-9:
                        return None
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Try rank 45 first (best if rank-10 works)
    result = _try_rank45()
    if result is not None:
        return result
    
    # Try rank 46 (six Strassen + one schoolbook)
    result = _try_rank46()
    if result is not None:
        return result
    
    # Try rank 47 (one rank-11 block + five Strassen)
    result = _try_rank47()
    if result is not None:
        return result
    
    # Final fallback: schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
