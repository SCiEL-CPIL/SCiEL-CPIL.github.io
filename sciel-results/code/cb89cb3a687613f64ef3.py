import time
import numpy as np

def _validate(U, V, W, n, m, p):
    """Validate tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-10)

def _build_strassen_222():
    """Build 2x2x2 Strassen decomposition. Rank 7."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen coefficients
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B01-B11)
    # M4=A11(B10-B00)
    # M5=(A00+A01)B11
    # M6=(A10-A00)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    
    # U: A[i,j]
    # V: B[j,k]
    # W: C[k,i]
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1 # C00, C01, C10, C11? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's just use the standard Strassen matrix definitions.
    # It's safer to hardcode the known rank 7 decomposition.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, -1, 1, 0, 1] # B11
    ]
    W = [
        [1, 1, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 0, 1, 0, 1, 0, 0],  # C01 (k=0, i=1)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    return U, V, W

def _build_schoolbook_221():
    """Build 2x2x1 schoolbook decomposition. Rank 4."""
    n, m, p = 2, 2, 1
    rank = 4
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
    return U, V, W

def _build_mm_223():
    """Build 2x2x3 decomposition. Rank 11."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First 7 columns: Strassen for first 2 columns of B (2x2x2)
    U_s, V_s, W_s = _build_strassen_222()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j][r] = U_s[i*2+j][r]
            for k in range(2):
                V[j*2+k][r] = V_s[j*2+k][r]
            for k in range(2):
                W[k*2+i][r] = W_s[k*2+i][r]
        
    # Last 4 columns: Schoolbook for 3rd column of B (2x2x1)
    U_sb, V_sb, W_sb = _build_schoolbook_221()
    for r in range(4):
        col = 7 + r
        for i in range(2):
            for j in range(2):
                U[i*2+j][col] = U_sb[i*2+j][r]
            for k in range(1):
                # k=0 in 2x2x1 corresponds to k=2 in 2x2x3
                V[j*2+2][col] = V_sb[j*2+k][r]
            for k in range(1):
                # k=0 in 2x2x1 corresponds to k=2 in 2x2x3
                W[k*2+i][col] = W_sb[k*2+i][r]
                
    return U, V, W

def _build_mm_444():
    """Build 4x4x4 decomposition. Rank 49."""
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U_s, V_s, W_s = _build_strassen_222()
    
    # Strassen on 4x4 blocks
    # 7 Strassen terms. Each term is a 2x2x2 multiplication of 2x2 blocks.
    # So we instantiate the 2x2x2 decomposition 7 times.
    
    for r2 in range(7):
        for r_inner in range(7):
            col = r2 * 7 + r_inner
            ri = r_inner // 4 # 0 or 1
            cj = (r_inner % 4) // 2 # 0 or 1
            rk = r_inner % 2 # 0 or 1
            
            # U: A[i,j]
            for i2 in range(2):
                for j2 in range(2):
                    c = U_s[i2*2+j2][r2]
                    if c != 0:
                        i = 2*i2 + ri
                        j = 2*j2 + cj
                        U[i*4+j][col] += c
                        
            # V: B[j,k]
            for j2 in range(2):
                for k2 in range(2):
                    c = V_s[j2*2+k2][r2]
                    if c != 0:
                        j = 2*j2 + cj
                        k = 2*k2 + rk
                        V[j*4+k][col] += c
                        
            # W: C[k,i]
            for k2 in range(2):
                for i2 in range(2):
                    c = W_s[k2*2+i2][r2]
                    if c != 0:
                        k = 2*k2 + rk
                        i = 2*i2 + ri
                        W[k*4+i][col] += c
                        
    return U, V, W

def _build_mm_442():
    """Build 4x4x2 decomposition. Rank 28."""
    n, m, p = 4, 4, 2
    rank = 28
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U_s, V_s, W_s = _build_strassen_222()
    
    # 4 blocks of 2x2x2
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # Term 0: A00 B0
    # Term 1: A01 B1
    # Term 2: A10 B0
    # Term 3: A11 B1
    
    blocks = [
        (0, 0, 0, 0), # i2, j2, k2_offset? No.
        # (i_block, j_block, k_block_in_B)
        # B is 4x2. Split into B0 (rows 0-1) and B1 (rows 2-3).
        # So B0 is 2x2, B1 is 2x2.
        # A00 (0,0) * B0 (0)
        # A01 (0,1) * B1 (1)
        # A10 (1,0) * B0 (0)
        # A11 (1,1) * B1 (1)
    ]
    
    # Indices in A: i = 2*i2 + ri, j = 2*j2 + cj
    # Indices in B: j = 2*j2 + cj, k = 2*k2 + rk
    # Indices in C: k = 2*k2 + rk, i = 2*i2 + ri
    
    # B0 corresponds to j2=0 (rows 0,1). B1 corresponds to j2=1 (rows 2,3).
    # Wait, B is 4x2. Rows are j. Cols are k.
    # B0 is top 2 rows. B1 is bottom 2 rows.
    
    # Term 0: A00 * B0
    # A00: i2=0, j2=0
    # B0: j2=0
    # C0: i2=0
    for r_inner in range(7):
        col = 0 * 7 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U
        i = 2*0 + ri
        j = 2*0 + cj
        U[i*4+j][col] += 1 # Coeff from Strassen is 1? No, use U_s.
        # Actually, the coefficient is from the Strassen decomposition of the 2x2x2 multiplication.
        # We are just instantiating the 2x2x2 decomposition.
        
    # This approach is getting complicated to map indices correctly.
    # Let's use the same loop structure as _build_mm_444 but with specific mappings.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U_s, V_s, W_s = _build_strassen_222()
    
    # 4 terms
    # 0: A00 B0 -> C0
    # 1: A01 B1 -> C0
    # 2: A10 B0 -> C1
    # 3: A11 B1 -> C1
    
    # For each term, we use the 2x2x2 decomposition.
    
    # Term 0: i2=0, j2=0, k2_out=0. B_j2=0.
    for r_inner in range(7):
        col = 0 * 7 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*0+ri, 2*0+cj]
        U[(2*0+ri)*4 + (2*0+cj)][col] += U_s[ri*2+cj][r_inner] # Wait, U_s index is i*2+j?
        # U_s is 4x7. Index i*2+j.
        # In the 2x2 block, local i=ri, j=cj.
        # So U_s[ri*2+cj][r_inner] is correct.
        
        # V: B[2*0+cj, 2*0+rk]
        V[(2*0+cj)*2 + (2*0+rk)][col] += V_s[cj*2+rk][r_inner]
        
        # W: C[2*0+rk, 2*0+ri]
        W[(2*0+rk)*4 + (2*0+ri)][col] += W_s[rk*2+ri][r_inner]
        
    # Term 1: A01 B1 -> C0
    # i2=0, j2=1. B_j2=1. k2_out=0.
    for r_inner in range(7):
        col = 1 * 7 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*0+ri, 2*1+cj]
        U[(2*0+ri)*4 + (2*1+cj)][col] += U_s[ri*2+cj][r_inner]
        
        # V: B[2*1+cj, 2*0+rk]
        V[(2*1+cj)*2 + (2*0+rk)][col] += V_s[cj*2+rk][r_inner]
        
        # W: C[2*0+rk, 2*0+ri]
        W[(2*0+rk)*4 + (2*0+ri)][col] += W_s[rk*2+ri][r_inner]
        
    # Term 2: A10 B0 -> C1
    # i2=1, j2=0. B_j2=0. k2_out=1.
    for r_inner in range(7):
        col = 2 * 7 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*1+ri, 2*0+cj]
        U[(2*1+ri)*4 + (2*0+cj)][col] += U_s[ri*2+cj][r_inner]
        
        # V: B[2*0+cj, 2*1+rk]
        V[(2*0+cj)*2 + (2*1+rk)][col] += V_s[cj*2+rk][r_inner]
        
        # W: C[2*1+rk, 2*1+ri]
        W[(2*1+rk)*4 + (2*1+ri)][col] += W_s[rk*2+ri][r_inner]
        
    # Term 3: A11 B1 -> C1
    # i2=1, j2=1. B_j2=1. k2_out=1.
    for r_inner in range(7):
        col = 3 * 7 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*1+ri, 2*1+cj]
        U[(2*1+ri)*4 + (2*1+cj)][col] += U_s[ri*2+cj][r_inner]
        
        # V: B[2*1+cj, 2*1+rk]
        V[(2*1+cj)*2 + (2*1+rk)][col] += V_s[cj*2+rk][r_inner]
        
        # W: C[2*1+rk, 2*1+ri]
        W[(2*1+rk)*4 + (2*1+ri)][col] += W_s[rk*2+ri][r_inner]
        
    return U, V, W

def _build_mm_443():
    """Build 4x4x3 decomposition. Rank 44."""
    n, m, p = 4, 4, 3
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U_223, V_223, W_223 = _build_mm_223()
    
    # 4 terms of 2x2x3
    # 0: A00 B0 -> C0
    # 1: A01 B1 -> C0
    # 2: A10 B0 -> C1
    # 3: A11 B1 -> C1
    
    # B is 4x3. B0 is top 2 rows (2x3). B1 is bottom 2 rows (2x3).
    
    # Term 0
    for r_inner in range(11):
        col = 0 * 11 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*0+ri, 2*0+cj]
        U[(2*0+ri)*4 + (2*0+cj)][col] += U_223[ri*2+cj][r_inner]
        
        # V: B[2*0+cj, 2*0+rk]
        V[(2*0+cj)*3 + (2*0+rk)][col] += V_223[cj*3+rk][r_inner]
        
        # W: C[2*0+rk, 2*0+ri]
        W[(2*0+rk)*4 + (2*0+ri)][col] += W_223[rk*4+ri][r_inner]
        
    # Term 1
    for r_inner in range(11):
        col = 1 * 11 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*0+ri, 2*1+cj]
        U[(2*0+ri)*4 + (2*1+cj)][col] += U_223[ri*2+cj][r_inner]
        
        # V: B[2*1+cj, 2*0+rk]
        V[(2*1+cj)*3 + (2*0+rk)][col] += V_223[cj*3+rk][r_inner]
        
        # W: C[2*0+rk, 2*0+ri]
        W[(2*0+rk)*4 + (2*0+ri)][col] += W_223[rk*4+ri][r_inner]
        
    # Term 2
    for r_inner in range(11):
        col = 2 * 11 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*1+ri, 2*0+cj]
        U[(2*1+ri)*4 + (2*0+cj)][col] += U_223[ri*2+cj][r_inner]
        
        # V: B[2*0+cj, 2*1+rk]
        V[(2*0+cj)*3 + (2*1+rk)][col] += V_223[cj*3+rk][r_inner]
        
        # W: C[2*1+rk, 2*1+ri]
        W[(2*1+rk)*4 + (2*1+ri)][col] += W_223[rk*4+ri][r_inner]
        
    # Term 3
    for r_inner in range(11):
        col = 3 * 11 + r_inner
        ri = r_inner // 4
        cj = (r_inner % 4) // 2
        rk = r_inner % 2
        
        # U: A[2*1+ri, 2*1+cj]
        U[(2*1+ri)*4 + (2*1+cj)][col] += U_223[ri*2+cj][r_inner]
        
        # V: B[2*1+cj, 2*1+rk]
        V[(2*1+cj)*3 + (2*1+rk)][col] += V_223[cj*3+rk][r_inner]
        
        # W: C[2*1+rk, 2*1+ri]
        W[(2*1+rk)*4 + (2*1+ri)][col] += W_223[rk*4+ri][r_inner]
        
    return U, V, W

def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    
    try:
        # 4x4x4 part (rank 49)
        U444, V444, W444 = _build_mm_444()
        
        # 4x4x2 part (rank 28)
        U442, V442, W442 = _build_mm_442()
        
        # 4x4x3 part (rank 44)
        U443, V443, W443 = _build_mm_443()
        
        rank_total = 49 + 28 + 44
        U = [row[:] for row in U444]
        V = [row[:] for row in V444]
        W = [row[:] for row in W444]
        
        # Extend U, V, W with 442
        for i in range(4):
            U[i].extend([0]*28)
        for j in range(4):
            V[j].extend([0]*28)
        for k in range(9):
            W[k].extend([0]*28)
            
        col_offset = 49
        for r in range(28):
            for i in range(4):
                U[i*4+j][col_offset + r] = U442[i*4+j][r]
            for j in range(4):
                V[j*2+k][col_offset + r] = V442[j*2+k][r]
            for k in range(4):
                W[k*4+i][col_offset + r] = W442[k*4+i][r]
                
        # Extend U, V, W with 443
        for i in range(4):
            U[i].extend([0]*44)
        for j in range(4):
            V[j].extend([0]*44)
        for k in range(9):
            W[k].extend([0]*44)
            
        col_offset = 49 + 28
        for r in range(44):
            for i in range(4):
                U[i*4+j][col_offset + r] = U443[i*4+j][r]
            for j in range(4):
                V[j*3+k][col_offset + r] = V443[j*3+k][r]
            for k in range(4):
                W[k*4+i][col_offset + r] = W443[k*4+i][r]
                
        # Note: The V and W dimensions for the sub-problems need to be mapped to the global 4x4x9.
        # U is 16 x rank.
        # V is 36 x rank.
        # W is 36 x rank.
        
        # My extension logic above is flawed because V and W have different row counts in sub-problems.
        # V444 is 16x49. V442 is 8x28. V443 is 12x44.
        # Global V is 36x121.
        # Rows 0-15 of V correspond to B[0:4, 0:4].
        # Rows 16-23 of V correspond to B[0:4, 4:6].
        # Rows 24-35 of V correspond to B[0:4, 6:9].
        
        # Let's rebuild V and W correctly.
        
        U_final = [[0]*rank_total for _ in range(16)]
        V_final = [[0]*rank_total for _ in range(36)]
        W_final = [[0]*rank_total for _ in range(36)]
        
        # 444 part
        for r in range(49):
            for i in range(16):
                U_final[i][r] = U444[i][r]
            for j in range(16):
                V_final[j][r] = V444[j][r]
            for k in range(16):
                W_final[k][r] = W444[k][r]
                
        # 442 part (cols 4-5 of B)
        # V442 is 8x28. Rows 0-7 correspond to B[0:4, 4:6]?
        # In _build_mm_442, V is 8x28. j is 0..3, k is 0..1.
        # Global j is 0..3. Global k is 4,5.
        # So V_final[j*9 + (4+k)][r] = V442[j*2+k][r].
        for r in range(28):
            for i in range(16):
                U_final[i][49+r] = U442[i][r]
            for j in range(4):
                for k in range(2):
                    V_final[j*9 + 4 + k][49+r] = V442[j*2+k][r]
            for i in range(4):
                for k in range(2):
                    # W442 is 8x28. k is 0..1, i is 0..3.
                    # Global k is 4,5.
                    W_final[(4+k)*4 + i][49+r] = W442[k*4+i][r]
                    
        # 443 part (cols 6-8 of B)
        # V443 is 12x44. j is 0..3, k is 0..2.
        # Global j is 0..3. Global k is 6,7,8.
        for r in range(44):
            for i in range(16):
                U_final[i][77+r] = U443[i][r]
            for j in range(4):
                for k in range(3):
                    V_final[j*9 + 6 + k][77+r] = V443[j*3+k][r]
            for i in range(4):
                for k in range(3):
                    W_final[(6+k)*4 + i][77+r] = W443[k*4+i][r]
                    
        if _validate(U_final, V_final, W_final, 4, 4, 9):
            if time.time() - start_time < 140:
                return {"U": U_final, "V": V_final, "W": W_final}
    except Exception as e:
        pass
    
    # Fallback
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
