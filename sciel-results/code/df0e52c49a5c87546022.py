import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
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

    def build_strassen_2x2():
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]

        # P1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[1][0] = 1; V_s[2][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1; W_s[1][0] = 1; W_s[2][0] = 1; W_s[3][0] = 1

        # P2 = (a21+a22)b11
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1; W_s[3][1] = 1

        # P3 = a11(b12-b22)
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = -1
        W_s[1][2] = 1; W_s[3][2] = 1

        # P4 = a22(b21-b11)
        U_s[3][3] = 1
        V_s[2][3] = 1; V_s[0][3] = -1
        W_s[0][3] = 1; W_s[2][3] = 1

        # P5 = (a11+a12)b22
        U_s[0][4] = 1; U_s[1][4] = 1
        V_s[3][4] = 1
        W_s[0][4] = -1; W_s[1][4] = 1

        # P6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1; U_s[0][5] = -1
        V_s[0][5] = 1; V_s[1][5] = 1
        W_s[3][5] = 1

        # P7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1; U_s[3][6] = -1
        V_s[2][6] = 1; V_s[3][6] = 1
        W_s[0][6] = 1

        return U_s, V_s, W_s

    def build_4x4x7_optimized():
        # Build 4x4x4 using Strassen (rank 49)
        U_s, V_s, W_s = build_strassen_2x2()
        rank_44 = 49
        U_44 = [[0] * rank_44 for _ in range(16)]
        V_44 = [[0] * rank_44 for _ in range(16)]
        W_44 = [[0] * rank_44 for _ in range(16)]

        # For each Strassen term s (0..6) and each 2x2 subterm t (0..6):
        # U_s[ii*2+jj][t] gives the A-block contribution for sub-indices (ii,jj)
        # We need to map: for block (bi,bj) of A in 4x4, the local indices are (ii,jj)
        # so global A index is (bi*2+ii, bj*2+jj) -> row = (bi*2+ii)*4 + (bj*2+jj)
        
        # Strassen formulas for 2x2x2:
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
        # P3 = A00(B01-B11) -> C01 += P3, C11 += P3
        # P4 = A11(B10-B00) -> C00 += P4, C10 += P4
        # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        # P7 = (A01-A11)(B10+B11) -> C00 += P7

        # A coefficients: for each Strassen term s, coefficient for each 2x2 A block position
        # A_block_pos: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
        A_coef = [
            [1, 0, 0, 1],   # P1: A00 + A11
            [0, 0, 1, 1],   # P2: A10 + A11
            [1, 0, 0, 0],   # P3: A00
            [0, 0, 0, 1],   # P4: A11
            [1, 1, 0, 0],   # P5: A00 + A01
            [-1, 0, 1, 0],  # P6: -A00 + A10
            [0, 1, 0, -1],  # P7: A01 - A11
        ]
        B_coef = [
            [1, 0, 0, 1],   # P1: B00 + B11
            [1, 0, 0, 0],   # P2: B00
            [0, 1, 0, -1],  # P3: B01 - B11
            [-1, 0, 1, 0],  # P4: -B00 + B10
            [0, 0, 0, 1],   # P5: B11
            [1, 1, 0, 0],   # P6: B00 + B01
            [0, 0, 1, 1],   # P7: B10 + B11
        ]
        C_coef = [
            [1, 0, 0, 1],   # P1: C00 += 1, C11 += 1
            [0, 0, 1, -1],  # P2: C10 += 1, C11 -= 1
            [0, 1, 0, 1],   # P3: C01 += 1, C11 += 1
            [1, 0, 1, 0],   # P4: C00 += 1, C10 += 1
            [-1, 1, 0, 0],  # P5: C00 -= 1, C01 += 1
            [0, 0, 0, 1],   # P6: C11 += 1
            [1, 0, 0, 0],   # P7: C00 += 1
        ]

        for s in range(7):
            for t in range(7):
                col = s * 7 + t
                # U_s, V_s, W_s are 4x7 matrices (for 2x2 blocks: rows indexed by i*2+j)
                # U_s[row][t] where row = i*2+j for the 2x2 A block
                # We need to expand each 2x2 block position to 4x4
                
                # For A: U_s has entries at positions (ii*2+jj) for ii,jj in {0,1}
                # For block position (bi,bj) in the 4x4 A, local sub-block indices (ii,jj)
                # give global row = (bi*2+ii)*4 + (bj*2+jj)
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_coef[s][bi*2+bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii*2+jj][t]
                                if u_val == 0:
                                    continue
                                gi = bi*2 + ii
                                gj = bj*2 + jj
                                U_44[gi*4+gj][col] += a_mult * u_val
                
                # For B: V_s has entries at positions (jj*2+kk) for jj,kk in {0,1}
                # For block position (bj,bk) in the 4x4 B, local sub-block indices (jj,kk)
                # give global row = (bj*2+jj)*4 + (bk*2+kk)
                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_coef[s][bj*2+bk]
                        if b_mult == 0:
                            continue
                        for jj in range(2):
                            for kk in range(2):
                                v_val = V_s[jj*2+kk][t]
                                if v_val == 0:
                                    continue
                                gj = bj*2 + jj
                                gk = bk*2 + kk
                                V_44[gj*4+gk][col] += b_mult * v_val
                
                # For C: W_s has entries at positions (kk*2+ii) for kk,ii in {0,1}
                # W index: k*n+i = k*4+i for 4x4 case
                # For block position (ci,cj) in the 4x4 C (where ci is row=k, cj is col=i)
                # local sub-block indices (kk,ii) give global: gk=ci*2+kk, gi=cj*2+ii
                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_coef[s][ci*2+cj]
                        if c_mult == 0:
                            continue
                        for kk in range(2):
                            for ii in range(2):
                                w_val = W_s[kk*2+ii][t]
                                if w_val == 0:
                                    continue
                                gk = ci*2 + kk
                                gi = cj*2 + ii
                                W_44[gk*4+gi][col] += c_mult * w_val

        # Part 2: 4x4x3 using Schoolbook (Rank 48)
        rank_43 = 48
        total_rank = rank_44 + rank_43
        
        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]
        W_out = [[0] * total_rank for _ in range(28)]

        # Copy Strassen part
        for r in range(16):
            for c in range(rank_44):
                U_out[r][c] = U_44[r][c]
                V_out[r][c] = V_44[r][c]
                W_out[r][c] = W_44[r][c]

        # Add Schoolbook part for columns 4,5,6 of B
        col_offset = rank_44
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4
                    U_out[i * 4 + j][col_offset] = 1
                    V_out[j * 7 + kk][col_offset] = 1
                    W_out[kk * 4 + i][col_offset] = 1
                    col_offset += 1

        return {"U": U_out, "V": V_out, "W": W_out}

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        # Verify using numpy
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # tensor_decomp[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # For matrix mult: T[i,j,k] should be 1 for all valid i,j,k
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Check: all entries should be 1
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            # Find where it fails
            diff = np.abs(tensor_decomp - target)
            if np.max(diff) > 1e-9:
                return build_schoolbook()

        return result
    except Exception:
        return build_schoolbook()
