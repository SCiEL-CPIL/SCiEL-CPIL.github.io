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
        # Part 1: 4x4x4 using Strassen (Rank 49)
        U_s, V_s, W_s = build_strassen_2x2()
        rank_44 = 49
        U_44 = [[0] * rank_44 for _ in range(16)]
        V_44 = [[0] * rank_44 for _ in range(16)]
        W_44 = [[0] * rank_44 for _ in range(16)]

        # Strassen 2x2x2 coefficients
        # U_s[i*2+j][t], V_s[j*2+k][t], W_s[k*2+i][t]
        # For each Strassen term t (0..6), the 2x2 block multipliers

        # A_block_mults[s][bi*2+bj] = multiplier for A block (bi,bj) in Strassen term s
        # B_block_mults[s][bj*2+bk] = multiplier for B block (bj,bk) in Strassen term s  
        # C_block_mults[s][ci*2+cj] = multiplier for C block (ci,cj) in Strassen term s

        A_block_mults = [
            [1, 0, 0, 1],   # P1: (A11+A22)
            [0, 0, 1, 1],   # P2: (A21+A22)
            [1, 0, 0, 0],   # P3: A11
            [0, 0, 0, 1],   # P4: A22
            [1, 1, 0, 0],   # P5: (A11+A12)
            [-1, 0, 1, 0],  # P6: (A21-A11)
            [0, 1, 0, -1],  # P7: (A12-A22)
        ]
        B_block_mults = [
            [1, 0, 0, 1],   # P1: (B11+B22)
            [1, 0, 0, 0],   # P2: B11
            [0, 1, 0, -1],  # P3: (B12-B22)
            [-1, 0, 1, 0],  # P4: (B21-B11)
            [0, 0, 0, 1],   # P5: B22
            [1, 1, 0, 0],   # P6: (B11+B12)
            [0, 0, 1, 1],   # P7: (B21+B22)
        ]
        C_block_mults = [
            [1, 0, 0, 1],   # P1: C11+=, C22+=
            [0, 0, 1, -1],  # P2: C21+=, C22-=
            [0, 1, 0, 1],   # P3: C12+=, C22+=
            [1, 0, 1, 0],   # P4: C11+=, C21+=
            [-1, 1, 0, 0],  # P5: C11-=, C12+=
            [0, 0, 0, 1],   # P6: C22+=
            [1, 0, 0, 0],   # P7: C11+=
        ]

        for s in range(7):
            A_mults = A_block_mults[s]
            B_mults = B_block_mults[s]
            C_mults = C_block_mults[s]
            
            for t in range(7):
                col = s * 7 + t
                
                # A part: U_44[i*4+j][col]
                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_mults[bi * 2 + bj]
                        if a_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii * 2 + jj][t]
                                if u_val == 0: continue
                                a4_i = bi * 2 + ii
                                a4_j = bj * 2 + jj
                                U_44[a4_i * 4 + a4_j][col] += a_mult * u_val

                # B part: V_44[j*4+k][col]
                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_mults[bj * 2 + bk]
                        if b_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                v_val = V_s[ii * 2 + jj][t]
                                if v_val == 0: continue
                                b4_j = bj * 2 + ii
                                b4_k = bk * 2 + jj
                                V_44[b4_j * 4 + b4_k][col] += b_mult * v_val

                # C part: W_44[k*4+i][col]
                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_mults[ci * 2 + cj]
                        if c_mult == 0: continue
                        for ii in range(2):
                            for jj in range(2):
                                w_val = W_s[ii * 2 + jj][t]
                                if w_val == 0: continue
                                c4_k = ci * 2 + ii
                                c4_i = cj * 2 + jj
                                W_44[c4_k * 4 + c4_i][col] += c_mult * w_val

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
        # Use einsum
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)

        if not np.allclose(tensor_decomp, 1.0, atol=1e-9):
            return build_schoolbook()

        return result
    except Exception:
        return build_schoolbook()
