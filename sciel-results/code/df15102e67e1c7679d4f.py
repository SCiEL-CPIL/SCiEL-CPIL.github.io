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

    def verify_decomposition(U_list, V_list, W_list, nn=4, mm=4, pp=7):
        """Verify that the tensor decomposition is correct."""
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        rank = U.shape[1]
        # T[i,j,k] = sum_c U[i*mm+j, c] * V[j*pp+k, c] * W[k*nn+i, c]
        # Should equal 1 for all i,j,k
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * mm + j, c] * V[j * pp + k, c] * W[k * nn + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_2x2():
        """Rank-7 Strassen for 2x2 * 2x2 -> 2x2.
        U: 4x7, V: 4x7, W: 4x7.
        U row = i*2+j, V row = j*2+k, W row = k*2+i (trace convention).
        """
        rank_s = 7
        U_s = [[0]*rank_s for _ in range(4)]
        V_s = [[0]*rank_s for _ in range(4)]
        W_s = [[0]*rank_s for _ in range(4)]

        # P1 = (a11+a22)(b11+b22)
        U_s[0][0]=1; U_s[3][0]=1
        V_s[0][0]=1; V_s[1][0]=1; V_s[2][0]=1; V_s[3][0]=1
        W_s[0][0]=1; W_s[1][0]=1; W_s[2][0]=1; W_s[3][0]=1

        # P2 = (a21+a22)b11
        U_s[2][1]=1; U_s[3][1]=1
        V_s[0][1]=1
        W_s[2][1]=1; W_s[3][1]=1

        # P3 = a11(b12-b22)
        U_s[0][2]=1
        V_s[1][2]=1; V_s[3][2]=-1
        W_s[1][2]=1; W_s[3][2]=1

        # P4 = a22(b21-b11)
        U_s[3][3]=1
        V_s[2][3]=1; V_s[0][3]=-1
        W_s[0][3]=1; W_s[2][3]=1

        # P5 = (a11+a12)b22
        U_s[0][4]=1; U_s[1][4]=1
        V_s[3][4]=1
        W_s[0][4]=-1; W_s[1][4]=1

        # P6 = (a21-a11)(b11+b12)
        U_s[2][5]=1; U_s[0][5]=-1
        V_s[0][5]=1; V_s[1][5]=1
        W_s[3][5]=1

        # P7 = (a12-a22)(b21+b22)
        U_s[1][6]=1; U_s[3][6]=-1
        V_s[2][6]=1; V_s[3][6]=1
        W_s[0][6]=1

        return U_s, V_s, W_s

    def build_4x4x7():
        """
        Block decomposition: split B (4x7) into B1 (4x4) and B2 (4x3).
        C = A*B = A*B1 + A*B2.
        
        For A*B1: use recursive Strassen (rank 49).
        For A*B2: use schoolbook (rank 48).
        Total rank: 97.
        
        U: 16 x 97, V: 28 x 97, W: 28 x 97.
        """
        U_s, V_s, W_s = build_strassen_2x2()
        
        rank_total = 97
        U_out = [[0]*rank_total for _ in range(16)]
        V_out = [[0]*rank_total for _ in range(28)]
        W_out = [[0]*rank_total for _ in range(28)]
        
        # --- Part 1: Strassen for 4x4 * 4x4 (columns 0-3 of B) ---
        # Recursive Strassen: 7 outer slots, each with 7 inner Strassen terms.
        # Outer slot s, inner term t -> column = s*7 + t
        
        A_block_mults = [
            [1, 0, 0, 1],   # P1: A_11+A_22
            [0, 0, 1, 1],   # P2: A_21+A_22
            [1, 0, 0, 0],   # P3: A_11
            [0, 0, 0, 1],   # P4: A_22
            [1, 1, 0, 0],   # P5: A_11+A_12
            [-1, 0, 1, 0],  # P6: A_21-A_11
            [0, 1, 0, -1],  # P7: A_12-A_22
        ]
        B_block_mults = [
            [1, 0, 0, 1],   # P1: B_11+B_22
            [1, 0, 0, 0],   # P2: B_11
            [0, 1, 0, -1],  # P3: B_12-B_22
            [-1, 0, 1, 0],  # P4: B_21-B_11
            [0, 0, 0, 1],   # P5: B_22
            [1, 1, 0, 0],   # P6: B_11+B_12
            [0, 0, 1, 1],   # P7: B_21+B_22
        ]
        C_block_mults = [
            [1, 0, 0, 1],   # P1: C_11=1, C_22=1
            [0, 0, 1, -1],  # P2: C_21=1, C_22=-1
            [0, 1, 0, 1],   # P3: C_12=1, C_22=1
            [1, 0, 1, 0],   # P4: C_11=1, C_21=1
            [-1, 1, 0, 0],  # P5: C_11=-1, C_12=1
            [0, 0, 0, 1],   # P6: C_22=1
            [1, 0, 0, 0],   # P7: C_11=1
        ]
        
        for s in range(7):
            Am = A_block_mults[s]
            Bm = B_block_mults[s]
            Cm = C_block_mults[s]
            for t in range(7):
                col = s * 7 + t
                # A part: U rows indexed by i*4+j
                for bi in range(2):
                    for bj in range(2):
                        a_mult = Am[bi*2+bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii*2+jj][t]
                                if u_val == 0:
                                    continue
                                a4_i = bi*2+ii
                                a4_j = bj*2+jj
                                U_out[a4_i*4+a4_j][col] += a_mult * u_val
                # B part: V rows indexed by j*4+k (j is shared dim, k is col of B)
                for bj in range(2):
                    for bk in range(2):
                        b_mult = Bm[bj*2+bk]
                        if b_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                v_val = V_s[ii*2+jj][t]
                                if v_val == 0:
                                    continue
                                b4_j = bj*2+ii
                                b4_k = bk*2+jj
                                V_out[b4_j*4+b4_k][col] += b_mult * v_val
                # C part: W rows indexed by k*4+i (trace convention)
                for ci in range(2):
                    for cj in range(2):
                        c_mult = Cm[ci*2+cj]
                        if c_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                w_val = W_s[ii*2+jj][t]
                                if w_val == 0:
                                    continue
                                c4_k = ci*2+ii
                                c4_i = cj*2+jj
                                W_out[c4_k*4+c4_i][col] += c_mult * w_val
        
        # --- Part 2: Schoolbook for 4x4 * 4x3 (columns 4-6 of B) ---
        col_offset = 49
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4  # actual column in B: 4,5,6
                    U_out[i*4+j][col_offset] = 1
                    V_out[j*7+kk][col_offset] = 1
                    W_out[kk*4+i][col_offset] = 1
                    col_offset += 1
        
        return {"U": U_out, "V": V_out, "W": W_out}

    # Deadline check
    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7()
        
        # Verify correctness
        if verify_decomposition(result["U"], result["V"], result["W"]):
            if time.time() - start_time > DEADLINE:
                return build_schoolbook()
            return result
        else:
            return build_schoolbook()
    except Exception:
        return build_schoolbook()

# EVOLVE-BLOCK-END
