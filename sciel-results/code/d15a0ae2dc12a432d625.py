"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 4x4 x 4x9 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing lists of lists.
    """
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = 4, 4, 9
    
    # Fallback: Schoolbook
    def _schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    try:
        # 1. Strassen 2x2 (Rank 7)
        U_s2 = np.zeros((4, 7), dtype=float)
        V_s2 = np.zeros((4, 7), dtype=float)
        W_s2 = np.zeros((4, 7), dtype=float)
        
        # M1 = (a+d)(e+h)
        U_s2[0,0] = 1; U_s2[3,0] = 1
        V_s2[0,0] = 1; V_s2[3,0] = 1
        W_s2[0,0] = 1; W_s2[3,0] = 1

        # M2 = (c+d)e
        U_s2[2,1] = 1; U_s2[3,1] = 1
        V_s2[0,1] = 1
        W_s2[2,1] = 1; W_s2[3,1] = -1

        # M3 = a(f-h)
        U_s2[0,2] = 1
        V_s2[1,2] = 1; V_s2[3,2] = -1
        W_s2[1,2] = 1; W_s2[3,2] = 1

        # M4 = d(g-e)
        U_s2[3,3] = 1
        V_s2[2,3] = 1; V_s2[0,3] = -1
        W_s2[0,3] = 1; W_s2[2,3] = 1

        # M5 = (a+b)h
        U_s2[0,4] = 1; U_s2[1,4] = 1
        V_s2[3,4] = 1
        W_s2[0,4] = -1; W_s2[1,4] = 1

        # M6 = (c-a)(e+f)
        U_s2[2,5] = 1; U_s2[0,5] = -1
        V_s2[0,5] = 1; V_s2[1,5] = 1
        W_s2[3,5] = 1

        # M7 = (b-d)(g+h)
        U_s2[1,6] = 1; U_s2[3,6] = -1
        V_s2[2,6] = 1; V_s2[3,6] = 1
        W_s2[0,6] = 1; W_s2[2,6] = -1

        # 2. Strassen 4x4 (Rank 49)
        U_s4 = np.zeros((16, 49), dtype=float)
        V_s4 = np.zeros((16, 49), dtype=float)
        W_s4 = np.zeros((16, 49), dtype=float)
        
        for I in range(2):
            for J in range(2):
                for c in range(7):
                    cc = (I * 2 + J) * 7 + c
                    for ii in range(2):
                        for jj in range(2):
                            U_s4[(2 * I + ii) * 4 + (2 * J + jj), cc] += U_s2[ii * 2 + jj, c]
                    for jj in range(2):
                        for kk in range(2):
                            V_s4[(2 * J + jj) * 4 + (2 * I + kk), cc] += V_s2[jj * 2 + kk, c]
                    for kk in range(2):
                        for ii in range(2):
                            W_s4[(2 * I + kk) * 4 + (2 * J + ii), cc] += W_s2[kk * 2 + ii, c]

        # 3. Schoolbook 4x4x1 (Rank 16)
        U_s1 = np.zeros((16, 16), dtype=float)
        V_s1 = np.zeros((4, 16), dtype=float) # 4x1 part of B
        W_s1 = np.zeros((4, 16), dtype=float) # 4x1 part of C
        for i in range(4):
            for j in range(4):
                col_idx = i * 4 + j
                U_s1[i * 4 + j, col_idx] = 1
                V_s1[j, col_idx] = 1
                W_s1[i, col_idx] = 1

        # 4. Combine for 4x4 x 4x9
        rank_total = 49 + 49 + 16
        U_final = np.zeros((16, rank_total), dtype=float)
        V_final = np.zeros((36, rank_total), dtype=float)
        W_final = np.zeros((36, rank_total), dtype=float)
        
        # Block 1: Cols 0-3
        U_final[:, 0:49] = U_s4
        for j in range(4):
            for k in range(4):
                V_final[j * 9 + k, 0:49] = V_s4[j * 4 + k, :]
        for k in range(4):
            for i in range(4):
                W_final[k * 4 + i, 0:49] = W_s4[k * 4 + i, :]
                
        # Block 2: Cols 4-7
        U_final[:, 49:98] = U_s4
        for j in range(4):
            for k in range(4):
                V_final[j * 9 + (4 + k), 49:98] = V_s4[j * 4 + k, :]
        for k in range(4):
            for i in range(4):
                W_final[(4 + k) * 4 + i, 49:98] = W_s4[k * 4 + i, :]
                
        # Block 3: Col 8
        U_final[:, 98:114] = U_s1
        for j in range(4):
            V_final[j * 9 + 8, 98:114] = V_s1[j, :]
        for i in range(4):
            W_final[8 * 4 + i, 98:114] = W_s1[i, :]

        best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        best_rank = rank_total

    except Exception as e:
        print("Construction error:", e)
        import traceback
        traceback.print_exc()

    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Final Rank: {len(result['U'][0])}")
