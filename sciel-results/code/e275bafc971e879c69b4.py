"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses a block decomposition with Strassen for 2x2*2x2 submatrices.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    start = time.time()
    TIMEOUT = 120.0 - 5.0  # 115s deadline
    
    # We'll build the decomposition as a list of (U_vec, V_vec, W_vec) where each is a dict mapping index to coeff.
    # Then convert to arrays.
    
    def build_strassen(a0, b0):
        """Return list of 7 terms for 2x2*2x2 at A[a0:a0+2, a0:b0+2] and B[b0:b0+2, b0:p] but for 2x2 output.
        Actually, we'll do it for fixed A block and B block."""
        # A rows: a0, a0+1; A cols: b0, b0+1
        # B rows: b0, b0+1; B cols: c0, c0+1
        # C rows: a0, a0+1; C cols: c0, c0+1
        terms = []
        # m1
        terms.append(([((a0, b0), 1), ((a0+1, b0+1), 1)],
                      [((b0, c0), 1), ((b0+1, c0+1), 1)],
                      [((c0, a0), 1), ((c0+1, a0+1), 1)]))
        # m2
        terms.append(([((a0+1, b0), 1), ((a0+1, b0+1), 1)],
                      [((b0, c0), 1)],
                      [((c0, a0+1), 1), ((c0+1, a0+1), -1)]))
        # m3
        terms.append(([((a0, b0), 1)],
                      [((b0+1, c0), 1), ((b0+1, c0+1), -1)],
                      [((c0+1, a0), 1), ((c0+1, a0+1), 1)]))
        # m4
        terms.append(([((a0+1, b0+1), 1)],
                      [((b0+1, c0), 1), ((b0, c0), -1)],
                      [((c0, a0), 1), ((c0, a0+1), 1)]))
        # m5
        terms.append(([((a0, b0), 1), ((a0, b0+1), 1)],
                      [((b0+1, c0+1), 1)],
                      [((c0, a0), -1), ((c0+1, a0), 1)]))
        # m6
        terms.append(([((a0+1, b0), 1), ((a0, b0), -1)],
                      [((b0, c0), 1), ((b0+1, c0+1), 1)],
                      [((c0+1, a0+1), 1)]))
        # m7
        terms.append(([((a0, b0+1), 1), ((a0+1, b0+1), -1)],
                      [((b0+1, c0), 1), ((b0+1, c0+1), 1)],
                      [((c0, a0), 1)]))
        return terms
    
    U_list = []  # each element is a list of ( (i,j), coeff )
    V_list = []
    W_list = []
    
    # Strassen for A[0:2, 0:2] * B[0:2, c:c+2] for c in 0,2,4,6
    for c in range(0, 8, 2):
        for term in build_strassen(0, c):
            U_list.append(term[0])
            V_list.append(term[1])
            W_list.append(term[2])
    
    # Direct for A[0:2, 2] * B[2, :]
    for i in range(2):
        for k in range(8):
            U_list.append([((i, 2), 1)])
            V_list.append([((2, k), 1)])
            W_list.append([((k, i), 1)])
    
    # Direct for A[2, 0:2] * B[0:2, :]
    for j in range(2):
        for k in range(8):
            U_list.append([((2, j), 1)])
            V_list.append([((j, k), 1)])
            W_list.append([((k, 2), 1)])
    
    # Direct for A[2,2] * B[2,:]
    for k in range(8):
        U_list.append([((2, 2), 1)])
        V_list.append([((2, k), 1)])
        W_list.append([((k, 2), 1)])
    
    rank = len(U_list)
    
    # Build U, V, W as 2D lists
    U = [[0.0] * rank for _ in range(n*m)]
    V = [[0.0] * rank for _ in range(m*p)]
    W = [[0.0] * rank for _ in range(p*n)]
    
    for r in range(rank):
        for (i, j), val in U_list[r]:
            U[i*m + j][r] = val
        for (j, k), val in V_list[r]:
            V[j*p + k][r] = val
        for (k, i), val in W_list[r]:
            W[k*n + i][r] = val
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-8:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback to schoolbook
        rank_fb = n*m*p
        U_fb = [[0.0]*rank_fb for _ in range(n*m)]
        V_fb = [[0.0]*rank_fb for _ in range(m*p)]
        W_fb = [[0.0]*rank_fb for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][idx] = 1.0
                    V_fb[j*p+k][idx] = 1.0
                    W_fb[k*n+i][idx] = 1.0
                    idx += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Round to nice numbers
    def niceify(mat):
        out = []
        for row in mat:
            nice_row = []
            for x in row:
                if abs(x * 2 - round(x*2)) < 1e-6:
                    y = round(x*2)/2.0
                    if abs(y - round(y)) < 1e-9:
                        nice_row.append(int(round(y)))
                    else:
                        nice_row.append(y)
                else:
                    nice_row.append(float(x))
            out.append(nice_row)
        return out
    
    return {"U": niceify(U_arr), "V": niceify(V_arr), "W": niceify(W_arr)}
# EVOLVE-BLOCK-END
