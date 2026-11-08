import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x4 matrix multiplication.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks.
    A (2x4) is split into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    B (4x4) is split into four 2x2 blocks:
      B_top_left (rows 0-1, cols 0-1), B_top_right (rows 0-1, cols 2-3)
      B_bot_left (rows 2-3, cols 0-1), B_bot_right (rows 2-3, cols 2-3)
    
    C = A @ B gives:
      C_left  (2x2) = A_left @ B_top_left + A_right @ B_bot_left
      C_right (2x2) = A_left @ B_top_right + A_right @ B_bot_right
    
    Each 2x2 @ 2x2 product uses Strassen (rank 7), so total rank = 4 * 7 = 28.
    
    Output convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)  # (8, 28)
    V = np.zeros((m * p, rank), dtype=float)  # (16, 28)
    W = np.zeros((p * n, rank), dtype=float)  # (8, 28)
    
    # Strassen terms for 2x2 product: P = X @ Y
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # P = [[p00, p01], [p10, p11]]
    #
    # The 7 Strassen products:
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    #
    # P00 = M1 - M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    strassen_u = {  # Coefficients for each element of X in each term
        0: {0: 1, 3: 1},  # M1: x00+x11
        1: {2: 1, 3: 1},  # M2: x10+x11
        2: {0: 1},        # M3: x00
        3: {3: 1},        # M4: x11
        4: {0: 1, 1: 1},  # M5: x00+x01
        5: {2: 1, 0: -1}, # M6: x10-x00
        6: {1: 1, 3: -1}  # M7: x01-x11
    }
    
    strassen_v = {  # Coefficients for each element of Y in each term
        0: {0: 1, 3: 1},  # M1: y00+y11
        1: {0: 1},        # M2: y00
        2: {1: 1, 3: -1}, # M3: y01-y11
        3: {2: 1, 0: -1}, # M4: y10-y00
        4: {3: 1},        # M5: y11
        5: {0: 1, 1: 1},  # M6: y00+y01
        6: {2: 1, 3: 1}   # M7: y10+y11
    }
    
    strassen_w = {  # Coefficients for each element of P in each term
        0: {0: 1, 3: -1}, # M1: P00+M1, P11-M1
        1: {2: 1, 3: -1}, # M2: P10+M2, P11-M2
        2: {1: 1, 3: 1},  # M3: P01+M3, P11+M3
        3: {0: -1, 2: 1}, # M4: P00-M4, P10+M4
        4: {0: -1, 1: 1}, # M5: P00-M5, P01+M5
        5: {3: 1},        # M6: P11+M6
        6: {0: 1}         # M7: P00+M7
    }
    
    # Apply Strassen decomposition for X @ Y where:
    # X occupies rows x_r0..x_r1, cols x_c0..x_c1 of A (2x4)
    # Y occupies rows y_r0..y_r1, cols y_c0..y_c1 of B (4x4)
    # Result P occupies rows p_r0..p_r1, cols p_c0..p_c1 of C (2x4)
    def apply_strassen(x_r0, x_c0, y_r0, y_c0, p_r0, p_c0, base_col):
        """
        X is 2x2 in A starting at row x_r0, col x_c0
        Y is 2x2 in B starting at row y_r0, col y_c0
        P is 2x2 in C starting at row p_r0, col p_c0
        """
        for t in range(7):
            col = base_col + t
            
            # U: X elements. Global index in A: (x_r0 + lr) * m + (x_c0 + lc)
            # where lr, lc are local row/col in the 2x2 block (0 or 1)
            # Local index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            for loc_idx, coeff in strassen_u[t].items():
                lr = loc_idx // 2
                lc = loc_idx % 2
                g_r = x_r0 + lr
                g_c = x_c0 + lc
                u_idx = g_r * m + g_c
                U[u_idx, col] += coeff
            
            # V: Y elements. Global index in B: (y_r0 + lr) * p + (y_c0 + lc)
            for loc_idx, coeff in strassen_v[t].items():
                lr = loc_idx // 2
                lc = loc_idx % 2
                g_r = y_r0 + lr
                g_c = y_c0 + lc
                v_idx = g_r * p + g_c
                V[v_idx, col] += coeff
            
            # W: P elements. Output convention: W[k*n+i] where C[i,k]
            # Global C position: (p_r0 + lr, p_c0 + lc)
            # W index: (p_c0 + lc) * n + (p_r0 + lr)
            for loc_idx, coeff in strassen_w[t].items():
                lr = loc_idx // 2
                lc = loc_idx % 2
                g_r = p_r0 + lr
                g_c = p_c0 + lc
                w_idx = g_c * n + g_r
                W[w_idx, col] += coeff
    
    # Four 2x2 @ 2x2 products:
    # 1. A_left @ B_top_left -> C_left
    #    X: A rows 0-1, cols 0-1 => x_r0=0, x_c0=0
    #    Y: B rows 0-1, cols 0-1 => y_r0=0, y_c0=0
    #    P: C rows 0-1, cols 0-1 => p_r0=0, p_c0=0
    apply_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # 2. A_right @ B_bot_left -> C_left
    #    X: A rows 0-1, cols 2-3 => x_r0=0, x_c0=2
    #    Y: B rows 2-3, cols 0-1 => y_r0=2, y_c0=0
    #    P: C rows 0-1, cols 0-1 => p_r0=0, p_c0=0
    apply_strassen(0, 2, 2, 0, 0, 0, 7)
    
    # 3. A_left @ B_top_right -> C_right
    #    X: A rows 0-1, cols 0-1 => x_r0=0, x_c0=0
    #    Y: B rows 0-1, cols 2-3 => y_r0=0, y_c0=2
    #    P: C rows 0-1, cols 2-3 => p_r0=0, p_c0=2
    apply_strassen(0, 0, 0, 2, 0, 2, 14)
    
    # 4. A_right @ B_bot_right -> C_right
    #    X: A rows 0-1, cols 2-3 => x_r0=0, x_c0=2
    #    Y: B rows 2-3, cols 2-3 => y_r0=2, y_c0=2
    #    P: C rows 0-1, cols 2-3 => p_r0=0, p_c0=2
    apply_strassen(0, 2, 2, 2, 0, 2, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
            
    if not np.allclose(C, Cf, atol=1e-10):
        print("FAILED")
        print("Expected:", C)
        print("Got:", Cf)
        print("Max error:", np.max(np.abs(C - Cf)))
    else:
        print("OK rank", U.shape[1])
