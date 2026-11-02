import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def _build_strassen_32():
    """
    Build a rank-32 decomposition for 2x4 * 4x5.
    Splits into two 2x2 * 2x5 multiplications.
    Each 2x2 * 2x5 is decomposed into 2x2*2x2 (rank 7) + 2x2*2x2 (rank 7) + 2x2*2x1 (rank 2).
    Total 16 per half, 32 total.
    """
    n, m, p = 2, 4, 5
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
        
    col = 0
    
    def add_strassen_222(y00, y01, y10, y11, z00, z01, z10, z11, w00, w01, w10, w11):
        nonlocal col
        # M1 = (y00+y11)(z00+z11) -> w00, w11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w00[0], w00[1]), col] += 1
        W[w_idx(w11[0], w11[1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00 -> w10, w11
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w10[0], w10[1]), col] += 1
        W[w_idx(w11[0], w11[1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11) -> w01, w11
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w01[0], w01[1]), col] += 1
        W[w_idx(w11[0], w11[1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00) -> w00, w10
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w00[0], w00[1]), col] += 1
        W[w_idx(w10[0], w10[1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11 -> w00, w01
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w00[0], w00[1]), col] -= 1
        W[w_idx(w01[0], w01[1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01) -> w11
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w11[0], w11[1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11) -> w00
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w00[0], w00[1]), col] += 1
        col += 1

    def add_schoolbook_221(y0, y1, z0, z1, w0, w1):
        nonlocal col
        # C0 = y0 z0 + y1 z1
        U[u_idx(y0[0], y0[1]), col] = 1
        V[v_idx(z0[0], z0[1]), col] = 1
        W[w_idx(w0[0], w0[1]), col] += 1
        col += 1
        U[u_idx(y1[0], y1[1]), col] = 1
        V[v_idx(z1[0], z1[1]), col] = 1
        W[w_idx(w1[0], w1[1]), col] += 1
        col += 1

    # Left half: A_L (cols 0,1) * B_top (rows 0,1)
    # A_L: (0,0),(0,1),(1,0),(1,1)
    # B_top: (0,0),(0,1),(0,2),(0,3),(0,4) and (1,0)..(1,4)
    # Split B_top into B_top1 (cols 0,1), B_top2 (cols 2,3), B_top3 (col 4)
    
    # B_top1: (0,0),(0,1),(1,0),(1,1)
    # Output C cols 0,1
    add_strassen_222(
        (0,0), (0,1), (1,0), (1,1), # y
        (0,0), (0,1), (1,0), (1,1), # z
        (0,0), (0,1), (1,0), (1,1)  # w
    )
    
    # B_top2: (0,2),(0,3),(1,2),(1,3)
    # Output C cols 2,3
    add_strassen_222(
        (0,0), (0,1), (1,0), (1,1), # y
        (0,2), (0,3), (1,2), (1,3), # z
        (2,0), (3,0), (2,1), (3,1)  # w
    )
    
    # B_top3: (0,4),(1,4)
    # Output C col 4
    add_schoolbook_221(
        (0,0), (1,0), # y
        (0,4), (1,4), # z
        (4,0), (4,1)  # w
    )
    
    # Right half: A_R (cols 2,3) * B_bot (rows 2,3)
    # A_R: (0,2),(0,3),(1,2),(1,3)
    # B_bot: (2,0)..(3,4)
    
    # B_bot1: (2,0),(2,1),(3,0),(3,1)
    # Output C cols 0,1
    add_strassen_222(
        (0,2), (0,3), (1,2), (1,3), # y
        (2,0), (2,1), (3,0), (3,1), # z
        (0,0), (0,1), (1,0), (1,1)  # w
    )
    
    # B_bot2: (2,2),(2,3),(3,2),(3,3)
    # Output C cols 2,3
    add_strassen_222(
        (0,2), (0,3), (1,2), (1,3), # y
        (2,2), (2,3), (3,2), (3,3), # z
        (2,0), (3,0), (2,1), (3,1)  # w
    )
    
    # B_bot3: (2,4),(3,4)
    # Output C col 4
    add_schoolbook_221(
        (0,2), (1,2), # y
        (2,4), (3,4), # z
        (4,0), (4,1)  # w
    )
    
    return U, V, W

def _verify_decomposition(U, V, W):
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    U, V, W = _build_strassen_32()
    
    if not _verify_decomposition(U, V, W):
        # Fallback to rank 40
        n, m, p = 2, 4, 5
        rank = 40
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
