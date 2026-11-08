import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Tensor indexing:
    # A: i*m + j  (i in [0,1], j in [0,2])
    # B: j*p + k  (j in [0,2], k in [0,8])
    # C: k*n + i  (k in [0,8], i in [0,1])
    
    col = 0
    
    # === Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x9) ===
    # This covers C[i,k] contributions from A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # We use Strassen on 4 pairs of B columns: (0,1), (2,3), (4,5), (6,7)
    # and handle column 8 separately.
    
    # Strassen 2x2 decomposition applied to A_L = [[a00, a01], [a10, a11]]
    # and 2-column blocks of B.
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # A indices
        u_a = 0 * m + 0  # A_{0,0}
        u_b = 0 * m + 1  # A_{0,1}
        u_c = 1 * m + 0  # A_{1,0}
        u_d = 1 * m + 1  # A_{1,1}
        
        # B indices
        v_e = 0 * p + k0  # B_{0, k0}
        v_f = 0 * p + k1  # B_{0, k1}
        v_g = 1 * p + k0  # B_{1, k0}
        v_h = 1 * p + k1  # B_{1, k1}
        
        # C output indices (k*n + i)
        w_00 = k0 * n + 0  # C_{0, k0}
        w_01 = k1 * n + 0  # C_{0, k1}
        w_10 = k0 * n + 1  # C_{1, k0}
        w_11 = k1 * n + 1  # C_{1, k1}
        
        # 7 Strassen products
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        # M2 = (c+d)(e)
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        # M3 = (a)(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        # M4 = (d)(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        # M5 = (a+b)(h)
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Reconstruct outputs:
        # C_{0,k0} = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        # C_{0,k1} = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        # C_{1,k0} = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        # C_{1,k1} = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # Handle column k=8 for A[:, 0:2] * B[0:2, 8]
    k8 = 8
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    # 4 direct products
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # === Block 2: A[:, 2] (2x1) * B[2, :] (1x9) ===
    # C[i,k] += A[i,2] * B[2,k]
    u_ar0 = 0 * m + 2  # A_{0,2}
    u_ar1 = 1 * m + 2  # A_{1,2}
    
    for k in range(9):
        v_b = 2 * p + k  # B_{2, k}
        w_0k = k * n + 0
        w_1k = k * n + 1
        
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[w_0k][col] += 1
        
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[w_1k][col+1] += 1
        
        col += 2
    
    return {"U": U, "V": V, "W": W}
