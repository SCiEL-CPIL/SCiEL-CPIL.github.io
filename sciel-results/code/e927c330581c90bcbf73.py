import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 140

    # Rank 38 decomposition
    # 4 blocks of M_{2,3,2} (rank 9 each) + 1 block of M_{2,3,1} (rank 2)
    # M_{2,3,2} = M_{2,2,2} (Strassen, rank 7) + M_{2,1,2} (rank 2)
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Indices for U: 0,1 (row 0), 2,3 (row 1) ? No.
    # U size 2*3 = 6.
    # i in {0,1}, j in {0,1,2}.
    # Index = i*m + j = i*3 + j.
    # 0,1,2 for i=0.
    # 3,4,5 for i=1.
    
    # V size 3*9 = 27.
    # j in {0,1,2}, k in {0..8}.
    # Index = j*p + k = j*9 + k.
    
    # W size 9*2 = 18.
    # k in {0..8}, i in {0,1}.
    # Index = k*n + i = k*2 + i.

    # Strassen for 2x2 block (j=0,1)
    # U indices:
    # a: (0,0)=0, b: (0,1)=1, c: (1,0)=3, d: (1,1)=4
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        # Block b handles k = 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices for j=0,1
        v_e = 0 * p + k0  # j=0, k=k0
        v_f = 0 * p + k1  # j=0, k=k1
        v_g = 1 * p + k0  # j=1, k=k0
        v_h = 1 * p + k1  # j=1, k=k1
        
        # Strassen 7 mults
        # m1 = (A11 + A22)(B21 + B22) -> U: a,d. V: g,h.
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_g][col] += 1
        V[v_h][col] += 1
        
        # m2 = A22(B11 - B21) -> U: d. V: e,g.
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        V[v_g][col+1] -= 1
        
        # m3 = A11(B22 - B12) -> U: a. V: f,h.
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # m4 = A22(B11 + B12) -> U: d. V: e,f.
        U[u_d][col+3] += 1
        V[v_e][col+3] += 1
        V[v_f][col+3] += 1
        
        # m5 = (A11 + A12)B11 -> U: a,b. V: e.
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_e][col+4] += 1
        
        # m6 = (A21 - A11)(B11 + B21) -> U: c,a. V: e,g.
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_g][col+5] += 1
        
        # m7 = (A12 - A22)(B22 + B12) -> U: b,d. V: f,h.
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_f][col+6] += 1
        V[v_h][col+6] += 1
        
        # W updates for Strassen
        # C11 = m1 + m4 - m2 - m7
        # C12 = m5 + m7
        # C21 = m3 + m6
        # C22 = m1 - m3 + m2 + m5
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1   # m1
        W[w_00][col+3] += 1 # m4
        W[w_00][col+1] -= 1 # m2
        W[w_00][col+6] -= 1 # m7
        
        W[w_01][col+4] += 1 # m5
        W[w_01][col+6] += 1 # m7
        
        W[w_10][col+2] += 1 # m3
        W[w_10][col+5] += 1 # m6
        
        W[w_11][col] += 1   # m1
        W[w_11][col+2] -= 1 # m3
        W[w_11][col+1] += 1 # m2
        W[w_11][col+4] += 1 # m5
        
        col += 7

        # M_{2,1,2} part (j=2)
        # Rank 2
        # r=0: i=0, k=k0, k1?
        # We established:
        # i=0 needs W_{l,r} to cover k0, k1 for l=2k0, 2k0+1?
        # Wait, l = 2k+i.
        # For j=2:
        # T_{0,2, 2k0} = 1, T_{0,2, 2k1} = 1.
        # T_{1,2, 2k0+1} = 1, T_{1,2, 2k1+1} = 1.
        
        # r=0: U[0]=1, V[2]=1. W[2k0]=1, W[2k1]=1.
        U[u_a][col] += 1
        V[2 * p + k0][col] += 1
        V[2 * p + k1][col] += 1
        W[2 * k0][col] += 1
        W[2 * k1][col] += 1
        
        # r=1: U[3]=1, V[2]=1. W[2k0+1]=1, W[2k1+1]=1.
        U[u_c][col+1] += 1
        V[2 * p + k0][col+1] += 1
        V[2 * p + k1][col+1] += 1
        W[2 * k0 + 1][col+1] += 1
        W[2 * k1 + 1][col+1] += 1
        
        col += 2

    # Last block M_{2,3,1} (k=8)
    # Rank 2
    # i=0: T_{0,j,0} = 1 for j=0,1,2.
    # i=1: T_{1,j,1} = 1 for j=0,1,2.
    
    # r=0: i=0. U[0]=1. V[0]=1, V[1]=1, V[2]=1. W[0]=1.
    U[u_a][col] += 1
    V[0 * p + 8][col] += 1
    V[1 * p + 8][col] += 1
    V[2 * p + 8][col] += 1
    W[8 * n + 0][col] += 1
    
    # r=1: i=1. U[3]=1. V[0]=1, V[1]=1, V[2]=1. W[1]=1.
    U[u_c][col+1] += 1
    V[0 * p + 8][col+1] += 1
    V[1 * p + 8][col+1] += 1
    V[2 * p + 8][col+1] += 1
    W[8 * n + 1][col+1] += 1
    
    col += 2
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0

    Td = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
    if np.linalg.norm(Td - T) > 1e-9:
        # Fallback to rank 50 if verification fails
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        u_a, u_b, u_c, u_d = 0, 1, 3, 4
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = k0
            v_f = k1
            v_g = p + k0
            v_h = p + k1
            
            U[u_a][col] += 1
            U[u_d][col] += 1
            V[v_e][col] += 1
            V[v_h][col] += 1

            U[u_c][col+1] += 1
            U[u_d][col+1] += 1
            V[v_e][col+1] += 1

            U[u_a][col+2] += 1
            V[v_f][col+2] += 1
            V[v_h][col+2] -= 1

            U[u_d][col+3] += 1
            V[v_g][col+3] += 1
            V[v_e][col+3] -= 1

            U[u_a][col+4] += 1
            U[u_b][col+4] += 1
            V[v_h][col+4] += 1

            U[u_c][col+5] += 1
            U[u_a][col+5] -= 1
            V[v_e][col+5] += 1
            V[v_f][col+5] += 1

            U[u_b][col+6] += 1
            U[u_d][col+6] -= 1
            V[v_g][col+6] += 1
            V[v_h][col+6] += 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            W[w_00][col] += 1
            W[w_00][col+3] += 1
            W[w_00][col+4] -= 1
            W[w_00][col+6] += 1

            W[w_01][col+2] += 1
            W[w_01][col+4] += 1

            W[w_10][col+1] += 1
            W[w_10][col+3] += 1

            W[w_11][col] += 1
            W[w_11][col+1] -= 1
            W[w_11][col+2] += 1
            W[w_11][col+5] += 1

            col += 7

        k8 = 8
        v_e = k8
        v_g = p + k8

        U[u_a][col] += 1
        V[v_e][col] += 1
        U[u_b][col+1] += 1
        V[v_g][col+1] += 1
        U[u_c][col+2] += 1
        V[v_e][col+2] += 1
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1

        w_08 = k8 * n + 0
        w_18 = k8 * n + 1

        W[w_08][col] += 1
        W[w_08][col+1] += 1
        W[w_18][col+2] += 1
        W[w_18][col+3] += 1

        col += 4

        u_ar0, u_ar1 = 2, 5
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1
            V[v_b][col] += 1
            W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1
            V[v_b][col+1] += 1
            W[k * n + 1][col+1] += 1
            col += 2

    return {"U": U, "V": V, "W": W}
