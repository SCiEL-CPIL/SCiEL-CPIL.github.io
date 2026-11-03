import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Partitions
    # I: 3 -> {0,1}, {2}
    # K: 5 -> {0,1}, {2,3,4}
    # J: 7 -> {0,1}, {2,3,4,5,6}
    
    I1 = [0, 1]
    I2 = [2]
    K1 = [0, 1]
    K2 = [2, 3, 4]
    J1 = [0, 1]
    J2 = [2, 3, 4, 5, 6]
    
    blocks = []
    
    # Block 1: I1, K1, J1 -> 2x2x2 Strassen
    blocks.append(('strassen', I1, K1, J1))
    
    # Block 2: I1, K2, J1 -> 2x3x2 Schoolbook
    blocks.append(('schoolbook', I1, K2, J1))
    
    # Block 3: I1, K1, J2 -> 2x2x5
    # Decompose J2 (size 5) into 2, 2, 1
    J2_sub = [[2, 3], [4, 5], [6]]
    # Sub-blocks:
    # 2x2x2 (Strassen)
    blocks.append(('strassen', I1, K1, J2_sub[0]))
    # 2x2x2 (Strassen)
    blocks.append(('strassen', I1, K1, J2_sub[1]))
    # 2x2x1 (Schoolbook)
    blocks.append(('schoolbook', I1, K1, J2_sub[2]))
    
    # Block 4: I2, K1, J1 -> 1x2x2 Schoolbook
    blocks.append(('schoolbook', I2, K1, J1))
    
    # Block 5: I2, K2, J1 -> 1x3x2 Schoolbook
    blocks.append(('schoolbook', I2, K2, J1))
    
    # Block 6: I1, K2, J2 -> 2x3x5
    # Decompose J2 (size 5) into 2, 2, 1
    # 2x3x2 (Schoolbook)
    blocks.append(('schoolbook', I1, K2, J2_sub[0]))
    # 2x3x2 (Schoolbook)
    blocks.append(('schoolbook', I1, K2, J2_sub[1]))
    # 2x3x1 (Schoolbook)
    blocks.append(('schoolbook', I1, K2, J2_sub[2]))
    
    # Block 7: I2, K1, J2 -> 1x2x5
    # 1x2x2 (Schoolbook)
    blocks.append(('schoolbook', I2, K1, J2_sub[0]))
    # 1x2x2 (Schoolbook)
    blocks.append(('schoolbook', I2, K1, J2_sub[1]))
    # 1x2x1 (Schoolbook)
    blocks.append(('schoolbook', I2, K1, J2_sub[2]))
    
    # Block 8: I2, K2, J2 -> 1x3x5
    # 1x3x2 (Schoolbook)
    blocks.append(('schoolbook', I2, K2, J2_sub[0]))
    # 1x3x2 (Schoolbook)
    blocks.append(('schoolbook', I2, K2, J2_sub[1]))
    # 1x3x1 (Schoolbook)
    blocks.append(('schoolbook', I2, K2, J2_sub[2]))
    
    # Calculate total rank
    total_rank = 0
    for b in blocks:
        if b[0] == 'strassen':
            total_rank += 7
        else:
            ns = len(b[1])
            ms = len(b[2])
            ps = len(b[3])
            total_rank += ns * ms * ps
            
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for method, I_idx, K_idx, J_idx in blocks:
        ns = len(I_idx)
        ms = len(K_idx)
        ps = len(J_idx)
        
        if method == 'strassen':
            # Must be 2x2x2
            rank_s = 7
            # Local U, V, W for Strassen
            # Indices are local 0,1
            # Map local to global
            # U local (2*2, 7)
            # V local (2*2, 7)
            # W local (2*2, 7)
            
            # Strassen coefficients
            # M1: (a+d)(x+v)
            # M2: (c+d)x
            # M3: a(y-v)
            # M4: d(u-x)
            # M5: (a+b)v
            # M6: (c-a)(x+y)
            # M7: (b-d)(u+v)
            
            # U entries (A)
            # a: (0,0), b: (0,1), c: (1,0), d: (1,1)
            # M1: a+d -> U[0,0]=1, U[1,1]=1
            # M2: c+d -> U[1,0]=1, U[1,1]=1
            # M3: a -> U[0,0]=1
            # M4: d -> U[1,1]=1
            # M5: a+b -> U[0,0]=1, U[0,1]=1
            # M6: c-a -> U[1,0]=1, U[0,0]=-1
            # M7: b-d -> U[0,1]=1, U[1,1]=-1
            
            U_local = [
                [1, 0, 1, 0, 1, -1, 0], # a
                [0, 1, 0, 0, 1, 0, 1],  # b
                [0, 0, 1, 0, 0, 1, 0],  # c
                [1, 1, 0, 1, 0, 0, -1]  # d
            ]
            
            # V entries (B)
            # x: (0,0), y: (0,1), u: (1,0), v: (1,1)
            # M1: x+v -> V[0,0]=1, V[1,1]=1
            # M2: x -> V[0,0]=1
            # M3: y-v -> V[0,1]=1, V[1,1]=-1
            # M4: u-x -> V[1,0]=1, V[0,0]=-1
            # M5: v -> V[1,1]=1
            # M6: x+y -> V[0,0]=1, V[0,1]=1
            # M7: u+v -> V[1,0]=1, V[1,1]=1
            
            V_local = [
                [1, 1, 0, -1, 0, 1, 0], # x
                [0, 0, 1, 0, 0, 1, 0],  # y
                [0, 0, 0, 1, 0, 0, 1],  # u
                [1, 0, -1, 0, 1, 0, 1]  # v
            ]
            
            # W entries (C)
            # C11: M1+M4-M5+M7
            # C12: M3+M5
            # C21: M2+M4
            # C22: M1-M2+M3+M6
            
            W_local = [
                [1, 0, 0, 1, -1, 0, 1], # C11
                [0, 0, 1, 0, 1, 0, 0],  # C12
                [0, 1, 0, 1, 0, 0, 0],  # C21
                [1, -1, 1, 0, 0, 1, 0]  # C22
            ]
            
            # Map to global
            for li in range(ns):
                for lj in range(ms):
                    g_i = I_idx[li]
                    g_j = K_idx[lj]
                    for c in range(rank_s):
                        U[g_i * m + g_j][col_offset + c] = U_local[li * ms + lj][c]
            
            for lj in range(ms):
                for lk in range(ps):
                    g_j = K_idx[lj]
                    g_k = J_idx[lk]
                    for c in range(rank_s):
                        V[g_j * p + g_k][col_offset + c] = V_local[lj * ps + lk][c]
            
            for lk in range(ps):
                for li in range(ns):
                    g_k = J_idx[lk]
                    g_i = I_idx[li]
                    for c in range(rank_s):
                        W[g_k * n + g_i][col_offset + c] = W_local[lk * ns + li][c]
                        
            col_offset += rank_s
            
        else:
            # Schoolbook
            rank_s = ns * ms * ps
            for li in range(ns):
                for lj in range(ms):
                    for lk in range(ps):
                        g_i = I_idx[li]
                        g_j = K_idx[lj]
                        g_k = J_idx[lk]
                        
                        U[g_i * m + g_j][col_offset] = 1
                        V[g_j * p + g_k][col_offset] = 1
                        W[g_k * n + g_i][col_offset] = 1
                        col_offset += 1
                        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
