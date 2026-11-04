import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:, t], V[:, t])
            T_recon += np.einsum('ab,c->abc', uv, W[:, t])
        return np.max(np.abs(T_recon - T))

    # --- Sub-block Decompositions ---
    
    def strassen_2x2():
        # Rank 7 decomposition for 2x2 * 2x2
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # m1 = a00 * (b11 - b10)
        t = 0
        U[0, t] = 1
        V[3, t] = 1
        V[2, t] = -1
        W[0, t] = 1
        W[1, t] = 1
        
        # m2 = (a10 + a11) * b00
        t = 1
        U[2, t] = 1
        U[3, t] = 1
        V[0, t] = 1
        W[2, t] = 1
        W[3, t] = 1
        
        # m3 = a00 * (b01 - b00)
        t = 2
        U[0, t] = 1
        V[1, t] = 1
        V[0, t] = -1
        W[0, t] = 1
        W[2, t] = 1
        
        # m4 = a11 * (b00 + b11)
        t = 3
        U[3, t] = 1
        V[0, t] = 1
        V[3, t] = 1
        W[0, t] = 1
        W[1, t] = 1
        W[2, t] = 1
        W[3, t] = 1
        
        # m5 = (a00 - a10) * (b10 + b11)
        t = 4
        U[0, t] = 1
        U[2, t] = -1
        V[2, t] = 1
        V[3, t] = 1
        W[1, t] = 1
        W[3, t] = 1
        
        # m6 = (a01 + a10) * (b00 - b10)
        t = 5
        U[1, t] = 1
        U[2, t] = 1
        V[0, t] = 1
        V[2, t] = -1
        W[2, t] = 1
        W[0, t] = 1
        
        # m7 = (a11 - a01) * (b01 + b11)
        t = 6
        U[3, t] = 1
        U[1, t] = -1
        V[1, t] = 1
        V[3, t] = 1
        W[3, t] = 1
        W[1, t] = 1
        
        return U, V, W

    def bidi_3x3():
        # Rank 23 decomposition for 3x3 * 3x3 (Bini-De Falco-De Lorenzi)
        # This is complex. I will use a simpler, verified rank-23 scheme or fall back to schoolbook if unsure.
        # For safety and speed, I will use Schoolbook for 3x3 in this iteration to ensure correctness,
        # but I will use the optimized 3x2x3 block.
        # Rank 27.
        r = 27
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W

    def block_3x2x3():
        # Decomposition for (3x2) * (2x3) -> 3x3
        # Rank 17: 1 Strassen-2 (7) + 3 schoolbook parts (4+4+2=10)
        # A is 3x2, B is 2x3.
        # A = [[a00, a01], [a10, a11], [a20, a21]]
        # B = [[b00, b01, b02], [b10, b11, b12]]
        # C = A B
        #
        # Partition:
        # A_top = A[0:2, :] (2x2)
        # A_bot = A[2, :] (1x2)
        # B_left = B[:, 0:2] (2x2)
        # B_mid = B[:, 2] (2x1)
        #
        # C_top_left = A_top B_left (2x2) -> Strassen (Rank 7)
        # C_top_right = A_top B_mid (2x1) -> Schoolbook (Rank 4)
        # C_bot_left = A_bot B_left (1x2) -> Schoolbook (Rank 4)
        # C_bot_right = A_bot B_mid (1x1) -> Schoolbook (Rank 2)
        
        r = 17
        U = np.zeros((6, r)) # 3*2
        V = np.zeros((6, r)) # 2*3
        W = np.zeros((9, r)) # 3*3
        
        col = 0
        # 1. Strassen for A_top * B_left
        U_s, V_s, W_s = strassen_2x2()
        # Map Strassen indices to local indices
        # U_s indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Local A: 0,1,2,3
        # V_s indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Local B: 0,1,2,3 (cols 0,1)
        # W_s indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Local C: 0,1,3,4 (rows 0,1; cols 0,1)
        
        for t in range(7):
            for u_idx in range(4):
                if U_s[u_idx, t] != 0:
                    U[u_idx, col] += U_s[u_idx, t]
            for v_idx in range(4):
                if V_s[v_idx, t] != 0:
                    V[v_idx, col] += V_s[v_idx, t]
            for w_idx in range(4):
                if W_s[w_idx, t] != 0:
                    # w_idx 0->C[0,0], 1->C[0,1], 2->C[1,0], 3->C[1,1]
                    # Local C indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
                    # Strassen W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    if w_idx == 0: W[0, col] += W_s[w_idx, t]
                    elif w_idx == 1: W[1, col] += W_s[w_idx, t]
                    elif w_idx == 2: W[3, col] += W_s[w_idx, t]
                    elif w_idx == 3: W[4, col] += W_s[w_idx, t]
            col += 1
            
        # 2. A_top * B_mid (2x2 * 2x1 -> 2x1)
        # C[0,2] = a00 b02 + a01 b12
        # C[1,2] = a10 b02 + a11 b12
        # Rank 4
        for i in range(2):
            for j in range(2):
                U[i*2+j, col] = 1 # A[i,j]
                V[j*3+2, col] = 1 # B[j,2]
                W[i*3+2, col] = 1 # C[i,2]
                col += 1
                
        # 3. A_bot * B_left (1x2 * 2x2 -> 1x2)
        # C[2,0] = a20 b00 + a21 b10
        # C[2,1] = a20 b01 + a21 b11
        # Rank 4
        for j in range(2):
            for k in range(2):
                U[4+j, col] = 1 # A[2,j]
                V[j*3+k, col] = 1 # B[j,k]
                W[6+k, col] = 1 # C[2,k]
                col += 1
                
        # 4. A_bot * B_mid (1x2 * 2x1 -> 1x1)
        # C[2,2] = a20 b02 + a21 b12
        # Rank 2
        for j in range(2):
            U[4+j, col] = 1 # A[2,j]
            V[j*3+2, col] = 1 # B[j,2]
            W[8, col] = 1 # C[2,2]
            col += 1
            
        return U, V, W

    def schoolbook_block(n_a, m_a, n_b, m_b):
        # Generic schoolbook for (n_a x m_a) * (m_a x m_b)
        r = n_a * m_a * m_b
        U = np.zeros((n_a * m_a, r))
        V = np.zeros((m_a * m_b, r))
        W = np.zeros((n_a * m_b, r))
        col = 0
        for i in range(n_a):
            for j in range(m_a):
                for k in range(m_b):
                    U[i*m_a+j, col] = 1
                    V[j*m_b+k, col] = 1
                    W[k*n_a+i, col] = 1
                    col += 1
        return U, V, W

    # --- Main Construction ---
    
    # Partition indices
    # I1={0,1,2}, I2={3}
    # J1={0,1,2}, J2={3,4}
    # K1={0,1,2}, K2={3,4,5}
    
    r_total = 27 + 27 + 17 + 17 + 9 + 9 + 6 + 6
    U_full = np.zeros((nm, r_total))
    V_full = np.zeros((mp, r_total))
    W_full = np.zeros((pn, r_total))
    
    col = 0
    
    # 1. A[I1, J1] * B[J1, K1] -> C[I1, K1] (3x3 * 3x3)
    U_b, V_b, W_b = bidi_3x3()
    for t in range(27):
        for u_idx in range(9):
            if U_b[u_idx, t] != 0:
                i = u_idx // 3
                j = u_idx % 3
                U_full[i*5 + j, col] += U_b[u_idx, t]
        for v_idx in range(9):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[j*6 + k, col] += V_b[v_idx, t]
        for w_idx in range(9):
            if W_b[w_idx, t] != 0:
                k = w_idx // 3
                i = w_idx % 3
                W_full[k*4 + i, col] += W_b[w_idx, t]
        col += 1
        
    # 2. A[I1, J1] * B[J1, K2] -> C[I1, K2] (3x3 * 3x3)
    U_b, V_b, W_b = bidi_3x3()
    for t in range(27):
        for u_idx in range(9):
            if U_b[u_idx, t] != 0:
                i = u_idx // 3
                j = u_idx % 3
                U_full[i*5 + j, col] += U_b[u_idx, t]
        for v_idx in range(9):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[j*6 + (3+k), col] += V_b[v_idx, t]
        for w_idx in range(9):
            if W_b[w_idx, t] != 0:
                k = w_idx // 3
                i = w_idx % 3
                W_full[(3+k)*4 + i, col] += W_b[w_idx, t]
        col += 1
        
    # 3. A[I1, J2] * B[J2, K1] -> C[I1, K1] (3x2 * 2x3)
    U_b, V_b, W_b = block_3x2x3()
    for t in range(17):
        for u_idx in range(6):
            if U_b[u_idx, t] != 0:
                i = u_idx // 2
                j = u_idx % 2
                U_full[i*5 + (3+j), col] += U_b[u_idx, t]
        for v_idx in range(6):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[(3+j)*6 + k, col] += V_b[v_idx, t]
        for w_idx in range(9):
            if W_b[w_idx, t] != 0:
                k = w_idx // 3
                i = w_idx % 3
                W_full[k*4 + i, col] += W_b[w_idx, t]
        col += 1
        
    # 4. A[I1, J2] * B[J2, K2] -> C[I1, K2] (3x2 * 2x3)
    U_b, V_b, W_b = block_3x2x3()
    for t in range(17):
        for u_idx in range(6):
            if U_b[u_idx, t] != 0:
                i = u_idx // 2
                j = u_idx % 2
                U_full[i*5 + (3+j), col] += U_b[u_idx, t]
        for v_idx in range(6):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[(3+j)*6 + (3+k), col] += V_b[v_idx, t]
        for w_idx in range(9):
            if W_b[w_idx, t] != 0:
                k = w_idx // 3
                i = w_idx % 3
                W_full[(3+k)*4 + i, col] += W_b[w_idx, t]
        col += 1
        
    # 5. A[I2, J1] * B[J1, K1] -> C[I2, K1] (1x3 * 3x3)
    U_b, V_b, W_b = schoolbook_block(1, 3, 3, 3)
    for t in range(9):
        for u_idx in range(3):
            if U_b[u_idx, t] != 0:
                j = u_idx
                U_full[3*5 + j, col] += U_b[u_idx, t]
        for v_idx in range(9):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[j*6 + k, col] += V_b[v_idx, t]
        for w_idx in range(3):
            if W_b[w_idx, t] != 0:
                k = w_idx
                W_full[k*4 + 3, col] += W_b[w_idx, t]
        col += 1
        
    # 6. A[I2, J1] * B[J1, K2] -> C[I2, K2] (1x3 * 3x3)
    U_b, V_b, W_b = schoolbook_block(1, 3, 3, 3)
    for t in range(9):
        for u_idx in range(3):
            if U_b[u_idx, t] != 0:
                j = u_idx
                U_full[3*5 + j, col] += U_b[u_idx, t]
        for v_idx in range(9):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[j*6 + (3+k), col] += V_b[v_idx, t]
        for w_idx in range(3):
            if W_b[w_idx, t] != 0:
                k = w_idx
                W_full[(3+k)*4 + 3, col] += W_b[w_idx, t]
        col += 1
        
    # 7. A[I2, J2] * B[J2, K1] -> C[I2, K1] (1x2 * 2x3)
    U_b, V_b, W_b = schoolbook_block(1, 2, 2, 3)
    for t in range(6):
        for u_idx in range(2):
            if U_b[u_idx, t] != 0:
                j = u_idx
                U_full[3*5 + (3+j), col] += U_b[u_idx, t]
        for v_idx in range(6):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[(3+j)*6 + k, col] += V_b[v_idx, t]
        for w_idx in range(3):
            if W_b[w_idx, t] != 0:
                k = w_idx
                W_full[k*4 + 3, col] += W_b[w_idx, t]
        col += 1
        
    # 8. A[I2, J2] * B[J2, K2] -> C[I2, K2] (1x2 * 2x3)
    U_b, V_b, W_b = schoolbook_block(1, 2, 2, 3)
    for t in range(6):
        for u_idx in range(2):
            if U_b[u_idx, t] != 0:
                j = u_idx
                U_full[3*5 + (3+j), col] += U_b[u_idx, t]
        for v_idx in range(6):
            if V_b[v_idx, t] != 0:
                j = v_idx // 3
                k = v_idx % 3
                V_full[(3+j)*6 + (3+k), col] += V_b[v_idx, t]
        for w_idx in range(3):
            if W_b[w_idx, t] != 0:
                k = w_idx
                W_full[(3+k)*4 + 3, col] += W_b[w_idx, t]
        col += 1
        
    # Verify
    err = verify_decomposition(U_full, V_full, W_full)
    if err < 1e-10:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
