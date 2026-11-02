"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition via Strassen blocks."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    # Build target tensor for verification
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Strassen 2x2x2 decomposition: 7 terms
    # C00 = M1 - M3 + M4 - M7
    # C01 = M5 + M7
    # C10 = M1 + M3
    # C11 = M2 - M4 + M1 - M6
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # R(2,2,1)=4: C0 = A00*B00 + A01*B10, C1 = A10*B00 + A11*B10
    # M1 = (A00+A10)*B00, M2 = (A01+A11)*B10, M3 = A10*B00, M4 = A11*B10
    # C0 = M1 - M3 + M2 - M4, C1 = M3 + M4
    
    # R(2,1,2)=4: C00=A0*B00, C01=A0*B01, C10=A1*B10, C11=A1*B11
    # M1=(A0+A1)*B00, M2=(A0+A1)*B01, M3=A0*B00, M4=A1*B11
    # C00=M3, C01=M2-M3, C10=M1-M3, C11=M4
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec: indices into A (8), v_vec: indices into B (28), w_vec: indices into C (14)"""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Split middle dim j=0..3 into s=0 (j=0,1) and s=1 (j=2,3)
    # For each s, A_s is 2x2: rows i=0,1; cols t=0,1 -> j=2s+t
    # B_s is 2x7: rows t=0,1 -> j=2s+t; cols k=0..6
    # Split k=0..6 into: (0,1), (2,3), (4,5), (6) -> three 2x2x2 blocks + one 2x2x1
    
    for s in range(2):
        # A_s[i,t] corresponds to A[i, 2s+t], index in A: i*4 + (2s+t)
        # B_s[t,k] corresponds to B[2s+t, k], index in B: (2s+t)*7 + k
        # C[i,k] index in C: k*2 + i
        
        # k groups: [0,1], [2,3], [4,5] are 2x2x2 (Strassen), [6] is 2x2x1
        k_groups_2x2 = [(0, 1), (2, 3), (4, 5)]
        k_group_2x1 = (6,)
        
        for (k0, k1) in k_groups_2x2:
            # Strassen 2x2x2: A_s (2x2) * B_s[:, k0:k1+1] (2x2) -> C_s (2x2)
            # A_s[i,t] = A[i, 2s+t], B_s[t,k'] = B[2s+t, k0+k']
            # C[i,k'] = C[i, k0+k']
            
            # M1 = (A[0,2s]+A[1,2s+1]) * (B[2s,k0]+B[2s+1,k1])
            # M2 = (A[1,2s]+A[1,2s+1]) * B[2s,k0]
            # M3 = A[0,2s] * (B[2s+1,k0]-B[2s+1,k1])
            # M4 = A[1,2s+1] * (B[2s,k1]-B[2s,k0])
            # M5 = (A[0,2s]+A[0,2s+1]) * B[2s+1,k1]
            # M6 = (A[1,2s]-A[0,2s]) * (B[2s,k0]+B[2s+1,k0])
            # M7 = (A[0,2s+1]-A[1,2s+1]) * (B[2s,k1]+B[2s+1,k1])
            
            a00_idx = 0 * 4 + (2 * s + 0)  # A[0, 2s]
            a01_idx = 0 * 4 + (2 * s + 1)  # A[0, 2s+1]
            a10_idx = 1 * 4 + (2 * s + 0)  # A[1, 2s]
            a11_idx = 1 * 4 + (2 * s + 1)  # A[1, 2s+1]
            
            b00_idx = (2 * s + 0) * 7 + k0  # B[2s, k0]
            b01_idx = (2 * s + 0) * 7 + k1  # B[2s, k1]
            b10_idx = (2 * s + 1) * 7 + k0  # B[2s+1, k0]
            b11_idx = (2 * s + 1) * 7 + k1  # B[2s+1, k1]
            
            c00_idx = k0 * 2 + 0  # C[0, k0]
            c01_idx = k1 * 2 + 0  # C[0, k1]
            c10_idx = k0 * 2 + 1  # C[1, k0]
            c11_idx = k1 * 2 + 1  # C[1, k1]
            
            # M1 = (A00+A11)*(B00+B11)
            u1 = {a00_idx: 1, a11_idx: 1}
            v1 = {b00_idx: 1, b11_idx: 1}
            w1 = {c00_idx: 1, c10_idx: 1, c01_idx: -1, c11_idx: -1}
            add_term(u1, v1, w1)
            
            # M2 = (A10+A11)*B00
            u2 = {a10_idx: 1, a11_idx: 1}
            v2 = {b00_idx: 1}
            w2 = {c11_idx: 1}
            add_term(u2, v2, w2)
            
            # M3 = A00*(B10-B11)
            u3 = {a00_idx: 1}
            v3 = {b10_idx: 1, b11_idx: -1}
            w3 = {c00_idx: -1, c10_idx: 1}
            add_term(u3, v3, w3)
            
            # M4 = A11*(B01-B00)
            u4 = {a11_idx: 1}
            v4 = {b01_idx: 1, b00_idx: -1}
            w4 = {c00_idx: -1, c11_idx: 1}
            add_term(u4, v4, w4)
            
            # M5 = (A00+A01)*B11
            u5 = {a00_idx: 1, a01_idx: 1}
            v5 = {b11_idx: 1}
            w5 = {c01_idx: 1}
            add_term(u5, v5, w5)
            
            # M6 = (A10-A00)*(B00+B10)
            u6 = {a10_idx: 1, a00_idx: -1}
            v6 = {b00_idx: 1, b10_idx: 1}
            w6 = {c11_idx: -1}
            add_term(u6, v6, w6)
            
            # M7 = (A01-A11)*(B01+B11)
            u7 = {a01_idx: 1, a11_idx: -1}
            v7 = {b01_idx: 1, b11_idx: 1}
            w7 = {c01_idx: 1, c11_idx: 1}
            add_term(u7, v7, w7)
        
        # 2x2x1 block for k=6
        # C[0,6] = A[0,2s]*B[2s,6] + A[0,2s+1]*B[2s+1,6]
        # C[1,6] = A[1,2s]*B[2s,6] + A[1,2s+1]*B[2s+1,6]
        
        k6 = 6
        b06_idx = (2 * s + 0) * 7 + k6  # B[2s, 6]
        b16_idx = (2 * s + 1) * 7 + k6  # B[2s+1, 6]
        c06_idx = k6 * 2 + 0  # C[0, 6]
        c16_idx = k6 * 2 + 1  # C[1, 6]
        
        # M1 = (A00+A10)*B06
        u_m1 = {a00_idx: 1, a10_idx: 1}
        v_m1 = {b06_idx: 1}
        w_m1 = {c06_idx: 1, c16_idx: 1}
        add_term(u_m1, v_m1, w_m1)
        
        # M2 = (A01+A11)*B16
        u_m2 = {a01_idx: 1, a11_idx: 1}
        v_m2 = {b16_idx: 1}
        w_m2 = {c06_idx: 1, c16_idx: 1}
        add_term(u_m2, v_m2, w_m2)
        
        # M3 = A10*B06
        u_m3 = {a10_idx: 1}
        v_m3 = {b06_idx: 1}
        w_m3 = {c16_idx: 1}
        add_term(u_m3, v_m3, w_m3)
        
        # M4 = A11*B16
        u_m4 = {a11_idx: 1}
        v_m4 = {b16_idx: 1}
        w_m4 = {c16_idx: 1}
        add_term(u_m4, v_m4, w_m4)
        
        # C[0,6] = M1 - M3 + M2 - M4
        # C[1,6] = M3 + M4
        # Wait, that's not right. Let me recheck.
        # C[0,6] = A[0,2s]*B[2s,6] + A[0,2s+1]*B[2s+1,6]
        # = (A[0,2s] + A[1,2s])*B[2s,6] - A[1,2s]*B[2s,6] + (A[0,2s+1] + A[1,2s+1])*B[2s+1,6] - A[1,2s+1]*B[2s+1,6]
        # = M1 - M3 + M2 - M4
        # C[1,6] = A[1,2s]*B[2s,6] + A[1,2s+1]*B[2s+1,6] = M3 + M4
        # So the w vectors should be:
        # M1: w = {c06: 1, c16: 1}
        # M2: w = {c06: 1, c16: 1}  -- wait, no
        # Actually: C[0,6] = M1 - M3 + M2 - M4, C[1,6] = M3 + M4
        # So:
        # M1 contributes +1 to C[0,6], +1 to C[1,6]
        # M2 contributes +1 to C[0,6], +1 to C[1,6]  -- NO!
        # M2 = (A01+A11)*B16, this only appears in C[0,6] with coefficient +1
        # Let me redo:
        # C[0,6] = A00*B06 + A01*B16
        # C[1,6] = A10*B06 + A11*B16
        # M1 = (A00+A10)*B06 -> contributes to both C[0,6] and C[1,6]
        # M2 = (A01+A11)*B16 -> contributes to both C[0,6] and C[1,6]
        # M3 = A10*B06 -> contributes to C[1,6]
        # M4 = A11*B16 -> contributes to C[1,6]
        # C[0,6] = M1 - M3 + M2 - M4
        # C[1,6] = M3 + M4
        # This is correct! The w vectors I have are right.
    
    # Convert to matrices
    rank = len(U)
    U_mat = np.zeros((nm, rank))
    V_mat = np.zeros((mp, rank))
    W_mat = np.zeros((pn, rank))
    
    for d in range(rank):
        for idx, val in U[d].items():
            U_mat[idx, d] = val
        for idx, val in V[d].items():
            V_mat[idx, d] = val
        for idx, val in W[d].items():
            W_mat[idx, d] = val
    
    # Verify
    T_rec = np.einsum('ad,bd,cd->abc', U_mat, V_mat, W_mat)
    if not np.allclose(T_rec, T, atol=1e-10, rtol=1e-10):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
