"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Verify schoolbook is correct
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for c in range(U.shape[1]):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-6:
                        return False
        return True

    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    def add_strassen_block(I, J, K):
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # M1 = (A00+A11)(B00+B11)
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1.0; u1[i1*m + j1] = 1.0
        v1[j0*p + k0] = 1.0; v1[j1*p + k1] = 1.0
        w1[k0*n + i0] = 1.0; w1[k1*n + i1] = 1.0
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[i1*m + j0] = 1.0; u2[i1*m + j1] = 1.0
        v2[j0*p + k0] = 1.0
        w2[k0*n + i1] = 1.0; w2[k1*n + i1] = -1.0
        add_term(u2, v2, w2)
        
        # M3 = A00(B01-B11)
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[i0*m + j0] = 1.0
        v3[j0*p + k1] = 1.0; v3[j1*p + k1] = -1.0
        w3[k1*n + i0] = 1.0; w3[k1*n + i1] = 1.0
        add_term(u3, v3, w3)
        
        # M4 = A11(B10-B00)
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[i1*m + j1] = 1.0
        v4[j1*p + k0] = 1.0; v4[j0*p + k0] = -1.0
        w4[k0*n + i0] = 1.0; w4[k0*n + i1] = 1.0
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)B11
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[i0*m + j0] = 1.0; u5[i0*m + j1] = 1.0
        v5[j1*p + k1] = 1.0
        w5[k0*n + i0] = -1.0; w5[k1*n + i0] = 1.0
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B01)
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[i1*m + j0] = 1.0; u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0; v6[j0*p + k1] = 1.0
        w6[k1*n + i1] = 1.0
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)(B10+B11)
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[i0*m + j1] = 1.0; u7[i1*m + j1] = -1.0
        v7[j1*p + k0] = 1.0; v7[j1*p + k1] = 1.0
        w7[k0*n + i0] = 1.0
        add_term(u7, v7, w7)

    def add_laderman_block(I, J, K):
        # Laderman's algorithm for 3x3x3
        # Indices i, j, k in {0, 1, 2}
        i0, i1, i2 = sorted(I)
        j0, j1, j2 = sorted(J)
        k0, k1, k2 = sorted(K)
        
        # Laderman's 23 terms
        # M1 = (A00+A22)(B00+B22)
        u1 = np.zeros(n*m); v1 = np.zeros(m*p); w1 = np.zeros(p*n)
        u1[i0*m + j0] = 1; u1[i2*m + j2] = 1
        v1[j0*p + k0] = 1; v1[j2*p + k2] = 1
        w1[k0*n + i0] = 1; w1[k2*n + i2] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11+A12)(B00+B10+B20)
        u2 = np.zeros(n*m); v2 = np.zeros(m*p); w2 = np.zeros(p*n)
        u2[i1*m + j0] = 1; u2[i1*m + j1] = 1; u2[i1*m + j2] = 1
        v2[j0*p + k0] = 1; v2[j1*p + k0] = 1; v2[j2*p + k0] = 1
        w2[k0*n + i1] = 1
        add_term(u2, v2, w2)
        
        # M3 = (A01+A10+A22)(B01+B11+B21)
        u3 = np.zeros(n*m); v3 = np.zeros(m*p); w3 = np.zeros(p*n)
        u3[i0*m + j1] = 1; u3[i1*m + j0] = 1; u3[i2*m + j2] = 1
        v3[j0*p + k1] = 1; v3[j1*p + k1] = 1; v3[j2*p + k1] = 1
        w3[k1*n + i0] = 1
        add_term(u3, v3, w3)
        
        # M4 = (A02+A11+A20)(B02+B12+B22)
        u4 = np.zeros(n*m); v4 = np.zeros(m*p); w4 = np.zeros(p*n)
        u4[i0*m + j2] = 1; u4[i1*m + j1] = 1; u4[i2*m + j0] = 1
        v4[j0*p + k2] = 1; v4[j1*p + k2] = 1; v4[j2*p + k2] = 1
        w4[k2*n + i0] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A12+A21)(B10+B20)
        u5 = np.zeros(n*m); v5 = np.zeros(m*p); w5 = np.zeros(p*n)
        u5[i1*m + j2] = 1; u5[i2*m + j1] = 1
        v5[j1*p + k0] = 1; v5[j2*p + k0] = 1
        w5[k0*n + i1] = -1; w5[k0*n + i2] = -1
        add_term(u5, v5, w5)
        
        # M6 = (A20+A01)(B20+B00)
        u6 = np.zeros(n*m); v6 = np.zeros(m*p); w6 = np.zeros(p*n)
        u6[i2*m + j0] = 1; u6[i0*m + j1] = 1
        v6[j2*p + k0] = 1; v6[j0*p + k0] = 1
        w6[k0*n + i2] = 1; w6[k0*n + i0] = -1
        add_term(u6, v6, w6)
        
        # M7 = (A02+A11)(B02+B12)
        u7 = np.zeros(n*m); v7 = np.zeros(m*p); w7 = np.zeros(p*n)
        u7[i0*m + j2] = 1; u7[i1*m + j1] = 1
        v7[j0*p + k2] = 1; v7[j1*p + k2] = 1
        w7[k2*n + i0] = -1; w7[k2*n + i1] = -1
        add_term(u7, v7, w7)
        
        # M8 = (A10+A22)(B10+B22)
        u8 = np.zeros(n*m); v8 = np.zeros(m*p); w8 = np.zeros(p*n)
        u8[i1*m + j0] = 1; u8[i2*m + j2] = 1
        v8[j1*p + k0] = 1; v8[j2*p + k2] = 1
        w8[k0*n + i1] = 1; w8[k2*n + i2] = -1
        add_term(u8, v8, w8)
        
        # M9 = (A21+A02)(B21+B02)
        u9 = np.zeros(n*m); v9 = np.zeros(m*p); w9 = np.zeros(p*n)
        u9[i2*m + j1] = 1; u9[i0*m + j2] = 1
        v9[j2*p + k1] = 1; v9[j0*p + k2] = 1
        w9[k1*n + i2] = 1; w9[k2*n + i0] = -1
        add_term(u9, v9, w9)
        
        # M10 = (A01+A12)(B01+B12)
        u10 = np.zeros(n*m); v10 = np.zeros(m*p); w10 = np.zeros(p*n)
        u10[i0*m + j1] = 1; u10[i1*m + j2] = 1
        v10[j0*p + k1] = 1; v10[j1*p + k2] = 1
        w10[k1*n + i0] = 1; w10[k2*n + i1] = -1
        add_term(u10, v10, w10)
        
        # M11 = A00*B11
        u11 = np.zeros(n*m); v11 = np.zeros(m*p); w11 = np.zeros(p*n)
        u11[i0*m + j0] = 1
        v11[j1*p + k1] = 1
        w11[k1*n + i0] = 1
        add_term(u11, v11, w11)
        
        # M12 = A11*B00
        u12 = np.zeros(n*m); v12 = np.zeros(m*p); w12 = np.zeros(p*n)
        u12[i1*m + j1] = 1
        v12[j0*p + k0] = 1
        w12[k0*n + i1] = 1
        add_term(u12, v12, w12)
        
        # M13 = A22*B11
        u13 = np.zeros(n*m); v13 = np.zeros(m*p); w13 = np.zeros(p*n)
        u13[i2*m + j2] = 1
        v13[j1*p + k1] = 1
        w13[k1*n + i2] = 1
        add_term(u13, v13, w13)
        
        # M14 = A11*B22
        u14 = np.zeros(n*m); v14 = np.zeros(m*p); w14 = np.zeros(p*n)
        u14[i1*m + j1] = 1
        v14[j2*p + k2] = 1
        w14[k2*n + i1] = 1
        add_term(u14, v14, w14)
        
        # M15 = A22*B00
        u15 = np.zeros(n*m); v15 = np.zeros(m*p); w15 = np.zeros(p*n)
        u15[i2*m + j2] = 1
        v15[j0*p + k0] = 1
        w15[k0*n + i2] = 1
        add_term(u15, v15, w15)
        
        # M16 = A00*B22
        u16 = np.zeros(n*m); v16 = np.zeros(m*p); w16 = np.zeros(p*n)
        u16[i0*m + j0] = 1
        v16[j2*p + k2] = 1
        w16[k2*n + i0] = 1
        add_term(u16, v16, w16)
        
        # M17 = A10*B21
        u17 = np.zeros(n*m); v17 = np.zeros(m*p); w17 = np.zeros(p*n)
        u17[i1*m + j0] = 1
        v17[j2*p + k1] = 1
        w17[k1*n + i1] = 1
        add_term(u17, v17, w17)
        
        # M18 = A21*B10
        u18 = np.zeros(n*m); v18 = np.zeros(m*p); w18 = np.zeros(p*n)
        u18[i2*m + j1] = 1
        v18[j1*p + k0] = 1
        w18[k0*n + i2] = 1
        add_term(u18, v18, w18)
        
        # M19 = A02*B11
        u19 = np.zeros(n*m); v19 = np.zeros(m*p); w19 = np.zeros(p*n)
        u19[i0*m + j2] = 1
        v19[j1*p + k1] = 1
        w19[k1*n + i0] = 1
        add_term(u19, v19, w19)
        
        # M20 = A11*B02
        u20 = np.zeros(n*m); v20 = np.zeros(m*p); w20 = np.zeros(p*n)
        u20[i1*m + j1] = 1
        v20[j0*p + k2] = 1
        w20[k2*n + i1] = 1
        add_term(u20, v20, w20)
        
        # M21 = A20*B11
        u21 = np.zeros(n*m); v21 = np.zeros(m*p); w21 = np.zeros(p*n)
        u21[i2*m + j0] = 1
        v21[j1*p + k1] = 1
        w21[k1*n + i2] = 1
        add_term(u21, v21, w21)
        
        # M22 = A11*B20
        u22 = np.zeros(n*m); v22 = np.zeros(m*p); w22 = np.zeros(p*n)
        u22[i1*m + j1] = 1
        v22[j2*p + k0] = 1
        w22[k0*n + i1] = 1
        add_term(u22, v22, w22)
        
        # M23 = A01*B22
        u23 = np.zeros(n*m); v23 = np.zeros(m*p); w23 = np.zeros(p*n)
        u23[i0*m + j1] = 1
        v23[j2*p + k2] = 1
        w23[k2*n + i0] = 1
        add_term(u23, v23, w23)

    def add_schoolbook_block(I, J, K):
        for i in I:
            for j in J:
                for k in K:
                    u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    add_term(u, v, w)

    # Apply Laderman to I={0,1,2}, J={0,1,2}, K={0,1,2}
    add_laderman_block({0, 1, 2}, {0, 1, 2}, {0, 1, 2})
    
    # Apply Strassen/Schoolbook to remaining blocks
    # Group 1: I={0,1,2}, J={3,4}, K={0,1,2}
    add_strassen_block({0, 1}, {3, 4}, {0, 1})
    add_schoolbook_block({0, 1}, {3, 4}, {2})
    add_schoolbook_block({2}, {3, 4}, {0, 1, 2})
    
    # Group 2: I={0,1,2}, J={0,1,2}, K={3,4,5,6}
    # K={3,4}
    add_strassen_block({0, 1}, {0, 1}, {3, 4})
    add_schoolbook_block({0, 1}, {2}, {3, 4})
    add_schoolbook_block({2}, {0, 1, 2}, {3, 4})
    # K={5,6}
    add_strassen_block({0, 1}, {0, 1}, {5, 6})
    add_schoolbook_block({0, 1}, {2}, {5, 6})
    add_schoolbook_block({2}, {0, 1, 2}, {5, 6})
    
    # Group 3: I={0,1,2}, J={3,4}, K={3,4,5,6}
    add_strassen_block({0, 1}, {3, 4}, {3, 4})
    add_strassen_block({0, 1}, {3, 4}, {5, 6})
    add_schoolbook_block({2}, {3, 4}, {3, 4, 5, 6})
    
    U_mat = np.array(U).T
    V_mat = np.array(V).T
    W_mat = np.array(W).T
    
    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
