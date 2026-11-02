import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_from_block(U_block, V_block, W_block):
        R_block = U_block.shape[1]
        R_total = 3 * R_block
        U_full = np.zeros((4, R_total))
        V_full = np.zeros((18, R_total))
        W_full = np.zeros((18, R_total))
        
        for b in range(3):
            start_r = b * R_block
            end_r = start_r + R_block
            U_full[:, start_r:end_r] = U_block
            for k in range(2):
                for j_local in range(3):
                    j_global = 3*b + j_local
                    v_idx_block = k*3 + j_local
                    v_idx_full = k*9 + j_global
                    V_full[v_idx_full, start_r:end_r] = V_block[v_idx_block, :]
            for j_local in range(3):
                j_global = 3*b + j_local
                for i in range(2):
                    w_idx_block = j_local*2 + i
                    w_idx_full = j_global*2 + i
                    W_full[w_idx_full, start_r:end_r] = W_block[w_idx_block, :]
        return U_full, V_full, W_full

    # Direct construction of rank-9 decomposition for [2,2,3]
    # Using Strassen for first 2 columns (rank 7) + schoolbook for 3rd column (rank 2)
    def build_rank9_223():
        # Strassen for 2x2x2: A is 2x2, B1 is 2x2 (first 2 cols of B)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For [2,2,3]: A is 2x2, B is 2x3
        # Let B = [B1 | B2] where B1 is 2x2 (cols 0,1) and B2 is 2x1 (col 2)
        # C = A*B = [A*B1 | A*B2]
        # A*B1 via Strassen (rank 7)
        # A*B2 via schoolbook: 2 multiplications
        # Total rank: 9
        
        # U indices: 0=A00, 1=A10, 2=A01, 3=A11
        # V indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
        # W indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
        
        R = 9
        U = np.zeros((4, R))
        V = np.zeros((6, R))
        W = np.zeros((6, R))
        
        # Strassen terms for B1 (cols 0,1):
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[4, 0] = 1  # C00 += M1, C11 += M1
        
        # M2 = (A10+A11)B00
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[3, 1] = 1; W[4, 1] = -1  # C10 += M2, C11 -= M2
        
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[3, 2] = 1; V[4, 2] = -1
        W[1, 2] = 1; W[4, 2] = 1  # C01 += M3, C11 += M3
        
        # M4 = A11(B01-B10)
        U[3, 3] = 1
        V[1, 3] = 1; V[3, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1  # C00 += M4, C10 += M4
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[2, 4] = 1
        V[4, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
        
        # M6 = (A10-A00)(B00+B10)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[3, 5] = 1
        W[4, 5] = 1  # C11 += M6
        
        # M7 = (A01-A11)(B01+B11)
        U[2, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1  # C00 += M7
        
        # Schoolbook for B2 (col 2):
        # C02 = A00*B02 + A01*B12
        # C12 = A10*B02 + A11*B12
        U[0, 7] = 1
        V[2, 7] = 1
        W[2, 7] = 1  # C02 += A00*B02
        
        U[2, 7] = 1
        V[5, 7] = 1
        W[2, 7] += 1  # C02 += A01*B12
        
        U[1, 8] = 1
        V[2, 8] = 1
        W[5, 8] = 1  # C12 += A10*B02
        
        U[3, 8] = 1
        V[5, 8] = 1
        W[5, 8] += 1  # C12 += A11*B12
        
        return U, V, W

    # Build rank-9 block decomposition
    U_b, V_b, W_b = build_rank9_223()
    
    if verify_decomp(U_b, V_b, W_b, 2, 2, 3):
        U_full, V_full, W_full = build_from_block(U_b, V_b, W_b)
        if verify_decomp(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: Rank 32 construction
    terms = []
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
        terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
        terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
        terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
        terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
    
    k = 8
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
