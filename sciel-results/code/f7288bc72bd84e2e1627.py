import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Fallback: Strassen (7) x Schoolbook (27) = 189
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    U3_sb = np.zeros((9, 27), dtype=int)
    V3_sb = np.zeros((9, 27), dtype=int)
    W3_sb = np.zeros((9, 27), dtype=int)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U3_sb[i * 3 + j, r] = 1
                V3_sb[j * 3 + k, r] = 1
                W3_sb[k * 3 + i, r] = 1

    def build_kronecker(U_small, V_small, W_small, U_large, V_large, W_large):
        rank_small = U_small.shape[1]
        rank_large = U_large.shape[1]
        rank = rank_small * rank_large
        dim_small = int(np.sqrt(U_small.shape[0]))
        dim_large = int(np.sqrt(U_large.shape[0]))
        dim_total = dim_small * dim_large
        
        U_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        V_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        W_total = np.zeros((dim_total * dim_total, rank), dtype=int)
        
        nz_Us = [(i, j, r, val) for i in range(dim_small) for j in range(dim_small) for r in range(rank_small) for val in [U_small[i*dim_small+j, r]] if val != 0]
        nz_Vs = [(j, k, r, val) for j in range(dim_small) for k in range(dim_small) for r in range(rank_small) for val in [V_small[j*dim_small+k, r]] if val != 0]
        nz_Ws = [(k, i, r, val) for k in range(dim_small) for i in range(dim_small) for r in range(rank_small) for val in [W_small[k*dim_small+i, r]] if val != 0]
        
        nz_Ul = [(i, j, r, val) for i in range(dim_large) for j in range(dim_large) for r in range(rank_large) for val in [U_large[i*dim_large+j, r]] if val != 0]
        nz_Vl = [(j, k, r, val) for j in range(dim_large) for k in range(dim_large) for r in range(rank_large) for val in [V_large[j*dim_large+k, r]] if val != 0]
        nz_Wl = [(k, i, r, val) for k in range(dim_large) for i in range(dim_large) for r in range(rank_large) for val in [W_large[k*dim_large+i, r]] if val != 0]
        
        for i_s, j_s, r_s, v_s in nz_Us:
            for i_l, j_l, r_l, v_l in nz_Ul:
                r = r_s * rank_large + r_l
                i_t = i_s * dim_large + i_l
                j_t = j_s * dim_large + j_l
                U_total[i_t * dim_total + j_t, r] += v_s * v_l
                
        for j_s, k_s, r_s, v_s in nz_Vs:
            for j_l, k_l, r_l, v_l in nz_Vl:
                r = r_s * rank_large + r_l
                j_t = j_s * dim_large + j_l
                k_t = k_s * dim_large + k_l
                V_total[j_t * dim_total + k_t, r] += v_s * v_l
                
        for k_s, i_s, r_s, v_s in nz_Ws:
            for k_l, i_l, r_l, v_l in nz_Wl:
                r = r_s * rank_large + r_l
                k_t = k_s * dim_large + k_l
                i_t = i_s * dim_large + i_l
                W_total[k_t * dim_total + i_t, r] += v_s * v_l
                
        return U_total, V_total, W_total

    def verify(U, V, W, rank, num_samples=100):
        np.random.seed(42)
        for _ in range(num_samples):
            i = np.random.randint(0, 6)
            j = np.random.randint(0, 6)
            k = np.random.randint(0, 6)
            s = 0.0
            for r in range(rank):
                s += U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r]
            if abs(s - 1.0) > 1e-6:
                return False
        return True

    # Primary Strategy: Strassen (7) x Schoolbook (27) = 189
    U6, V6, W6 = build_kronecker(U2, V2, W2, U3_sb, V3_sb, W3_sb)
    
    if np.allclose(U6, np.round(U6)):
        U6 = U6.astype(int)
    if np.allclose(V6, np.round(V6)):
        V6 = V6.astype(int)
    if np.allclose(W6, np.round(W6)):
        W6 = W6.astype(int)
        
    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
# EVOLVE-BLOCK-END
