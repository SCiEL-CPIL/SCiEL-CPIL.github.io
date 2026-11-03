import numpy as np
import time

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    r = U.shape[1]
    if r == 0:
        return False
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    if not np.allclose(T_reconstructed, 1.0, atol=1e-9, rtol=1e-9):
        return False
    
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    if not np.allclose(U2, np.round(U2), atol=1e-9): return False
    if not np.allclose(V2, np.round(V2), atol=1e-9): return False
    if not np.allclose(W2, np.round(W2), atol=1e-9): return False
    
    return True

def _make_term(u_dict, v_dict, w_dict):
    u_vec = np.zeros(n * m)
    v_vec = np.zeros(m * p)
    w_vec = np.zeros(p * n)
    for idx, val in u_dict.items():
        u_vec[idx] = val
    for idx, val in v_dict.items():
        v_vec[idx] = val
    for idx, val in w_dict.items():
        w_vec[idx] = val
    return u_vec, v_vec, w_vec

def _strassen_2x2_block(i_base, j_base, k_base):
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    U1 = {u_idx(i_base, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V1 = {v_idx(j_base, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W1 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U1); V_terms.append(V1); W_terms.append(W1)
    
    U2 = {u_idx(i_base+1, j_base): 1, u_idx(i_base+1, j_base+1): 1}
    V2 = {v_idx(j_base, k_base): 1}
    W2 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U2); V_terms.append(V2); W_terms.append(W2)
    
    U3 = {u_idx(i_base, j_base): 1}
    V3 = {v_idx(j_base, k_base+1): 1, v_idx(j_base+1, k_base+1): -1}
    W3 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U3); V_terms.append(V3); W_terms.append(W3)
    
    U4 = {u_idx(i_base+1, j_base+1): 1}
    V4 = {v_idx(j_base+1, k_base): 1, v_idx(j_base, k_base): -1}
    W4 = {w_idx(k_base, i_base+1): 1}
    U_terms.append(U4); V_terms.append(V4); W_terms.append(W4)
    
    U5 = {u_idx(i_base, j_base): 1, u_idx(i_base, j_base+1): 1}
    V5 = {v_idx(j_base+1, k_base+1): 1}
    W5 = {w_idx(k_base+1, i_base): 1}
    U_terms.append(U5); V_terms.append(V5); W_terms.append(W5)
    
    U6 = {u_idx(i_base+1, j_base): 1, u_idx(i_base, j_base): -1}
    V6 = {v_idx(j_base, k_base): 1, v_idx(j_base, k_base+1): 1}
    W6 = {w_idx(k_base, i_base+1): 1, w_idx(k_base+1, i_base+1): 1}
    U_terms.append(U6); V_terms.append(V6); W_terms.append(W6)
    
    U7 = {u_idx(i_base, j_base+1): 1, u_idx(i_base+1, j_base+1): -1}
    V7 = {v_idx(j_base+1, k_base): 1, v_idx(j_base+1, k_base+1): 1}
    W7 = {w_idx(k_base, i_base): 1, w_idx(k_base+1, i_base): 1}
    U_terms.append(U7); V_terms.append(V7); W_terms.append(W7)
    
    return U_terms, V_terms, W_terms

def construct():
    start_time = time.time()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Two 2x2 Strassen blocks for (i,j) in {0,1}^2
    for kb in [0, 2]:
        U_terms, V_terms, W_terms = _strassen_2x2_block(0, 0, kb)
        for u_t, v_t, w_t in zip(U_terms, V_terms, W_terms):
            u_vec, v_vec, w_vec = _make_term(u_t, v_t, w_t)
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    # Row i=2: 3 cols x 4 rows = 12 entries, pair k values
    for j in range(m):
        for k_pair in [(0, 2), (1, 3)]:
            k1, k2 = k_pair
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            u_vec[2 * m + j] = 1
            v_vec[j * p + k1] = 1
            v_vec[j * p + k2] = 1
            w_vec[k1 * n + 2] = 1
            w_vec[k2 * n + 2] = 1
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Schoolbook fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]) if res['U'] else 0)
