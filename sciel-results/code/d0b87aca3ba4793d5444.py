import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    _start_time = time.time()
    _deadline = _start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Target: Standard Matrix Multiplication
    # We use random test matrices to define the residual
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Pre-flatten A and B
    A_flat = A_test.flatten()
    B_flat = B_test.flatten()
    
    def residual(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # Compute S_r = sum_{i,j} U_{ijr} A_{ij}
        # U is (n, m, R). A is (n, m).
        # S_r = np.einsum('ijr,ij->r', U, A_test)
        S = np.einsum('ijr,ij->r', U, A_test)
        
        # Compute T_r = sum_{j,k} V_{jkr} B_{jk}
        T = np.einsum('jkr,jk->r', V, B_test)
        
        # Compute C_{ik} = sum_r S_r T_r W_{kir}
        # W is (p, n, R). Indices k, i, r.
        # C is (n, p). Indices i, k.
        # C_{ik} = sum_r (S_r T_r) W_{k,i,r}
        ST = S * T
        # W_transpose for einsum?
        # We want C[i, k] = sum_r ST[r] * W[k, i, r]
        C_computed = np.einsum('r,kir->ik', ST, W)
        
        return (C_expected - C_computed).flatten()

    def get_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        def add_term(u_vec, v_vec, w_vec, col_idx):
            for idx, coeff in u_vec:
                U[idx, col_idx] += coeff
            for idx, coeff in v_vec:
                V[idx, col_idx] += coeff
            for idx, coeff in w_vec:
                W[idx, col_idx] += coeff
                
        def strassen_terms():
            terms = []
            u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}
            terms.append((u, v, w))
            u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}
            terms.append((u, v, w))
            u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}
            terms.append((u, v, w))
            u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}
            terms.append((u, v, w))
            u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}
            terms.append((u, v, w))
            u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}
            terms.append((u, v, w))
            u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}
            terms.append((u, v, w))
            return terms

        base_terms = strassen_terms()
        
        def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
            for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
                col = start_col + t_idx
                u_global = {}
                for loc_idx, coeff in u_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = a_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    u_global[g_idx] = u_global.get(g_idx, 0) + coeff
                
                v_global = {}
                for loc_idx, coeff in v_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = b_row_offset * 2 + r
                    g_c = b_col_offset * 2 + c
                    g_idx = g_r * 4 + g_c
                    v_global[g_idx] = v_global.get(g_idx, 0) + coeff
                
                w_global = {}
                for loc_idx, coeff in w_loc.items():
                    r = loc_idx // 2
                    c = loc_idx % 2
                    g_r = r
                    g_c = c_col_offset * 2 + c
                    g_idx = g_c * 2 + g_r
                    w_global[g_idx] = w_global.get(g_idx, 0) + coeff
                
                add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

        apply_strassen(0, 0, 0, 0, 0)
        apply_strassen(1, 1, 0, 0, 7)
        apply_strassen(0, 0, 1, 1, 14)
        apply_strassen(1, 1, 1, 1, 21)
        
        return U, V, W

    def verify_solution(U, V, W, R):
        C_computed = np.zeros((n, p))
        U_float = U.astype(float)
        V_float = V.astype(float)
        W_float = W.astype(float)
        
        # Using einsum for speed
        S = np.einsum('ijr,ij->r', U_float.reshape(n, m, R), A_test)
        T = np.einsum('jkr,jk->r', V_float.reshape(m, p, R), B_test)
        ST = S * T
        C_computed = np.einsum('r,kir->ik', ST, W_float.reshape(p, n, R))
        
        return np.allclose(C_expected, C_computed, atol=1e-6)

    U28, V28, W28 = get_strassen_28()
    best_result = {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    for R in [24, 25, 26, 27]:
        if time.time() > _deadline:
            break
        
        U_init = U28[:, :R]
        V_init = V28[:, :R]
        W_init = W28[:, :R]
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                residual, 
                x0, 
                args=(R,),
                method='trf',
                max_nfev=500
            )
            
            if result.cost < 1e-8:
                U_opt = result.x[:n*m*R].reshape(n*m, R)
                V_opt = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_opt = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Round to 0.5
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                if verify_solution(U_round, V_round, W_round, R):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                    
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
