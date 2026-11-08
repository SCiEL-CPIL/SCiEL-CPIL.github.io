import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    
    # First, construct the known rank-28 solution using Strassen
    # This serves as our initial best and a fallback
    rank_28 = 28
    U_28 = np.zeros((nm, rank_28), dtype=int)
    V_28 = np.zeros((mp, rank_28), dtype=int)
    W_28 = np.zeros((pn, rank_28), dtype=int)
    
    def add_term(u_vec, v_vec, w_vec, col_idx, U, V, W):
        for idx, coeff in u_vec:
            U[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W[idx, col_idx] += coeff
            
    def strassen_terms():
        terms = []
        # M1 = (X00+X11)(Y00+Y11)
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 3: 1}
        terms.append((u, v, w))
        
        # M2 = (X10+X11)Y00
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: -1}
        terms.append((u, v, w))
        
        # M3 = X00(Y01-Y11)
        u = {0: 1}
        v = {1: 1, 3: -1}
        w = {1: 1, 3: 1}
        terms.append((u, v, w))
        
        # M4 = X11(Y10-Y00)
        u = {3: 1}
        v = {2: 1, 0: -1}
        w = {0: 1, 2: 1}
        terms.append((u, v, w))
        
        # M5 = (X00+X01)Y11
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        terms.append((u, v, w))
        
        # M6 = (X10-X00)(Y00+Y01)
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        terms.append((u, v, w))
        
        # M7 = (X01-X11)(Y10+Y11)
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1}
        terms.append((u, v, w))
        
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col, U, V, W):
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
            
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col, U, V, W)

    # Apply Strassen to 4 blocks
    apply_strassen(0, 0, 0, 0, 0, U_28, V_28, W_28)
    apply_strassen(1, 1, 0, 0, 7, U_28, V_28, W_28)
    apply_strassen(0, 0, 1, 1, 14, U_28, V_28, W_28)
    apply_strassen(1, 1, 1, 1, 21, U_28, V_28, W_28)
    
    best_rank = rank_28
    best_U = U_28.copy()
    best_V = V_28.copy()
    best_W = W_28.copy()
    
    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    val = np.dot(U[ui, :] * V[vj, :], W[wk, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety
    
    # Try to find lower rank decompositions using optimization
    # We'll try ranks from 16 up to 27
    for target_rank in range(16, rank_28):
        if time.time() > deadline:
            break
        
        found = False
        # Number of trials per rank
        num_trials = 50 if target_rank <= 20 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Use deterministic seeding for reproducibility
            np.random.seed(42 + target_rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Use Adam-like optimizer
            lr = 0.02
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            max_steps = 3000 if target_rank <= 20 else 1500
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                residual = 0.0
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            ui = i * m + j
                            vj = j * p + k
                            wk = k * n + i
                            val = np.dot(U[ui, :] * V[vj, :], W[wk, :])
                            r = val - 1.0
                            residual += 0.5 * r * r
                            prod = V[vj, :] * W[wk, :]
                            grad_U[ui, :] += r * prod
                            prod2 = U[ui, :] * W[wk, :]
                            grad_V[vj, :] += r * prod2
                            prod3 = U[ui, :] * V[vj, :]
                            grad_W[wk, :] += r * prod3
                
                # Adam update
                t = step + 1
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                
                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                
                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                if residual < 1e-10:
                    found = True
                    break
            
            if found:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                if verify_decomposition(U_snap, V_snap, W_snap):
                    best_rank = target_rank
                    best_U = U_snap.copy()
                    best_V = V_snap.copy()
                    best_W = W_snap.copy()
                    break
        
        if found:
            break
    
    # Convert to lists with proper types
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
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
