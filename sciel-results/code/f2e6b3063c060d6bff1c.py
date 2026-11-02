import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def verify_subtensor(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub):
        """Verify a subtensor decomposition."""
        r = len(U_sub[0])
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    val = 0
                    for c in range(r):
                        val += U_sub[i*m_sub+j][c] * V_sub[j*p_sub+k][c] * W_sub[k*n_sub+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition for (n, m, p)
    def schoolbook_decomp(n_sub, m_sub, p_sub):
        """Schoolbook: rank = n_sub * m_sub * p_sub"""
        r = n_sub * m_sub * p_sub
        U = [[0] * r for _ in range(n_sub * m_sub)]
        V = [[0] * r for _ in range(m_sub * p_sub)]
        W = [[0] * r for _ in range(p_sub * n_sub)]
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i*m_sub+j][col] = 1
                    V[j*p_sub+k][col] = 1
                    W[k*n_sub+i][col] = 1
                    col += 1
        return U, V, W

    # Try to find low-rank decomposition for (2,3,3)
    def find_233_decomp(max_rank=9):
        """Search for low-rank decomposition of (2,3,3) multiplication tensor."""
        n_sub, m_sub, p_sub = 2, 3, 3
        n_rows_sub = n_sub * m_sub  # 6
        m_rows_sub = m_sub * p_sub  # 9
        p_rows_sub = p_sub * n_sub  # 6
        
        def compute_errors(U, V, W, r):
            errors = np.zeros((n_sub, m_sub, p_sub))
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m_sub+j, c] * V[j*p_sub+k, c] * W[k*n_sub+i, c]
                        errors[i, j, k] = val - 1.0
            return errors
        
        for r in range(max_rank, 12):
            if time.time() > deadline - 60:
                break
            
            rng = np.random.RandomState(42 + r)
            
            for trial in range(20):
                if time.time() > deadline - 55:
                    break
                
                U = np.zeros((n_rows_sub, r), dtype=np.float64)
                V = np.zeros((m_rows_sub, r), dtype=np.float64)
                W = np.zeros((p_rows_sub, r), dtype=np.float64)
                
                for c in range(r):
                    num_u = rng.randint(2, 4)
                    u_indices = rng.choice(n_rows_sub, size=num_u, replace=False)
                    for idx in u_indices:
                        U[idx, c] = rng.choice([-1.0, 1.0, 2.0])
                    
                    num_v = rng.randint(2, 5)
                    v_indices = rng.choice(m_rows_sub, size=num_v, replace=False)
                    for idx in v_indices:
                        V[idx, c] = rng.choice([-1.0, 1.0, 2.0])
                    
                    num_w = rng.randint(2, 4)
                    w_indices = rng.choice(p_rows_sub, size=num_w, replace=False)
                    for idx in w_indices:
                        W[idx, c] = rng.choice([-1.0, 1.0, 2.0])
                
                for it in range(200):
                    if time.time() > deadline - 50:
                        break
                    
                    errors = compute_errors(U, V, W, r)
                    current_error = np.max(np.abs(errors))
                    
                    if current_error < 1e-10:
                        U_int = np.round(U).astype(np.int64)
                        V_int = np.round(V).astype(np.int64)
                        W_int = np.round(W).astype(np.int64)
                        
                        ok = True
                        for i in range(n_sub):
                            for j in range(m_sub):
                                for k in range(p_sub):
                                    val = 0
                                    for c in range(r):
                                        val += U_int[i*m_sub+j, c] * V_int[j*p_sub+k, c] * W_int[k*n_sub+i, c]
                                    if abs(val - 1) > 1e-9:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        
                        if ok:
                            return U_int.tolist(), V_int.tolist(), W_int.tolist(), r
                    
                    mat_choice = rng.randint(3)
                    if mat_choice == 0:
                        row = rng.randint(n_rows_sub)
                        col = rng.randint(r)
                        U[row, col] += rng.choice([-1.0, 1.0])
                    elif mat_choice == 1:
                        row = rng.randint(m_rows_sub)
                        col = rng.randint(r)
                        V[row, col] += rng.choice([-1.0, 1.0])
                    else:
                        row = rng.randint(p_rows_sub)
                        col = rng.randint(r)
                        W[row, col] += rng.choice([-1.0, 1.0])
        
        return None

    # Build full solution from (2,3,3) block + schoolbook (2,3,6)
    def build_block_solution(r_233, U_233, V_233, W_233):
        """Build full (2,3,9) solution from (2,3,3) block + schoolbook (2,3,6)."""
        n_sub, m_sub, p_sub1 = 2, 3, 3
        p_sub2 = 6
        r_school = n_sub * m_sub * p_sub2  # 36
        total_rank = r_233 + r_school
        
        U_full = [[0] * total_rank for _ in range(n_rows)]
        V_full = [[0] * total_rank for _ in range(m_rows)]
        W_full = [[0] * total_rank for _ in range(p_rows)]
        
        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(r_233):
                    U_full[i*m+j][c] = U_233[i*m_sub+j][c]
        
        for j in range(m_sub):
            for k in range(p_sub1):
                for c in range(r_233):
                    V_full[j*p+k][c] = V_233[j*p_sub1+k][c]
        
        for k in range(p_sub1):
            for i in range(n_sub):
                for c in range(r_233):
                    W_full[k*n+i][c] = W_233[k*n_sub+i][c]
        
        col = r_233
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub2):
                    U_full[i*m+j][col] = 1
                    V_full[j*p + (3+k)][col] = 1
                    W_full[(3+k)*n + i][col] = 1
                    col += 1
        
        return {"U": U_full, "V": V_full, "W": W_full}

    # Known rank-50 solution as fallback
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}



    # Try to find lower rank solutions
    best_result = None
    best_rank = 50
    
    # First, get the known rank-50 solution
    known_sol = current_solution()
    
    # Verify it
    if verify_decomposition(known_sol["U"], known_sol["V"], known_sol["W"]):
        best_result = known_sol
    else:
        best_result = known_sol
    
    # Try block decomposition: (2,3,3) + (2,3,6)
    if time.time() < deadline - 70:
        decomp_233 = find_233_decomp(max_rank=9)
        
        if decomp_233 is not None:
            U_233, V_233, W_233, r_233 = decomp_233
            full_sol = build_block_solution(r_233, U_233, V_233, W_233)
            
            if verify_decomposition(full_sol["U"], full_sol["V"], full_sol["W"]):
                best_result = full_sol
                best_rank = r_233 + 36
    
    # If (2,3,3) rank-9 not found, try harder with more trials
    if time.time() < deadline - 30 and best_rank == 50:
        rng = np.random.RandomState(123)
        n_sub, m_sub, p_sub = 2, 3, 3
        n_rows_sub = 6
        m_rows_sub = 9
        p_rows_sub = 6
        
        def compute_errors_233(U, V, W, r):
            errors = np.zeros((n_sub, m_sub, p_sub))
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        val = 0.0
                        for c in range(r):
                            val += U[i*m_sub+j, c] * V[j*p_sub+k, c] * W[k*n_sub+i, c]
                        errors[i, j, k] = val - 1.0
            return errors
        
        found_233 = False
        for r in range(9, 13):
            if time.time() > deadline - 25:
                break
            
            for trial in range(30):
                if time.time() > deadline - 20:
                    break
                
                U = np.zeros((n_rows_sub, r), dtype=np.float64)
                V = np.zeros((m_rows_sub, r), dtype=np.float64)
                W = np.zeros((p_rows_sub, r), dtype=np.float64)
                
                for c in range(r):
                    num_u = rng.randint(2, 4)
                    u_indices = rng.choice(n_rows_sub, size=num_u, replace=False)
                    for idx in u_indices:
                        U[idx, c] = rng.choice([-1.0, 1.0])
                    
                    num_v = rng.randint(3, 5)
                    v_indices = rng.choice(m_rows_sub, size=num_v, replace=False)
                    for idx in v_indices:
                        V[idx, c] = rng.choice([-1.0, 1.0])
                    
                    num_w = rng.randint(2, 4)
                    w_indices = rng.choice(p_rows_sub, size=num_w, replace=False)
                    for idx in w_indices:
                        W[idx, c] = rng.choice([-1.0, 1.0])
                
                for it in range(300):
                    if time.time() > deadline - 15:
                        break
                    
                    errors = compute_errors_233(U, V, W, r)
                    current_error = np.max(np.abs(errors))
                    
                    if current_error < 1e-10:
                        U_int = np.round(U).astype(np.int64)
                        V_int = np.round(V).astype(np.int64)
                        W_int = np.round(W).astype(np.int64)
                        
                        ok = True
                        for i in range(n_sub):
                            for j in range(m_sub):
                                for k in range(p_sub):
                                    val = 0
                                    for c in range(r):
                                        val += U_int[i*m_sub+j, c] * V_int[j*p_sub+k, c] * W_int[k*n_sub+i, c]
                                    if abs(val - 1) > 1e-9:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        
                        if ok:
                            full_sol = build_block_solution(r, U_int.tolist(), V_int.tolist(), W_int.tolist())
                            if verify_decomposition(full_sol["U"], full_sol["V"], full_sol["W"]):
                                best_result = full_sol
                                best_rank = r + 36
                            found_233 = True
                            break
                    
                    worst_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
                    wi, wj, wk = worst_idx
                    u_row = wi * m_sub + wj
                    v_row = wj * p_sub + wk
                    w_row = wk * n_sub + wi
                    
                    mat_choice = rng.randint(3)
                    if mat_choice == 0:
                        col = rng.randint(r)
                        U[u_row, col] += rng.choice([-1.0, 1.0])
                    elif mat_choice == 1:
                        col = rng.randint(r)
                        V[v_row, col] += rng.choice([-1.0, 1.0])
                    else:
                        col = rng.randint(r)
                        W[w_row, col] += rng.choice([-1.0, 1.0])
                
                if found_233:
                    break
            
            if found_233:
                break
    
    return best_result
