import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 by 5x5 matrix.
    """
    n, m, p = 4, 5, 5
    deadline = time.time() + 110.0
    
    # --- Rank 96 Solution (Strassen blocks) ---
    rank_96 = 96
    U96 = [[0] * rank_96 for _ in range(n * m)]
    V96 = [[0] * rank_96 for _ in range(m * p)]
    W96 = [[0] * rank_96 for _ in range(p * n)]
    
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([2, 3], [0, 1], [0, 1])
    ]
    
    skip_set = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    skip_set.add((i, j, k))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in skip_set:
                    continue
                U96[i * m + j][col] = 1
                V96[j * p + k][col] = 1
                W96[k * n + i][col] = 1
                col += 1
                
    for I, J, K in blocks:
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U96[i1 * m + j1][col] += 1
        U96[i2 * m + j2][col] += 1
        V96[j1 * p + k1][col] += 1
        V96[j2 * p + k2][col] += 1
        W96[k1 * n + i1][col] += 1
        W96[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U96[i2 * m + j1][col] += 1
        U96[i2 * m + j2][col] += 1
        V96[j1 * p + k1][col] += 1
        W96[k1 * n + i2][col] += 1
        W96[k2 * n + i2][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U96[i1 * m + j1][col] += 1
        V96[j1 * p + k2][col] += 1
        V96[j2 * p + k2][col] += -1
        W96[k2 * n + i1][col] += 1
        W96[k2 * n + i2][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U96[i2 * m + j2][col] += 1
        V96[j2 * p + k1][col] += 1
        V96[j1 * p + k1][col] += -1
        W96[k1 * n + i1][col] += 1
        W96[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U96[i1 * m + j1][col] += 1
        U96[i1 * m + j2][col] += 1
        V96[j2 * p + k2][col] += 1
        W96[k1 * n + i1][col] += -1
        W96[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U96[i2 * m + j1][col] += 1
        U96[i1 * m + j1][col] += -1
        V96[j1 * p + k1][col] += 1
        V96[j1 * p + k2][col] += 1
        W96[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U96[i1 * m + j2][col] += 1
        U96[i2 * m + j2][col] += -1
        V96[j2 * p + k1][col] += 1
        V96[j2 * p + k2][col] += 1
        W96[k1 * n + i1][col] += 1
        col += 1
        
    best_U, best_V, best_W = [row[:] for row in U96], [row[:] for row in V96], [row[:] for row in W96]
    best_rank = rank_96

    # Try to find rank 95 by removing one term and solving a linearized system
    U_np = np.array(U96, dtype=float)
    V_np = np.array(V96, dtype=float)
    W_np = np.array(W96, dtype=float)
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_eq = len(indices)
    
    for r_remove in range(rank_96):
        if time.time() > deadline - 10:
            break
            
        # Compute residual after removing term r_remove
        residual = np.zeros(num_eq)
        for idx, (i, j, k) in enumerate(indices):
            u_val = U_np[i*m+j, r_remove]
            v_val = V_np[j*p+k, r_remove]
            w_val = W_np[k*n+i, r_remove]
            residual[idx] = -(u_val * v_val * w_val)
        
        if np.allclose(residual, 0, atol=1e-10):
            new_U = [[row[r] for r in range(rank_96) if r != r_remove] for row in U96]
            new_V = [[row[r] for r in range(rank_96) if r != r_remove] for row in V96]
            new_W = [[row[r] for r in range(rank_96) if r != r_remove] for row in W96]
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(new_U[i*m+j][r] * new_V[j*p+k][r] * new_W[k*n+i][r] for r in range(rank_96 - 1))
                        if abs(val - 1) > 1e-10:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = rank_96 - 1
                break
        
        # Build linear system
        num_vars = (rank_96 - 1) * (n*m + m*p + p*n)
        A_rows, A_cols, A_vals = [], [], []
        
        var_offset_U = 0
        var_offset_V = (rank_96 - 1) * n * m
        var_offset_W = (rank_96 - 1) * (n * m + m * p)
        
        for eq_idx, (i, j, k) in enumerate(indices):
            for r_prime in range(rank_96):
                if r_prime == r_remove:
                    continue
                r_new = r_prime if r_prime < r_remove else r_prime - 1
                
                u_val = U_np[i*m+j, r_prime]
                v_val = V_np[j*p+k, r_prime]
                w_val = W_np[k*n+i, r_prime]
                
                coeff_U = v_val * w_val
                if abs(coeff_U) > 1e-15:
                    A_rows.append(eq_idx)
                    A_cols.append(var_offset_U + r_new * n * m + i * m + j)
                    A_vals.append(coeff_U)
                
                coeff_V = u_val * w_val
                if abs(coeff_V) > 1e-15:
                    A_rows.append(eq_idx)
                    A_cols.append(var_offset_V + r_new * m * p + j * p + k)
                    A_vals.append(coeff_V)
                
                coeff_W = u_val * v_val
                if abs(coeff_W) > 1e-15:
                    A_rows.append(eq_idx)
                    A_cols.append(var_offset_W + r_new * p * n + k * n + i)
                    A_vals.append(coeff_W)
        
        if len(A_vals) == 0:
            continue
        
        try:
            from scipy.sparse import csr_matrix
            from scipy.optimize import lsq_linear
            A_sparse = csr_matrix((A_vals, (A_rows, A_cols)), shape=(num_eq, num_vars))
            result = lsq_linear(A_sparse, residual)
            if result.status != 0:
                continue
            
            x = result.x
            if np.max(np.abs(x)) > 1.0:
                continue
            
            new_U = [row[:] for row in U96]
            new_V = [row[:] for row in V96]
            new_W = [row[:] for row in W96]
            
            for r_prime in range(rank_96):
                if r_prime == r_remove:
                    continue
                r_new = r_prime if r_prime < r_remove else r_prime - 1
                
                for i in range(n):
                    for j in range(m):
                        delta = x[var_offset_U + r_new * n * m + i * m + j]
                        if abs(delta) > 1e-10:
                            new_U[i*m+j][r_prime] += delta
                
                for j in range(m):
                    for k in range(p):
                        delta = x[var_offset_V + r_new * m * p + j * p + k]
                        if abs(delta) > 1e-10:
                            new_V[j*p+k][r_prime] += delta
                
                for k in range(p):
                    for i in range(n):
                        delta = x[var_offset_W + r_new * p * n + k * n + i]
                        if abs(delta) > 1e-10:
                            new_W[k*n+i][r_prime] += delta
            
            new_U = [[row[r] for r in range(rank_96) if r != r_remove] for row in new_U]
            new_V = [[row[r] for r in range(rank_96) if r != r_remove] for row in new_V]
            new_W = [[row[r] for r in range(rank_96) if r != r_remove] for row in new_W]
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(new_U[i*m+j][r] * new_V[j*p+k][r] * new_W[k*n+i][r] for r in range(rank_96 - 1))
                        if abs(val - 1) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = rank_96 - 1
                break
                
        except Exception:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
