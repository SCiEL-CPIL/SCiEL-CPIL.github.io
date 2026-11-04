import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 50s margin
    
    # Target tensor
    # T[i, j, k] = 1 if valid
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Try ranks from low to high
    # Known upper bound is 24.
    # Try 20, 19, 18
    ranks_to_try = [20, 19, 18]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            max_iter = 50
            tol = 1e-10
            
            for it in range(max_iter):
                # Solve for U
                # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
                # For fixed V, W, U is linear.
                # Reshape T to (n*m, m*p, p*n)? No, indices are coupled.
                # T_{i,j,k} = sum_r U_{i,j,r} (V_{j,k,r} W_{k,i,r})? No.
                # It's a 3-way tensor.
                # U_{i,j,r} = sum_{j', k'} T_{i,j',k'} V_{j',k,r} W_{k,i,r} / sum_{j',k'} V_{j',k,r}^2 W_{k,i,r}^2 ?
                # This is not straightforward because V and W are coupled in the summation.
                # Standard ALS for CP decomposition:
                # U_{i, r} = sum_{j, k} T_{i, j, k} V_{j, r} W_{k, r} / sum_{j, k} V_{j, r}^2 W_{k, r}^2
                # But here the indices are not independent in the same way.
                # T_{i, j, k} depends on i, j, k.
                # U index is (i, j). V index is (j, k). W index is (k, i).
                # So U_{i, j, r} = sum_{k} T_{i, j, k} V_{j, k, r} W_{k, i, r} / sum_{k} V_{j, k, r}^2 W_{k, i, r}^2 ?
                # No, the summation for U_{i, j, r} should be over the other indices involved in the term.
                # The term is U_{i, j, r} V_{j, k, r} W_{k, i, r}.
                # To solve for U_{i, j, r}, we fix V and W.
                # The contribution to T_{i, j, k} from U_{i, j, r} is U_{i, j, r} V_{j, k, r} W_{k, i, r}.
                # Summing over k, the total contribution of U_{i, j, r} to the tensor slice (i, j) is:
                # sum_k U_{i, j, r} V_{j, k, r} W_{k, i, r} = U_{i, j, r} sum_k V_{j, k, r} W_{k, i, r}.
                # This doesn't look like a standard linear system.
                # Actually, the tensor is T_{i, j, k}.
                # We want to minimize || T - sum_r U_r \otimes V_r \otimes W_r ||^2.
                # U_r is a vector of length n*m. V_r is length m*p. W_r is length p*n.
                # The r-th term is the outer product of U_r, V_r, W_r.
                # Element (i, j, k) of the r-th term is U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i].
                # This is a valid CP decomposition.
                # ALS update for U_r:
                # U_r[i*m+j] = sum_{k} T[i, j, k] V_r[j*p+k] W_r[k*n+i] / sum_{k} V_r[j*p+k]^2 W_r[k*n+i]^2 ?
                # No, the denominator should be sum over the indices that are summed out.
                # The term is U_{i,j,r} V_{j,k,r} W_{k,i,r}.
                # For a fixed r, and fixed V_r, W_r, the part of the objective function involving U_r is:
                # sum_{i,j,k} (T_{i,j,k} - ... - U_{i,j,r} V_{j,k,r} W_{k,i,r} - ...)^2
                # The linear part in U_{i,j,r} is:
                # -2 sum_{i,j,k} T_{i,j,k} U_{i,j,r} V_{j,k,r} W_{k,i,r}
                # + sum_{i,j,k} U_{i,j,r}^2 V_{j,k,r}^2 W_{k,i,r}^2
                # Note that V_{j,k,r} and W_{k,i,r} depend on k.
                # So the coefficient of U_{i,j,r}^2 is sum_k V_{j,k,r}^2 W_{k,i,r}^2.
                # And the linear coefficient is sum_k T_{i,j,k} V_{j,k,r} W_{k,i,r}.
                # So U_{i,j,r} = sum_k T_{i,j,k} V_{j,k,r} W_{k,i,r} / sum_k V_{j,k,r}^2 W_{k,i,r}^2.
                # This is a valid update.
                
                # Let's implement this.
                
                # Update U
                for r_idx in range(r):
                    # V_r is V[:, r_idx]
                    # W_r is W[:, r_idx]
                    # U_r is U[:, r_idx]
                    for i in range(n):
                        for j in range(m):
                            num = 0.0
                            den = 0.0
                            for k in range(p):
                                v_val = V[j * p + k, r_idx]
                                w_val = W[k * n + i, r_idx]
                                num += T[i, j, k] * v_val * w_val
                                den += v_val * v_val * w_val * w_val
                            if den > 1e-12:
                                U[i * m + j, r_idx] = num / den
                            else:
                                U[i * m + j, r_idx] = 0.0
                                
                # Update V
                for r_idx in range(r):
                    for j in range(m):
                        for k in range(p):
                            num = 0.0
                            den = 0.0
                            for i in range(n):
                                u_val = U[i * m + j, r_idx]
                                w_val = W[k * n + i, r_idx]
                                num += T[i, j, k] * u_val * w_val
                                den += u_val * u_val * w_val * w_val
                            if den > 1e-12:
                                V[j * p + k, r_idx] = num / den
                            else:
                                V[j * p + k, r_idx] = 0.0
                                
                # Update W
                for r_idx in range(r):
                    for k in range(p):
                        for i in range(n):
                            num = 0.0
                            den = 0.0
                            for j in range(m):
                                u_val = U[i * m + j, r_idx]
                                v_val = V[j * p + k, r_idx]
                                num += T[i, j, k] * u_val * v_val
                                den += u_val * u_val * v_val * v_val
                            if den > 1e-12:
                                W[k * n + i, r_idx] = num / den
                            else:
                                W[k * n + i, r_idx] = 0.0
                                
            # Check error
            # Reconstruct
            T_rec = np.zeros((n, m, p))
            for r_idx in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rec[i, j, k] += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                            
            error = np.linalg.norm(T - T_rec)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            T_snap = np.zeros((n, m, p))
            for r_idx in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_snap[i, j, k] += U_snap[i * m + j, r_idx] * V_snap[j * p + k, r_idx] * W_snap[k * n + i, r_idx]
                            
            snap_error = np.linalg.norm(T - T_snap)
            
            if snap_error < 1e-9:
                # Found a valid solution
                return {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
