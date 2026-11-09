import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin

    def time_left():
        return DEADLINE - (time.time() - t0)

    # Residual calculation
    def calculate_residual(U, V, W):
        r = U.shape[1]
        # Residual tensor R[i, j1, j2, k]
        # R[i, j1, j2, k] = sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] - delta_{j1, j2}
        # We want max abs residual or L2 norm.
        max_res = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for c in range(r):
                            s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
                        expected = 1.0 if j1 == j2 else 0.0
                        res = abs(s - expected)
                        if res > max_res:
                            max_res = res
        return max_res

    # ALS Solver
    def solve_als(U, V, W, rank, max_iters=1000, tol=1e-8):
        # Precompute some structures
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        for it in range(max_iters):
            if time_left() < 10:
                break
                
            # 1. Update U
            # For each i, j1, we have a linear system in U[:, c]
            # Actually, it's easier to solve for all U at once?
            # The equation is: sum_c U[i*m+j1, c] * (W[k*n+i, c] * V[j2*p+k, c]) = delta_{j1, j2}
            # Let A_c = W[k*n+i, c] * V[j2*p+k, c]
            # This couples i, j1, j2, k.
            
            # Better approach: Fix V and W. The residual is linear in U.
            # We can minimize sum_{i, j1, j2, k} (sum_c U[i*m+j1, c] * K[i, j2, k, c] - delta_{j1, j2})^2
            # where K[i, j2, k, c] = W[k*n+i, c] * V[j2*p+k, c]
            
            # This is a least squares problem for each (i, j1) pair?
            # No, U[i*m+j1, c] appears in the sum for all j2, k.
            # So for a fixed i, j1, we have a vector U_vec of length rank.
            # The residual for this U_vec is:
            # R_{j2, k} = sum_c U_vec[c] * K[i, j2, k, c] - delta_{j1, j2}
            # We want to minimize || R ||^2.
            # This is a linear least squares problem: M * U_vec = b
            # where M is (m*p) x rank, and b is (m*p) x 1.
            # M[j2*p+k, c] = K[i, j2, k, c] = W[k*n+i, c] * V[j2*p+k, c]
            # b[j2*p+k] = delta_{j1, j2} (which is 1 if k=0? No, delta is on j1, j2. k is free.)
            # Wait, the delta is delta_{j1, j2}. It does not depend on k or i.
            # So b[j2*p+k] = 1 if j1==j2 else 0.
            
            for i in range(n):
                for j1 in range(m):
                    # Construct M (m*p) x rank
                    M = np.zeros((m*p, rank))
                    b = np.zeros(m*p)
                    for j2 in range(m):
                        for k in range(p):
                            idx_v = j2*p + k
                            idx_w = k*n + i
                            for c in range(rank):
                                M[idx_v, c] = W[idx_w, c] * V[idx_v, c]
                            b[idx_v] = 1.0 if j1 == j2 else 0.0
                    
                    # Solve M * U_vec = b
                    try:
                        U_vec, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        U[i*m+j1, :] = U_vec
                    except np.linalg.LinAlgError:
                        pass

            # 2. Update V
            # Similar logic. Fix U, W.
            # Residual linear in V.
            # For fixed j2, k:
            # R_{i, j1} = sum_c V[j2*p+k, c] * (W[k*n+i, c] * U[i*m+j1, c]) - delta_{j1, j2}
            # M[j1, c] = W[k*n+i, c] * U[i*m+j1, c] -- wait, i is in the index.
            # The residual is over i, j1.
            # So for fixed j2, k, we solve for V_vec (length rank).
            # M is (n*m) x rank.
            # M[i*m+j1, c] = W[k*n+i, c] * U[i*m+j1, c]
            # b[i*m+j1] = delta_{j1, j2} (which is 1 if j1==j2, else 0).
            
            for j2 in range(m):
                for k in range(p):
                    M = np.zeros((n*m, rank))
                    b = np.zeros(n*m)
                    for i in range(n):
                        for j1 in range(m):
                            idx_u = i*m + j1
                            idx_w = k*n + i
                            for c in range(rank):
                                M[idx_u, c] = W[idx_w, c] * U[idx_u, c]
                            b[idx_u] = 1.0 if j1 == j2 else 0.0
                    
                    try:
                        V_vec, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        V[j2*p+k, :] = V_vec
                    except np.linalg.LinAlgError:
                        pass

            # 3. Update W
            # Fix U, V.
            # Residual linear in W.
            # For fixed k, i:
            # R_{j1, j2} = sum_c W[k*n+i, c] * (U[i*m+j1, c] * V[j2*p+k, c]) - delta_{j1, j2}
            # M is (m*m) x rank.
            # M[j1*m+j2, c] = U[i*m+j1, c] * V[j2*p+k, c]
            # b[j1*m+j2] = delta_{j1, j2}.
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m*m, rank))
                    b = np.zeros(m*m)
                    for j1 in range(m):
                        for j2 in range(m):
                            idx_u = i*m + j1
                            idx_v = j2*p + k
                            for c in range(rank):
                                M[j1*m+j2, c] = U[idx_u, c] * V[idx_v, c]
                            b[j1*m+j2] = 1.0 if j1 == j2 else 0.0
                    
                    try:
                        W_vec, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        W[k*n+i, :] = W_vec
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            res = calculate_residual(U, V, W)
            if res < tol:
                break

        return U, V, W

    # Try to find a solution for rank 22
    best_rank = 63
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    # Initialize with schoolbook
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, c] = 1.0
                best_V[j*p+k, c] = 1.0
                best_W[k*n+i, c] = 1.0
                c += 1

    random.seed(42)
    np.random.seed(42)

    for target_rank in [22, 23, 24, 25, 26, 27, 28, 30]:
        if time_left() < 20:
            break
        if target_rank >= best_rank:
            continue

        for attempt in range(10):
            if time_left() < 5:
                break
            
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            U, V, W = solve_als(U, V, W, target_rank, max_iters=500, tol=1e-7)
            
            res = calculate_residual(U, V, W)
            if res < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == target_rank:
            break

    def to_list(arr):
        result = []
        for row in arr:
            new_row = []
            for x in row:
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

# EVOLVE-BLOCK-END
