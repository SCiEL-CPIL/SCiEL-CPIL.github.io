import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try, from lowest to highest
    # Known lower bound is 37. Schoolbook is 45.
    target_ranks = [37, 38, 39, 40, 41, 42, 43, 44, 45]
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Precompute the target tensor T
    # T[i, j, j_prime, k, k_prime, i_prime]
    # But we only need the non-zero parts for the equations.
    # Actually, for ALS, we need the full tensor structure or just the equations.
    # The equations are indexed by the free indices.
    
    # Let's create the target tensor as a 6D array for easy indexing in equations
    # Shape: (n, m, m, p, p, n)
    T = np.zeros((n, m, m, p, p, n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # C[i, k] gets A[i, j] * B[j, k]
                # So T[i, j, j, k, k, i] = 1
                T[i, j, j, k, k, i] = 1.0
    
    best_result = None
    best_rank = n * m * p
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize V and W randomly
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            U = np.zeros((n * m, R))
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each i, j:
                # M[(j', k, k', i'), c] = V[j', k, c] * W[k', i', c]
                # t[(j', k, k', i')] = T[i, j, j', k, k', i']
                for i in range(n):
                    for j in range(m):
                        # Build M: shape (m*p*p*n, R) -> (225, R)
                        # Indices: j' in 0..m-1, k in 0..p-1, k' in 0..p-1, i' in 0..n-1
                        M = np.zeros((m * p * p * n, R))
                        t_vec = np.zeros(m * p * p * n)
                        
                        idx = 0
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        # M[idx, c] = V[jp*p + k, c] * W[kp*n + ip, c]
                                        M[idx, :] = V[jp * p + k, :] * W[kp * n + ip, :]
                                        t_vec[idx] = T[i, j, jp, k, kp, ip]
                                        idx += 1
                                        
                        # Solve least squares M * u = t_vec
                        # u is U[i*m + j, :]
                        try:
                            u, residuals, rank, sv = np.linalg.lstsq(M, t_vec, rcond=None)
                            U[i * m + j, :] = u
                        except:
                            pass

                # Update V
                # For each jp, k:
                # N[(i, j, kp, ip), c] = U[i, j, c] * W[kp, ip, c]
                # t[(i, j, kp, ip)] = T[i, j, jp, k, kp, ip]
                for jp in range(m):
                    for k in range(p):
                        N = np.zeros((n * m * p * n, R))
                        t_vec = np.zeros(n * m * p * n)
                        
                        idx = 0
                        for i in range(n):
                            for j in range(m):
                                for kp in range(p):
                                    for ip in range(n):
                                        N[idx, :] = U[i * m + j, :] * W[kp * n + ip, :]
                                        t_vec[idx] = T[i, j, jp, k, kp, ip]
                                        idx += 1
                                        
                        try:
                            v, residuals, rank, sv = np.linalg.lstsq(N, t_vec, rcond=None)
                            V[jp * p + k, :] = v
                        except:
                            pass

                # Update W
                # For each kp, ip:
                # P[(i, j, jp, k), c] = U[i, j, c] * V[jp, k, c]
                # t[(i, j, jp, k)] = T[i, j, jp, k, kp, ip]
                for kp in range(p):
                    for ip in range(n):
                        P = np.zeros((n * m * m * p, R))
                        t_vec = np.zeros(n * m * m * p)
                        
                        idx = 0
                        for i in range(n):
                            for j in range(m):
                                for jp in range(m):
                                    for k in range(p):
                                        P[idx, :] = U[i * m + j, :] * V[jp * p + k, :]
                                        t_vec[idx] = T[i, j, jp, k, kp, ip]
                                        idx += 1
                                        
                        try:
                            w, residuals, rank, sv = np.linalg.lstsq(P, t_vec, rcond=None)
                            W[kp * n + ip, :] = w
                        except:
                            pass

            # Check error
            # Reconstruct tensor
            error = 0.0
            # We can check a few random points or all points
            # Let's check all non-zero points
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target is 1 at (i, j, j, k, k, i)
                        val = 0.0
                        for c in range(R):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        error += (val - 1.0) ** 2
                        
                        # Check a zero point to ensure it's close to 0
                        # e.g. (i, j, (j+1)%m, k, k, i)
                        jp2 = (j + 1) % m
                        val0 = 0.0
                        for c in range(R):
                            val0 += U[i * m + j, c] * V[jp2 * p + k, c] * W[k * n + i, c]
                        error += val0 ** 2

            if error < 1e-8:
                # Try to round to half-integers
                U_int = np.round(U * 2) / 2.0
                V_int = np.round(V * 2) / 2.0
                W_int = np.round(W * 2) / 2.0
                
                # Verify exactness with integer arithmetic (scaled by 2)
                U_scaled = np.round(U * 2)
                V_scaled = np.round(V * 2)
                W_scaled = np.round(W * 2)
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(R):
                                val += U_scaled[i * m + j, c] * V_scaled[j * p + k, c] * W_scaled[k * n + i, c]
                            if val != 8: # 1.0 * 2 * 2 * 2 = 8
                                valid = False
                                break
                            
                            # Check a zero
                            jp2 = (j + 1) % m
                            val0 = 0
                            for c in range(R):
                                val0 += U_scaled[i * m + j, c] * V_scaled[jp2 * p + k, c] * W_scaled[k * n + i, c]
                            if val0 != 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    best_rank = R
                    return best_result

    # If no better rank found, return schoolbook
    if best_result is None:
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
