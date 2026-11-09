import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s limit to be safe
    
    # Precompute target tensor coefficients
    # Target[i, k, a, b, c, d] = 1 if a==i, b==c, d==k else 0
    # We can flatten the indices for M and Target
    # Indices for M: (a, b, c, d) -> 6 * 27 = 162
    # Indices for i, k: 2 * 9 = 18
    
    targets = {}
    for i in range(n):
        for k in range(p):
            t_vec = np.zeros(m * m * p * p) # Wait, A is n x m, B is m x p.
            # A indices: a in [0, n), b in [0, m)
            # B indices: c in [0, m), d in [0, p)
            # Total pairs: n*m * m*p
            # Here n=2, m=3, p=9.
            # A pairs: 2*3=6. B pairs: 3*9=27. Total 162.
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            if a == i and b == c and d == k:
                                idx = (a * m + b) * (m * p) + (c * p + d)
                                t_vec[idx] = 1.0
            targets[(i, k)] = t_vec

    # List of (i, k) pairs
    ik_pairs = [(i, k) for i in range(n) for k in range(p)]
    
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from 21 to 35
    for R in range(21, 36):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                    
                # Compute W and residuals
                W = np.zeros((p * n, R))
                total_res = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                
                for i, k in ik_pairs:
                    t_vec = targets[(i, k)]
                    # Build M[i,k] of shape (162, R)
                    M = np.zeros((n * m * m * p, R))
                    for r in range(R):
                        u_vec = U[:, r] # shape (6,)
                        v_vec = V[:, r] # shape (27,)
                        # M[:, r] is outer product of u_vec and v_vec?
                        # No, M[alpha, r] = u[r, a,b] * v[r, c,d]
                        # alpha = (a,b,c,d)
                        # We can use np.outer(u_vec, v_vec) and reshape
                        M[:, r] = np.outer(u_vec, v_vec).flatten()
                    
                    # Solve M w = t_vec
                    w_vec, residuals, rank, s = np.linalg.lstsq(M, t_vec, rcond=None)
                    W[k * n + i, :] = w_vec
                    
                    res_vec = M @ w_vec - t_vec
                    res_norm_sq = np.dot(res_vec, res_vec)
                    total_res += res_norm_sq
                    
                    # Gradients
                    # dE/du[r, a,b] = 2 * sum_{i,k} sum_alpha res[i,k,alpha] * v[r, c(alpha), d(alpha)] * w[i,k,r]
                    # dE/dv[r, c,d] = 2 * sum_{i,k} sum_alpha res[i,k,alpha] * u[r, a(alpha), b(alpha)] * w[i,k,r]
                    
                    # We need to map alpha back to (a,b,c,d)
                    # This is slow if done in Python loops.
                    # Optimize:
                    # res_vec is shape (162,)
                    # w_vec is shape (R,)
                    # v_vec is shape (27,)
                    # u_vec is shape (6,)
                    
                    # For each r:
                    # grad_U[:, r] += 2 * sum_alpha res[alpha] * v[alpha_v] * w[r]
                    # alpha = (a,b) * 27 + (c,d)
                    # v index is (c,d)
                    # u index is (a,b)
                    
                    # We can reshape res_vec to (6, 27)
                    res_mat = res_vec.reshape(n * m, m * p)
                    
                    # grad_U[:, r] = 2 * w[r] * (res_mat @ v_vec) ?
                    # res_mat[a,b, c,d] = res[alpha]
                    # Sum over c,d: sum_{c,d} res[a,b,c,d] v[c,d]
                    # This is exactly (res_mat @ v_vec)[a,b]
                    grad_U[:, r] += 2.0 * w_vec[r] * (res_mat @ v_vec)
                    
                    # grad_V[:, r] = 2 * w[r] * (res_mat.T @ u_vec) ?
                    # Sum over a,b: sum_{a,b} res[a,b,c,d] u[a,b]
                    # This is (res_mat.T @ u_vec)[c,d]
                    grad_V[:, r] += 2.0 * w_vec[r] * (res_mat.T @ u_vec)
                
                # Check convergence
                if total_res < 1e-10:
                    # Found a solution
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                
                # Adaptive learning rate?
                if total_res > 1e5:
                    lr *= 0.5
                elif total_res < 1e-3:
                    lr *= 1.2

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
