"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    deadline = time.time() + 130  # Reserve 20s margin for 150s total
    
    # Try to find a lower rank decomposition
    # Focus on ranks that are most likely to be achievable
    for target_rank in [30, 31, 32, 33]:
        remaining = deadline - time.time()
        if remaining < 10:
            break
        tl = min(25.0, remaining - 8)
        if tl < 5:
            break
        res = attempt_lower_rank(target_rank, time_limit=tl)
        if res is not None and valid_res(res, 3, 3, 4):
            return res
    
    # Fallback: return exactly rank 34 decomposition
    result = construct_r34()
    # Verify the fallback is valid
    if valid_res(result, 3, 3, 4):
        return result
    # If somehow invalid, try to fix or return as-is
    return result

def attempt_lower_rank(target_rank, time_limit=20.0):
    n, m, p = 3, 3, 4
    R = target_rank
    start_time = time.time()
    
    # Build the target tensor indices
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                indices_a.append(a)
                indices_b.append(b)
                indices_c.append(c)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    # Vectorized objective function with gradient
    def objective(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
        error = np.sum((vals - 1.0)**2)
        return error
    
    def objective_and_grad(params):
        U = params[0:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
        error = np.sum((vals - 1.0)**2)
        
        # Vectorized gradient computation
        dU = np.zeros((n*m, R))
        dV = np.zeros((m*p, R))
        dW = np.zeros((p*n, R))
        
        dv = 2 * (vals - 1.0)  # shape (36,)
        
        # For each of the 36 terms, accumulate gradients
        # U[indices_a[idx], :] * V[indices_b[idx], :] * W[indices_c[idx], :]
        # Use numpy advanced indexing
        U_sel = U[indices_a, :]  # (36, R)
        V_sel = V[indices_b, :]  # (36, R)
        W_sel = W[indices_c, :]  # (36, R)
        
        # dU contribution: dv[idx] * V_sel[idx] * W_sel[idx] for each row indices_a[idx]
        dU_contrib = dv[:, None] * V_sel * W_sel  # (36, R)
        # Scatter back
        np.add.at(dU, indices_a, dU_contrib)
        
        dV_contrib = dv[:, None] * U_sel * W_sel  # (36, R)
        np.add.at(dV, indices_b, dV_contrib)
        
        dW_contrib = dv[:, None] * U_sel * V_sel  # (36, R)
        np.add.at(dW, indices_c, dW_contrib)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return error, grad
    
    from scipy.optimize import minimize
    
    best_error = float('inf')
    best_params = None
    
    # Pre-compute the rank-34 base solution for structured initialization
    base = construct_r34()
    U_base = np.array(base['U'], dtype=float)
    V_base = np.array(base['V'], dtype=float)
    W_base = np.array(base['W'], dtype=float)
    
    num_restarts = min(10, max(4, int(time_limit / 2.5)))
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 3:
            break
        
        np.random.seed(1234 + restart * 7919)
        
        # Create initial guess with structure
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        if restart == 0:
            # Compress the rank-34 solution: project 34 columns into R
            # Use random linear combinations of pairs of base columns
            for r in range(R):
                c1 = np.random.randint(0, 34)
                c2 = np.random.randint(0, 34)
                w1, w2 = np.random.randn(2)
                U[:, r] = w1 * U_base[:, c1] + w2 * U_base[:, c2]
                V[:, r] = w1 * V_base[:, c1] + w2 * V_base[:, c2]
                W[:, r] = w1 * W_base[:, c1] + w2 * W_base[:, c2]
        
        elif restart == 1:
            # Block-structured initialization matching the 2x17 block pattern
            # The rank-34 has two blocks of 17 columns (j=0,1 and j=2,3)
            half = R // 2
            rem = R - 2 * half
            
            for r in range(half):
                # First block: j=0,1
                i = r % n
                k = (r // n) % m
                j = (r // (n*m)) % 2
                U[i*m+k, r] = 1.0
                V[k*p+j, r] = 1.0
                W[j*n+i, r] = 1.0
                # Add cross terms for off-diagonal
                if r % 3 == 0:
                    U[(i+1)%n*m+k, r] = 0.5
                    W[j*n+(i+1)%n, r] = -0.5
                elif r % 3 == 1:
                    V[k*p+(j+1)%2, r] = -0.5
                    W[j*n+i, r] = 1.5
                else:
                    U[i*m+(k+1)%m, r] = 0.5
                    V[(k+1)%m*p+j, r] = -0.5
            
            for r in range(half, R):
                r2 = r - half
                i = r2 % n
                k = (r2 // n) % m
                j = 2 + (r2 // (n*m)) % 2
                U[i*m+k, r] = 1.0
                V[k*p+j, r] = 1.0
                W[j*n+i, r] = 1.0
                if r2 % 3 == 0:
                    U[(i+1)%n*m+k, r] = 0.5
                    W[j*n+(i+1)%n, r] = -0.5
                elif r2 % 3 == 1:
                    V[k*p+(j+2)%4, r] = -0.5
                    W[j*n+i, r] = 1.5
                else:
                    U[i*m+(k+1)%m, r] = 0.5
                    V[(k+1)%m*p+j, r] = -0.5
        
        elif restart == 2:
            # Dense random with moderate scale
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
        
        elif restart == 3:
            # Use base solution with noise
            for r in range(R):
                c1 = np.random.randint(0, 34)
                c2 = np.random.randint(0, 34)
                w1, w2 = np.random.randn(2) * 0.8
                U[:, r] = w1 * U_base[:, c1] + w2 * U_base[:, c2]
                V[:, r] = w1 * V_base[:, c1] + w2 * V_base[:, c2]
                W[:, r] = w1 * W_base[:, c1] + w2 * W_base[:, c2]
                # Add small noise
                U[:, r] += np.random.randn(n*m) * 0.1
                V[:, r] += np.random.randn(m*p) * 0.1
                W[:, r] += np.random.randn(p*n) * 0.1
        
        elif restart == 4:
            # Sparse structured: each term has at most 2 nonzeros per factor
            for r in range(R):
                i = r % n
                k = (r // n) % m
                j = (r // (n*m)) % p
                
                U[i*m+k, r] = 1.0
                V[k*p+j, r] = 1.0
                W[j*n+i, r] = 1.0
                
                # Add one cross term
                if r % 2 == 0:
                    U[(i+1)%n*m+k, r] = 0.5
                    W[j*n+(i+1)%n, r] = -0.5
                else:
                    V[k*p+(j+1)%p, r] = -0.5
                    W[(j+1)%p*n+i, r] = 0.5
        
        else:
            # Random sparse with 2-3 nonzeros
            for r in range(R):
                num_u = np.random.randint(2, 4)
                num_v = np.random.randint(2, 4)
                num_w = np.random.randint(2, 4)
                
                u_idx = np.random.choice(n*m, size=num_u, replace=False)
                v_idx = np.random.choice(m*p, size=num_v, replace=False)
                w_idx = np.random.choice(p*n, size=num_w, replace=False)
                
                U[u_idx, r] = np.random.choice([-1, 0.5, 1], size=num_u)
                V[v_idx, r] = np.random.choice([-1, 0.5, 1], size=num_v)
                W[w_idx, r] = np.random.choice([-1, 0.5, 1], size=num_w)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            # Use L-BFGS-B with gradient
            res = minimize(objective_and_grad, params, method='L-BFGS-B', jac=True,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
                # If we found a very good solution, try to polish/round it
                if res.fun < 1e-4:
                    U_sol = res.x[0:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try rounding to various denominators
                    for denom in [2, 4, 8, 16]:
                        U_rounded = np.round(U_sol * denom) / denom
                        V_rounded = np.round(V_sol * denom) / denom
                        W_rounded = np.round(W_sol * denom) / denom
                        
                        # Verify exactness
                        vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                        if np.allclose(vals, 1.0, atol=1e-9):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
                    # High-precision polish with tighter tolerances
                    if time.time() - start_time < time_limit - 5:
                        res2 = minimize(objective_and_grad, res.x, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200})
                        
                        if res2.fun < 1e-6:
                            U_sol2 = res2.x[0:n*m*R].reshape(n*m, R)
                            V_sol2 = res2.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                            W_sol2 = res2.x[n*m*R + m*p*R:].reshape(p*n, R)
                            
                            # Final rounding attempt with more denominators
                            for denom in [2, 4, 8, 16, 32]:
                                U_rounded = np.round(U_sol2 * denom) / denom
                                V_rounded = np.round(V_sol2 * denom) / denom
                                W_rounded = np.round(W_sol2 * denom) / denom
                                
                                vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                                if np.allclose(vals, 1.0, atol=1e-9):
                                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                                    
        except Exception:
            continue
            
    return None

def valid_res(res, n, m, p):
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    R = U.shape[1]
    if V.shape[1] != R or W.shape[1] != R:
        return False
        
    indices_a = []
    indices_b = []
    indices_c = []
    for i in range(n):
        for k in range(m):
            for j in range(p):
                indices_a.append(i * m + k)
                indices_b.append(k * p + j)
                indices_c.append(j * n + i)
                
    indices_a = np.array(indices_a)
    indices_b = np.array(indices_b)
    indices_c = np.array(indices_c)
    
    vals = np.sum(U[indices_a, :] * V[indices_b, :] * W[indices_c, :], axis=1)
    return np.allclose(vals, 1.0, atol=1e-9)

def construct_r34():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
