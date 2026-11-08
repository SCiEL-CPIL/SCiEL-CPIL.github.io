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
    return construct_r34()

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
    
    # Objective function with gradient
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
        
        # Compute gradients
        dU = np.zeros((n*m, R))
        dV = np.zeros((m*p, R))
        dW = np.zeros((p*n, R))
        
        for idx in range(len(indices_a)):
            a = indices_a[idx]
            b = indices_b[idx]
            c = indices_c[idx]
            dv = 2 * (vals[idx] - 1.0)
            
            dU[a, :] += dv * V[b, :] * W[c, :]
            dV[b, :] += dv * U[a, :] * W[c, :]
            dW[c, :] += dv * U[a, :] * V[b, :]
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return error, grad
    
    from scipy.optimize import minimize
    
    best_error = float('inf')
    best_params = None
    
    # Strategy: Use structured initialization exploiting the block structure
    # The rank-34 solution has two blocks of 17 columns each
    # Try to find a lower-rank solution with similar structure
    
    num_restarts = min(8, max(3, int(time_limit / 3.0)))
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit - 3:
            break
        
        np.random.seed(1234 + restart * 7919)
        
        # Create initial guess with structure
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        if restart == 0:
            # Structured: each rank-1 term targets specific (i,k,j) patterns
            for r in range(R):
                # Cycle through different patterns
                pattern = r % 6
                
                if pattern == 0:
                    # Pure diagonal-like
                    i = r // 6 % n
                    k = (r // 6) % m
                    j = (r // 6) % p
                    U[i*m+k, r] = 1.0
                    V[k*p+j, r] = 1.0
                    W[j*n+i, r] = 1.0
                elif pattern == 1:
                    # Two-term in U
                    i1 = r // 6 % n
                    i2 = (i1 + 1) % n
                    k = (r // 6) % m
                    j = (r // 6) % p
                    U[i1*m+k, r] = 1.0
                    U[i2*m+k, r] = 1.0
                    V[k*p+j, r] = 1.0
                    W[j*n+i1, r] = 1.0
                    W[j*n+i2, r] = -1.0
                elif pattern == 2:
                    # Two-term in V
                    i = r // 6 % n
                    k = (r // 6) % m
                    j1 = (r // 6) % p
                    j2 = (j1 + 1) % p
                    U[i*m+k, r] = 1.0
                    V[k*p+j1, r] = 1.0
                    V[k*p+j2, r] = -1.0
                    W[j1*n+i, r] = 1.0
                    W[j2*n+i, r] = 1.0
                elif pattern == 3:
                    # Two-term in W
                    i = r // 6 % n
                    k = (r // 6) % m
                    j = (r // 6) % p
                    U[i*m+k, r] = 1.0
                    V[k*p+j, r] = 1.0
                    W[j*n+i, r] = 1.0
                    W[j*n+(i+1)%n, r] = 1.0
                elif pattern == 4:
                    # Three-term sparse
                    i = r // 6 % n
                    k = (r // 6) % m
                    j = (r // 6) % p
                    U[i*m+k, r] = 1.0
                    U[(i+1)%n*m+k, r] = 0.5
                    V[k*p+j, r] = 1.0
                    W[j*n+i, r] = 1.0
                    W[j*n+(i+1)%n, r] = -0.5
                else:
                    # Random sparse
                    u_idx = np.random.choice(n*m, size=2, replace=False)
                    v_idx = np.random.choice(m*p, size=2, replace=False)
                    w_idx = np.random.choice(p*n, size=2, replace=False)
                    U[u_idx, r] = np.random.choice([-1, 1], size=2)
                    V[v_idx, r] = np.random.choice([-1, 1], size=2)
                    W[w_idx, r] = np.random.choice([-1, 1], size=2)
        
        elif restart == 1:
            # Dense random with small scale
            U = np.random.randn(n*m, R) * 0.3
            V = np.random.randn(m*p, R) * 0.3
            W = np.random.randn(p*n, R) * 0.3
        
        elif restart == 2:
            # Block-structured: first half for j=0,1, second half for j=2,3
            half = R // 2
            for r in range(half):
                i = r % n
                k = (r // n) % m
                j = (r // (n*m)) % 2
                U[i*m+k, r] = 1.0
                V[k*p+j, r] = 1.0
                W[j*n+i, r] = 1.0
                # Add cross terms
                if r < 9:
                    U[(i+1)%n*m+k, r] = 0.5
                    W[j*n+(i+1)%n, r] = -0.5
            
            for r in range(half, R):
                r2 = r - half
                i = r2 % n
                k = (r2 // n) % m
                j = 2 + (r2 // (n*m)) % 2
                U[i*m+k, r] = 1.0
                V[k*p+j, r] = 1.0
                W[j*n+i, r] = 1.0
                if r2 < 9:
                    U[(i+1)%n*m+k, r] = 0.5
                    W[j*n+(i+1)%n, r] = -0.5
        
        elif restart == 3:
            # Use base solution structure, compressed
            base = construct_r34()
            U_base = np.array(base['U'])
            V_base = np.array(base['V'])
            W_base = np.array(base['W'])
            
            # Try to project 34 columns into R
            for r in range(R):
                # Pick 2-3 base columns
                num_base = min(3, 34 - r)
                base_cols = np.random.choice(34, size=num_base, replace=False)
                weights = np.random.randn(num_base)
                U[:, r] = U_base[:, base_cols] @ weights
                V[:, r] = V_base[:, base_cols] @ weights
                W[:, r] = W_base[:, base_cols] @ weights
        
        else:
            # Random with structure hints
            for r in range(R):
                # Sparse with 2-3 nonzeros per factor
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
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 100})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
                # If we found a very good solution, try to polish/round it
                if res.fun < 1e-5:
                    U_sol = res.x[0:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Try rounding to nearest 0.25
                    for denom in [2, 4, 8]:
                        U_rounded = np.round(U_sol * denom) / denom
                        V_rounded = np.round(V_sol * denom) / denom
                        W_rounded = np.round(W_sol * denom) / denom
                        
                        # Verify exactness
                        vals = np.sum(U_rounded[indices_a, :] * V_rounded[indices_b, :] * W_rounded[indices_c, :], axis=1)
                        if np.allclose(vals, 1.0, atol=1e-9):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
                    # High-precision polish
                    if time.time() - start_time < time_limit - 5:
                        res2 = minimize(objective_and_grad, res.x, method='L-BFGS-B', jac=True,
                                       options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-13})
                        
                        if res2.fun < 1e-7:
                            U_sol2 = res2.x[0:n*m*R].reshape(n*m, R)
                            V_sol2 = res2.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                            W_sol2 = res2.x[n*m*R + m*p*R:].reshape(p*n, R)
                            
                            # Final rounding attempt
                            for denom in [2, 4, 8]:
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
