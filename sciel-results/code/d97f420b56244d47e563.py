import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target ranks to try, from low to high
    # Schoolbook is 36.
    # Let's try to find a rank lower than 36.
    # Known results might suggest lower ranks.
    # We will try r = 24 to 35.
    
    # Time limit guard
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def solve_for_rank(r, max_restarts=20):
        num_vars = r * (n*m + m*p + p*n)
        num_eqs = len(indices)
        
        # We can use a simple gradient descent or least squares.
        # Since the problem is small, we can use scipy.optimize.least_squares.
        # But we need to handle the non-convexity.
        
        best_solution = None
        best_residual = float('inf')
        
        for restart in range(max_restarts):
            # Random initialization
            # Use small random values
            np.random.seed(42 + restart)
            U_flat = np.random.uniform(-1, 1, r * n * m)
            V_flat = np.random.uniform(-1, 1, r * m * p)
            W_flat = np.random.uniform(-1, 1, r * p * n)
            
            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            def residuals(x):
                U = x[:r*n*m].reshape(r, n, m)
                V = x[r*n*m : r*n*m + r*m*p].reshape(r, m, p)
                W = x[r*n*m + r*m*p :].reshape(r, p, n)
                
                # Compute tensor product
                # T[i,j,k] = sum_c U[c,i,j] V[c,j,k] W[c,k,i]
                # We need this to be 1 for all i,j,k.
                # Let's compute the error.
                res = np.zeros((n, m, p))
                for c in range(r):
                    res += U[c, :, :][:, :, None] * V[c, :, :][None, :, :] * W[c, :, :][None, None, :]
                    # Wait, U is [r, n, m], V is [r, m, p], W is [r, p, n]
                    # U[c, i, j] * V[c, j, k] * W[c, k, i]
                    # Let's use einsum
                # Actually, einsum is faster
                T = np.einsum('cij,cjk,cki->ijk', U, V, W)
                # T[i,j,k] should be 1
                return (T - 1).flatten()
            
            # Use least_squares
            try:
                res = residuals(x0)
                if np.linalg.norm(res) < best_residual:
                    best_residual = np.linalg.norm(res)
                    best_solution = x0.copy()
                
                # If initial guess is already good, skip optimization?
                # Otherwise, run optimization
                # We can use a simple gradient step or scipy
                # Since time is limited, we'll use a few iterations of a simple solver
                # or just check if random works (unlikely).
                
                # Let's try a few steps of gradient descent
                lr = 0.01
                for step in range(50):
                    if time.time() > deadline:
                        return None, float('inf')
                    grad = np.zeros_like(x0)
                    # Numerical gradient or analytical?
                    # Analytical gradient is better.
                    # T[i,j,k] = sum_c U[c,i,j] V[c,j,k] W[c,k,i]
                    # dT/dU[c,i,j] = V[c,j,k] W[c,k,i]
                    # dT/dV[c,j,k] = U[c,i,j] W[c,k,i]
                    # dT/dW[c,k,i] = U[c,i,j] V[c,j,k]
                    
                    # This is complex to vectorize easily without loops.
                    # Given small size, loops are fine.
                    
                    dU = np.zeros_like(U)
                    dV = np.zeros_like(V)
                    dW = np.zeros_like(W)
                    
                    # T is n x m x p
                    # Error E = T - 1
                    E = T - 1
                    
                    for c in range(r):
                        # dU[c,i,j] = sum_k V[c,j,k] W[c,k,i] E[i,j,k]
                        # dV[c,j,k] = sum_i U[c,i,j] W[c,k,i] E[i,j,k]
                        # dW[c,k,i] = sum_j U[c,i,j] V[c,j,k] E[i,j,k]
                        
                        # We can use einsum for these too
                        dU[c] = np.einsum('jk,ki,ijk->ij', V[c], W[c], E)
                        dV[c] = np.einsum('ij,ki,ijk->jk', U[c], W[c], E)
                        dW[c] = np.einsum('ij,jk,ijk->ki', U[c], V[c], E)
                    
                    grad[:r*n*m] = dU.flatten()
                    grad[r*n*m : r*n*m + r*m*p] = dV.flatten()
                    grad[r*n*m + r*m*p :] = dW.flatten()
                    
                    x0 -= lr * grad
                    
                    # Update U, V, W
                    U = x0[:r*n*m].reshape(r, n, m)
                    V = x0[r*n*m : r*n*m + r*m*p].reshape(r, m, p)
                    W = x0[r*n*m + r*m*p :].reshape(r, p, n)
                    T = np.einsum('cij,cjk,cki->ijk', U, V, W)
                    res = (T - 1).flatten()
                    norm_res = np.linalg.norm(res)
                    
                    if norm_res < best_residual:
                        best_residual = norm_res
                        best_solution = x0.copy()
                        
                    if norm_res < 1e-9:
                        return x0, norm_res
                        
            except Exception as e:
                continue
                
        return best_solution, best_residual

    # Try ranks
    for r in range(20, 36):
        if time.time() > deadline:
            break
        sol, res = solve_for_rank(r, max_restarts=10)
        if sol is not None and res < 1e-8:
            # Found a solution!
            # Round to half-integers
            U = sol[:r*n*m].reshape(r, n, m)
            V = sol[r*n*m : r*n*m + r*m*p].reshape(r, m, p)
            W = sol[r*n*m + r*m*p :].reshape(r, p, n)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check if rounded solution is exact
            T_rounded = np.einsum('cij,cjk,cki->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_rounded, 1, atol=1e-9):
                # Convert to lists
                U_list = [[0]*r for _ in range(n*m)]
                V_list = [[0]*r for _ in range(m*p)]
                W_list = [[0]*r for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        for c in range(r):
                            U_list[i*m+j][c] = float(U_rounded[c, i, j])
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            V_list[j*p+k][c] = float(V_rounded[c, j, k])
                for k in range(p):
                    for i in range(n):
                        for c in range(r):
                            W_list[k*n+i][c] = float(W_rounded[c, k, i])
                
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no better rank found, return schoolbook
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
