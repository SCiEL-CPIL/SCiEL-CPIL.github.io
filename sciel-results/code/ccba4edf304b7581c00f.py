"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least max(n*p, m*p, n*m) is not rigorous, but let's try
    # Actually, a better lower bound: the tensor has support of size n*m*p=42
    # Each rank-1 term can cover at most... it's complex
    
    # Let's try ranks from 21 to 42
    best_rank = n * m * p  # schoolbook
    best_solution = None
    
    deadline = time.time() + 140  # leave 10s margin
    
    for r in range(21, n * m * p + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + r * 100 + trial)
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Define the objective: sum of squared (T_decomposed - 1)^2
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j, :], V_flat[j*p+k, :] * W_flat[k*n+i, :])
                            error += (val - 1.0) ** 2
                return error
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            result = minimize(objective, params, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.fun < 1e-8:
                # Found a valid decomposition
                U_sol = result.x[:n*m*r].reshape(n*m, r)
                V_sol = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Try to round to rationals
                # Scale to get nice numbers
                best_rank = r
                best_solution = (U_sol, V_sol, W_sol)
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fall back to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        U_sol, V_sol, W_sol = U, V, W
        best_rank = r
    else:
        U_sol, V_sol, W_sol = best_solution
        r = best_rank
    
    # Convert to lists with rounding
    def to_list(mat):
        # Round to nearest 0.5 or try to find rational
        lst = []
        for row in mat:
            lrow = []
            for val in row:
                # Try rounding to nearest integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_int) < abs(val - nearest_half):
                    lrow.append(nearest_int)
                else:
                    lrow.append(nearest_half)
            lst.append(lrow)
        return lst
    
    U_list = to_list(U_sol)
    V_list = to_list(V_sol)
    W_list = to_list(W_sol)
    
    # Verify the rounded solution
    def verify(U_l, V_l, W_l, n, m, p):
        r = len(U_l[0]) if U_l else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_l[i*m+j][c] * V_l[j*p+k][c] * W_l[k*n+i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    if not verify(U_list, V_list, W_list, n, m, p):
        # Rounding failed, fall back to schoolbook
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
