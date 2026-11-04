import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank
    # We try rank 41
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Heuristic search for rank 41
    # This is a non-linear least squares problem
    # Variables: U (6x41), V (21x41), W (14x41)
    # Total 41 * (6+21+14) = 41 * 41 = 1681 variables
    
    def residual(params, r):
        if time.time() > deadline:
            return 1e10
        U = params[:n*m*r].reshape(n*m, r)
        V = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        # Compute the tensor product and compare to target
        # Target T_{i,j,k} is 1 if it corresponds to A_{i,j} B_{j,k} contributing to C_{i,k}
        # Actually, we check the matrix multiplication identity.
        # C_{i,k} = sum_c (sum_j U[i*m+j, c] A[i,j]) (sum_j' V[j'*p+k, c] B[j',k]) W[k*n+i, c]
        # This must equal sum_j A[i,j] B[j,k]
        
        # We can test this by picking random A, B and checking C.
        # Or better, we can expand the expression and match coefficients.
        # The expression is a polynomial in A, B.
        # It is a sum of terms A_{i1, j1} B_{j2, k2} W_{k2*n+i1, c} U_{i1*m+j1, c} V_{j2*p+k2, c}
        # We want this to match sum_j A_{i,j} B_{j,k}.
        # So for each (i, k), the coefficient of A_{i,j} B_{j,k} must be 1 for j=j', and 0 otherwise?
        # No, the sum is over c.
        # For a fixed c, the term is (sum_j U_{i*m+j, c} A_{i,j}) (sum_j' V_{j'*p+k, c} B_{j',k}) W_{k*n+i, c}.
        # This generates terms A_{i,j} B_{j',k} with coeff U_{i*m+j, c} V_{j'*p+k, c} W_{k*n+i, c}.
        # We sum over c.
        # We want the total coefficient of A_{i,j} B_{j,k} to be 1 (for j=j').
        # And the coefficient of A_{i,j} B_{j',k} to be 0 for j != j'.
        
        # This gives a system of linear equations in the products U*V*W?
        # No, it's a sum of products.
        # We can set up a residual vector for all (i, j, j', k) indices.
        # Number of constraints: n * m * m * p = 2 * 3 * 3 * 7 = 126.
        
        res = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j_prime in range(m):
                        # Target coefficient: 1 if j == j_prime, else 0
                        target = 1.0 if j == j_prime else 0.0
                        # Actual coefficient:
                        actual = 0.0
                        for c in range(r):
                            actual += U[i * m + j, c] * V[j_prime * p + k, c] * W[k * n + i, c]
                        res += (actual - target) ** 2
        return res

    # Try rank 41
    if time.time() < deadline - 10:
        r = 41
        # Random initialization
        np.random.seed(42)
        best_res = float('inf')
        for _ in range(5): # 5 restarts
            if time.time() > deadline - 5:
                break
            U_init = np.random.randn(n*m, r)
            V_init = np.random.randn(m*p, r)
            W_init = np.random.randn(p*n, r)
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use scipy.optimize
            try:
                from scipy.optimize import minimize
                result = minimize(residual, params_init, args=(r,), method='BFGS', options={'maxiter': 50, 'disp': False})
                if result.fun < best_res:
                    best_res = result.fun
                    if best_res < 1e-8:
                        # Found a solution
                        U_sol = result.x[:n*m*r].reshape(n*m, r)
                        V_sol = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                        W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                        
                        # Check if we can round to half-integers
                        # The prompt asks for integer/half-integer entries.
                        # If the solution is exact, the values should be close to such.
                        # Let's try to round to nearest 0.5
                        U_round = np.round(U_sol * 2) / 2
                        V_round = np.round(V_sol * 2) / 2
                        W_round = np.round(W_sol * 2) / 2
                        
                        # Verify the rounded solution
                        res_round = residual(np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()]), r)
                        if res_round < 1e-6:
                            best_rank = r
                            best_U = U_round.tolist()
                            best_V = V_round.tolist()
                            best_W = W_round.tolist()
                            break
            except Exception:
                pass

    return {"U": best_U, "V": best_V, "W": best_W}
