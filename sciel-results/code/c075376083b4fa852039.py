import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p  # 24
    
    # Target tensor T[i, j, k] = 1
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    def verify(U, V, W, R):
        """
        Verify if U, V, W provide an exact decomposition.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # Compute approximation
        # Sum over c
        # A_approx[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
        # Note: W3 index is [k, i, c] because W is (p*n, R) and index is k*n+i.
        # So W3[k, i, c] corresponds to W[k*n+i, c].
        
        # Efficient einsum
        # U3: i, j, c
        # V3: j, k, c
        # W3: k, i, c
        # Result: i, j, k
        approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Check if all entries are 1
        return np.allclose(approx, 1.0, atol=1e-9)

    def try_rank(R, max_restarts=20):
        """
        Attempt to find a rank R decomposition.
        """
        num_vars = R * (n * m + m * p + p * n)
        
        def unpack(params):
            U_flat = params[:R * n * m]
            V_flat = params[R * n * m : R * (n * m + m * p)]
            W_flat = params[R * (n * m + m * p) :]
            
            U = U_flat.reshape(R, n * m).T  # (n*m, R)
            V = V_flat.reshape(R, m * p).T  # (m*p, R)
            W = W_flat.reshape(R, p * n).T  # (p*n, R)
            return U, V, W

        def objective(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            
            approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            error = approx - 1.0
            return np.sum(error * error)

        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Random initialization
            # We use small integers to help convergence towards integer solutions
            # Scale factor helps avoid zero gradients
            init_params = np.random.uniform(-2, 2, size=num_vars)
            
            # Use Nelder-Mead as it doesn't require gradients and is robust
            # We limit the number of function evaluations to prevent long runs
            # maxiter is not the best limit, max_nfev is better for minimize
            try:
                res = minimize(objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12, 'disp': False})
                
                if res.fun < 1e-8:
                    # Candidate found, try to round to half-integers
                    U, V, W = unpack(res.x)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U * 2) / 2.0
                    V_rounded = np.round(V * 2) / 2.0
                    W_rounded = np.round(W * 2) / 2.0
                    
                    # Check if rounded version is exact
                    if verify(U_rounded, V_rounded, W_rounded, R):
                        # Return as list of lists
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    
                    # If rounding failed, maybe the solution is integer?
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify(U_int, V_int, W_int, R):
                        return {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        
            except Exception as e:
                pass
                
        return None

    # Search for lower ranks
    # Start from 23 down to 18
    # We prioritize higher ranks first as they are more likely to exist
    for R in range(23, 17, -1):
        if time.time() - start_time > max_time - 5:
            break
            
        # Increase restarts for lower R? No, lower R is harder.
        # But we have limited time.
        # Let's do 30 restarts for R=23, 20 for R=22, etc.
        restarts = 30 if R >= 22 else 20
        
        result = try_rank(R, max_restarts=restarts)
        if result:
            return result

    # Fallback: Schoolbook decomposition
    rank = target_rank
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
