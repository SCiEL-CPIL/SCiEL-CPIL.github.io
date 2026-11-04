"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Schoolbook rank = n*m*p = 30
    rank_schoolbook = n * m * p

    # Construct the schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    # Build the list of (a, b, c, target) constraints
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # target = 1 if i==i' and j==j' and k==k', else 0
    constraints_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            target = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            constraints_list.append((a, b, c, target))

    def residual(params, r):
        """Compute residual sum of squares for a given rank-r decomposition."""
        # params is a flat array of length r*(n*m + m*p + p*n)
        total_size = r * (n * m + m * p + p * n)
        if len(params) != total_size:
            return 1e10
        
        U_flat = params[:r * n * m].reshape(n * m, r)
        V_flat = params[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
        W_flat = params[r * n * m + r * m * p :].reshape(p * n, r)
        
        res = 0.0
        for (a, b, c, target) in constraints_list:
            val = np.dot(U_flat[a], V_flat[b] * W_flat[c])
            res += (val - target) ** 2
        return res

    def try_rank(r, max_iters=500, timeout=30):
        """Try to find a rank-r decomposition."""
        nonlocal best_rank, best_U, best_V, best_W
        
        if r >= best_rank:
            return False
        
        if time.time() > deadline - 10:
            return False
        
        total_size = r * (n * m + m * p + p * n)
        
        # Try multiple random initializations
        best_res = np.inf
        best_params = None
        
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random values
            np.random.seed(42 + attempt)
            x0 = np.random.randn(total_size) * 0.5
            
            # Use L-BFGS-B
            result = minimize(residual, x0, args=(r,), method='L-BFGS-B',
                            options={'maxiter': max_iters, 'maxfun': 5000,
                                    'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if result.fun < best_res:
                best_res = result.fun
                best_params = result.x
        
        if best_params is not None and best_res < 1e-10:
            # Found a valid solution
            U_flat = best_params[:r * n * m].reshape(n * m, r)
            V_flat = best_params[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
            W_flat = best_params[r * n * m + r * m * p :].reshape(p * n, r)
            
            # Convert to lists with rounded values
            U_list = [[float(x) for x in row] for row in U_flat]
            V_list = [[float(x) for x in row] for row in V_flat]
            W_list = [[float(x) for x in row] for row in W_flat]
            
            # Verify
            for (a, b, c, target) in constraints_list:
                val = np.dot(U_flat[a], V_flat[b] * W_flat[c])
                if abs(val - target) > 1e-9:
                    return False
            
            best_rank = r
            best_U, best_V, best_W = U_list, V_list, W_list
            return True
        
        return False

    # Try to reduce rank from 29 down to the lower bound
    for r in range(rank_schoolbook - 1, max(n*m, m*p, p*n) - 1, -1):
        if try_rank(r):
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
