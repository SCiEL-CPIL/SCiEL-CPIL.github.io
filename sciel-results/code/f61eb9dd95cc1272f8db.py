import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor values: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # We will try ranks starting from a lower bound
    # Lower bound is max(n+m-1, m+p-1, n+p-1) = 7
    # But for 2x3x5, we suspect 13 or 14.
    # Let's try a range of ranks.
    
    def get_loss(params, R):
        # Unpack params into U, V, W
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U = params[0:size_u].reshape(n * m, R)
        V = params[size_u:size_u + size_v].reshape(m * p, R)
        W = params[size_u + size_v:size_u + size_v + size_w].reshape(p * n, R)
        
        residual = 0.0
        # Calculate sum for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual += (1.0 - val) ** 2
        return residual

    def get_matrices(params, R):
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U = params[0:size_u].reshape(n * m, R)
        V = params[size_u:size_u + size_v].reshape(m * p, R)
        W = params[size_u + size_v:size_u + size_v + size_w].reshape(p * n, R)
        
        return U, V, W

    # Try ranks from 13 to 20
    for R in range(13, 21):
        # Initialize with random small values
        # Multiple restarts might be needed
        found = False
        for attempt in range(5):
            np.random.seed(attempt * 100 + R)
            # Random initialization
            init_u = np.random.rand(n * m, R) * 0.5
            init_v = np.random.rand(m * p, R) * 0.5
            init_w = np.random.rand(p * n, R) * 0.5
            
            params = np.concatenate([init_u.flatten(), init_v.flatten(), init_w.flatten()])
            
            # Use Nelder-Mead or L-BFGS-B
            # L-BFGS-B is better for larger R
            res = minimize(get_loss, params, args=(R,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.fun < 1e-8:
                # Check if we can snap to half-integers
                U, V, W = get_matrices(res.x, R)
                
                # Snap to nearest 0.5
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped matrices
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U_snap[i * m + j, r] * V_snap[j * p + k, r] * W_snap[k * n + i, r]
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Convert to lists of lists
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                else:
                    # If snapping fails, maybe the float solution is acceptable? 
                    # The task says "integer/half-integer entries". 
                    # If we can't snap, we might need to try harder or accept a higher rank.
                    # But usually, optimal decompositions have simple entries.
                    pass
        
    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
