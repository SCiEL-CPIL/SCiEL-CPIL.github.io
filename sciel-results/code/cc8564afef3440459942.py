import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 4
    # T[a, b, g] where a=i*m+j, b=j*p+k, g=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                g = k*n + i
                T[a, b, g] = 1.0

    start_time = time.time()
    timeout = 115.0  # Leave 5s margin for safety

    # Try ranks from a reasonable lower bound up to schoolbook rank 36
    # Lower bound for 3x3x3 is 23. For 3x3x4, it might be similar or slightly higher/lower.
    # Let's start from 20 to be safe, but 23 is a good start.
    for r in range(20, 37):
        if time.time() - start_time > timeout:
            break
            
        # More attempts for lower ranks as they are harder to find
        max_attempts = 3000 if r < 25 else 1000 if r < 30 else 500
        
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
                
            # Random U, V with entries -1, 0, 1
            # Using a sparse distribution might help? Or dense?
            # Let's try dense first.
            U = np.random.choice([-1, 0, 1], size=(n*m, r))
            V = np.random.choice([-1, 0, 1], size=(m*p, r))
            
            # Solve for W
            # Equation: sum_c U[a,c] V[b,c] W[g,c] = T[a,b,g]
            # For each g, we have a linear system for W[g,:].
            # M_g[(a,b), c] = U[a,c] * V[b,c]
            # b_g[(a,b)] = T[a,b,g]
            
            W = np.zeros((p*n, r), dtype=np.float64)
            valid = True
            max_residual = 0.0
            
            # Precompute U*V outer products for each c?
            # M_g is 108 x r.
            # We can construct M_g efficiently.
            
            for g in range(p*n):
                # Construct M_g
                # Rows correspond to (a,b) pairs. There are n*m*m*p = 9*12 = 108 pairs.
                # But T is sparse. We only need to satisfy equations where T != 0?
                # No, we must satisfy ALL equations. T=0 must also be 0.
                # So we need the full system.
                
                # Vectorize the construction of M_g
                # U is 9x r, V is 12x r
                # M_g[row, c] = U[a, c] * V[b, c]
                # We can use broadcasting.
                # U_col = U.T  # r x 9
                # V_col = V.T  # r x 12
                # Product = U_col[:, :, None] * V_col[:, None, :] # r x 9 x 12
                # Reshape to 108 x r
                # This is done for each g? No, M_g depends only on U and V, not g.
                # Wait, M_g is the same for all g!
                # The RHS b_g changes.
                # So we can precompute M once!
                
                if g == 0:
                    # Precompute M (108 x r)
                    # M[idx, c] = U[a, c] * V[b, c]
                    # idx = a * (m*p) + b
                    M = np.zeros((n*m * m*p, r), dtype=np.float64)
                    for a in range(n*m):
                        for b in range(m*p):
                            M[a*(m*p)+b, :] = U[a, :] * V[b, :]
                
                b_g = T[:, :, g].flatten()
                
                # Solve M w_g = b_g
                try:
                    w_g, residuals, rank, sv = np.linalg.lstsq(M, b_g, rcond=None)
                except:
                    valid = False
                    break
                    
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        valid = False
                        break
                else:
                    # Check residual manually if lstsq doesn't provide it (shouldn't happen for overdetermined)
                    res = np.linalg.norm(M @ w_g - b_g)
                    if res > 1e-9:
                        valid = False
                        break
                        
                W[g, :] = w_g
                if residuals is not None and len(residuals) > 0:
                    max_residual = max(max_residual, residuals[0])
                    
            if not valid:
                continue
                
            # If we got here, we have a real-valued W that satisfies the equations.
            # We need integer or half-integer entries.
            # Try snapping to integers first.
            W_int = np.round(W)
            T_approx = np.einsum('ac,bc,gc->abg', U, V, W_int)
            if np.allclose(T_approx, T, atol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_int.tolist()}
            
            # Try snapping to half-integers
            W_half = np.round(W * 2) / 2
            T_approx = np.einsum('ac,bc,gc->abg', U, V, W_half)
            if np.allclose(T_approx, T, atol=1e-9):
                return {"U": U.tolist(), "V": V.tolist(), "W": W_half.tolist()}
                
            # Sometimes U and V need to be snapped too?
            # The problem statement says "integer/half-integer entries".
            # Usually, if we solve for W, U and V are fixed.
            # But we chose U, V randomly from {-1,0,1}, so they are already integers.
            # So only W needs snapping.
            
            # If W is close to integers but not quite, maybe we can adjust U, V?
            # That's a harder optimization problem.
            # For now, just try more random samples.

    # Fallback to schoolbook rank 36
    rank = 36
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
