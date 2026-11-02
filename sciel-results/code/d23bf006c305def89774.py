"""Program for matrix multiplication tensor decomposition (2x2 by 2x10)."""
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a (2x2) matrix by a (2x10) matrix.
    
    Uses direct optimization to find a low-rank decomposition.
    Falls back to Strassen (rank 35) if optimization fails.
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # U: (n*m, R) = (4, R)
    # V: (m*p, R) = (20, R)
    # W: (p*n, R) = (20, R)
    
    # Build the target tensor T
    # T[u, v, w] = 1 if (u,v,w) corresponds to a valid multiplication term
    # u = i*2 + j, v = j*10 + k, w = k*2 + i
    T = np.zeros((4, 20, 20))
    for i in range(2):
        for j in range(2):
            for k in range(10):
                u = i * 2 + j
                v = j * 10 + k
                w = k * 2 + i
                T[u, v, w] = 1.0

    def snap(x):
        return np.round(x * 2) / 2.0

    def verify(U, V, W):
        T_calc = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_calc, T, atol=1e-5)

    # Try to find a decomposition for decreasing ranks
    target_ranks = [25, 26, 28, 30, 32, 34]
    
    for R in target_ranks:
        # Try a few random initializations
        for attempt in range(3):
            # Initialize U, V, W randomly
            U = np.random.randn(4, R)
            V = np.random.randn(20, R)
            W = np.random.randn(20, R)
            
            def objective(params):
                U_flat = params[:4*R].reshape(4, R)
                V_flat = params[4*R:4*R+20*R].reshape(20, R)
                W_flat = params[4*R+20*R:].reshape(20, R)
                T_calc = np.einsum('ur,vr,wr->uvw', U_flat, V_flat, W_flat)
                diff = T_calc - T
                return np.sum(diff * diff)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = opt.minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:4*R].reshape(4, R)
                    V_opt = res.x[4*R:4*R+20*R].reshape(20, R)
                    W_opt = res.x[4*R+20*R:].reshape(20, R)
                    
                    U_snap = snap(U_opt)
                    V_snap = snap(V_opt)
                    W_snap = snap(W_opt)
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                    else:
                        # If snapping fails, check original floats (might be invalid for strict integer check)
                        # But we return floats if they are close enough? 
                        # The contract says "integer/half-integer". 
                        # We must return snapped values.
                        pass

            except Exception:
                continue

    # Fallback to Strassen (Rank 35)
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_2x2 = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    
    V_2x2 = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    
    W_2x2 = [
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ]
    
    col = 0
    for t in range(num_blocks):
        for c in range(7):
            for i in range(4):
                U[i][col] = U_2x2[c][i]
                
            v_vals = V_2x2[c]
            V[2*t][col] = v_vals[0]
            V[2*t + 1][col] = v_vals[1]
            V[10 + 2*t][col] = v_vals[2]
            V[11 + 2*t][col] = v_vals[3]
            
            w_vals = W_2x2[c]
            W[4*t][col] = w_vals[0]
            W[4*t + 1][col] = w_vals[1]
            W[4*t + 2][col] = w_vals[2]
            W[4*t + 3][col] = w_vals[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
