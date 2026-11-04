import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 4, 6
    
    # Target: 3x4x3 sub-problem
    n_sub, m_sub, p_sub = 3, 4, 3
    
    def solve_sub(rank):
        # Variables: U (12 x rank), V (12 x rank), W (9 x rank)
        # Total vars: 33 * rank
        # Equations: 36
        
        def residuals(x):
            U = x[:12*rank].reshape(12, rank)
            V = x[12*rank:24*rank].reshape(12, rank)
            W = x[24*rank:].reshape(9, rank)
            
            res = np.zeros(36)
            idx = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        # Sum over r
                        u_val = U[i*m_sub + j, :]
                        v_val = V[j*p_sub + k, :]
                        w_val = W[k*n_sub + i, :]
                        res[idx] = np.sum(u_val * v_val * w_val) - 1.0
                        idx += 1
            return res
            
        # Random init
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            x0 = np.random.randn(33 * rank) * 0.5
            try:
                sol = least_squares(residuals, x0, method='lm')
                if np.linalg.norm(sol.fun) < 1e-6:
                    # Check if we can round
                    U = sol.x[:12*rank].reshape(12, rank)
                    V = sol.x[12*rank:24*rank].reshape(12, rank)
                    W = sol.x[24*rank:].reshape(9, rank)
                    
                    # Round to nearest 0.5
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    if np.allclose(residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])), 0, atol=1e-9):
                        return U_r, V_r, W_r
            except:
                pass
        return None

    # Try ranks
    for r in [19, 20, 21]:
        if time.time() > deadline: break
        res = solve_sub(r)
        if res is not None:
            U_sub, V_sub, W_sub = res
            # Construct 3x4x6
            rank_total = 2 * r
            U = np.zeros((12, rank_total))
            V = np.zeros((24, rank_total))
            W = np.zeros((18, rank_total))
            
            # First half (k=0..2)
            U[:, :r] = U_sub
            # V mapping: j*3+k -> j*6+k
            for j in range(4):
                for k in range(3):
                    V[j*6 + k, :r] = V_sub[j*3 + k, :]
            # W mapping: k*3+i -> k*3+i
            W[:, :r] = W_sub
            
            # Second half (k=3..5)
            U[:, r:] = U_sub
            # V mapping: j*3+k' -> j*6+(k'+3)
            for j in range(4):
                for k in range(3):
                    V[j*6 + k + 3, r:] = V_sub[j*3 + k, :]
            # W mapping: k'*3+i -> (k'+3)*3+i = k'*3 + 9 + i
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3 + i, r:] = W_sub[k*3 + i, :]
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook
    rank = 72
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(18)]
    col = 0
    for i in range(3):
        for j in range(4):
            for k in range(6):
                U[i*4+j][col] = 1
                V[j*6+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
