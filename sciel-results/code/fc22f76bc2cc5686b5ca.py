import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 4, 6
    
    # Build the target tensor
    # A: n x m = 2 x 4, index a_idx = i*m + j = i*4 + j
    # B: m x p = 4 x 6, index b_idx = j*p + k = j*6 + k
    # C: n x p = 2 x 6, index c_idx = i*p + k = i*6 + k
    # W convention: W[k*n + i, r] = W[k*2 + i, r]
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                w_idx = k * n + i
                T[a_idx, b_idx, w_idx] = 1.0
    
    def verify(U, V, W, atol=1e-8):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=atol)
    
    def try_rank(rank, num_restarts=5, time_per_restart=15.0):
        """Try to find a rank-`rank` decomposition."""
        total_params = (n*m + m*p + p*n) * rank
        time_per = min(time_per_restart, (deadline - time.time()) / max(1, num_restarts))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize
            np.random.seed(restart * 1000 + rank)
            scale = 0.5
            U0 = np.random.randn(n*m, rank) * scale
            V0 = np.random.randn(m*p, rank) * scale
            W0 = np.random.randn(p*n, rank) * scale
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def unpack(params):
                U = params[:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                return U, V, W
            
            def loss(params):
                U, V, W = unpack(params)
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((T_recon - T)**2)
            
            # Optimize
            res = minimize(loss, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
            
            if res.fun > 1e-6:
                continue
            
            U, V, W = unpack(res.x)
            
            # Polish
            res2 = minimize(loss, res.x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-10})
            if res2.fun < res.fun:
                U, V, W = unpack(res2.x)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if verify(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
            
            # Try integer snap
            U_snap_i = np.round(U)
            V_snap_i = np.round(V)
            W_snap_i = np.round(W)
            if verify(U_snap_i, V_snap_i, W_snap_i):
                return U_snap_i, V_snap_i, W_snap_i
            
            # Try without snap if very accurate
            if verify(U, V, W, atol=1e-10):
                return U, V, W
        
        return None
    
    # Try decreasing ranks
    for rank in [36, 35, 34, 33, 32, 31, 30, 28, 24]:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank, num_restarts=3, time_per_restart=20.0)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank 42 solution
    def get_rank_42_solution():
        rank_total = 42
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        strassen_U = [
            {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
            {0: 1, 1: 1}, {0: -1, 2: 1}, {1: 1, 3: -1}
        ]
        strassen_V = [
            {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
            {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
        ]
        strassen_W = [
            {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
            {0: -1, 1: 1}, {3: 1}, {0: 1}
        ]

        col_offset = 0
        for j_block in range(3):
            for side in ['L', 'R']:
                if side == 'L':
                    u_col_offset = 0
                    v_row_offset = 0
                else:
                    u_col_offset = 2
                    v_row_offset = 2

                for prod_idx in range(7):
                    col = col_offset + prod_idx

                    for local_idx, sign in strassen_U[prod_idx].items():
                        if local_idx == 0: u_idx = u_col_offset
                        elif local_idx == 1: u_idx = u_col_offset + 1
                        elif local_idx == 2: u_idx = 4 + u_col_offset
                        else: u_idx = 4 + u_col_offset + 1
                        U[u_idx][col] += sign

                    for local_idx, sign in strassen_V[prod_idx].items():
                        if local_idx == 0: v_idx = v_row_offset * 6 + 2 * j_block
                        elif local_idx == 1: v_idx = v_row_offset * 6 + 2 * j_block + 1
                        elif local_idx == 2: v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                        else: v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                        V[v_idx][col] += sign

                    for local_idx, sign in strassen_W[prod_idx].items():
                        if local_idx == 0: w_idx = 2 * (2 * j_block)
                        elif local_idx == 1: w_idx = 2 * (2 * j_block + 1)
                        elif local_idx == 2: w_idx = 2 * (2 * j_block) + 1
                        else: w_idx = 2 * (2 * j_block + 1) + 1
                        W[w_idx][col] += sign

                col_offset += 7

        return {"U": U, "V": V, "W": W}
    
    return get_rank_42_solution()
