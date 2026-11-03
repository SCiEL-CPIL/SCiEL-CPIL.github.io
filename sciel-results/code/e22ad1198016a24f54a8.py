import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        # Re-construct the 17-rank logic for one block
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms)
        U_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: 1},       
            {0: 1},             
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 0: -1},      
            {1: 1, 3: -1}       
        ]
        V_coeffs = [
            {0: 1, 3: 1},       
            {0: 1},             
            {1: 1, 3: -1},      
            {2: 1, 0: -1},      
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 3: 1}        
        ]
        W_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: -1},      
            {1: 1, 3: 1},       
            {0: 1, 2: 1},       
            {0: -1, 1: 1},      
            {3: 1},             
            {0: 1}              
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part (10 terms)
        # i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2, k in {0,1}
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2."""
        U17, V17, W17 = get_17_rank_block()
        
        # Initialize with 16 terms (drop last one)
        U_init = U17[:, :16].flatten()
        V_init = V17[:, :16].flatten()
        W_init = W17[:, :16].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        def residuals(x):
            U = x[:144].reshape(9, 16)
            V = x[144:240].reshape(6, 16)
            W = x[240:336].reshape(6, 16)
            
            # Compute T
            T = np.zeros((3, 3, 2))
            for c in range(16):
                u = U[:, c].reshape(3, 3)
                v = V[:, c].reshape(3, 2)
                w = W[:, c].reshape(2, 3)
                # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                for i in range(3):
                    for j in range(3):
                        if abs(u[i,j]) < 1e-15: continue
                        for k in range(2):
                            T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            
            # Expected T is all 1s
            return (T - 1.0).flatten()
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                x_opt = result.x
                U_opt = x_opt[:144].reshape(9, 16)
                V_opt = x_opt[144:240].reshape(6, 16)
                W_opt = x_opt[240:336].reshape(6, 16)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Check if rounded solution works
                T_check = np.zeros((3, 3, 2))
                for c in range(16):
                    u = U_rounded[:, c].reshape(3, 3)
                    v = V_rounded[:, c].reshape(3, 2)
                    w = W_rounded[:, c].reshape(2, 3)
                    for i in range(3):
                        for j in range(3):
                            if abs(u[i,j]) < 1e-15: continue
                            for k in range(2):
                                T_check[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                
                if np.max(np.abs(T_check - 1.0)) < 1e-10:
                    return U_rounded, V_rounded, W_rounded
        except Exception:
            pass
        return None

    # Try to find rank 16 block
    block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        
    # Assemble full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((6, rank_total))
    W = np.zeros((6, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # Copy U block
        U[:, col_start:col_end] = U_block
        
        # Copy V block, adjusting k indices
        # V_block is 6x16 (or 17). Indices: j*2+k
        # We need to map k_local (0,1) to k_global (k0, k1)
        V_g = np.zeros((6, rank_block))
        for j in range(3):
            for k_local in range(2):
                k_global = k0 + k_local
                V_g[j*2 + k_local, :] = V_block[j*2 + k_local, :]
                # Wait, V_block is indexed by j*2+k_local.
                # The target V is indexed by j*p+k_global. Since p=8, but here we are building the block for a specific k-pair.
                # Actually, the full V is 24xRank.
                # In the block, V is 6xRank (j in 0..2, k_local in 0..1).
                # We need to place these into the correct rows of the full V.
                # Full V row index: j*8 + k_global.
                # But we are building the full U, V, W here.
                pass
        
        # Let's do it properly by filling the full matrices
        # This loop is inefficient if done naively, but rank is small.
        
    # Re-do assembly properly
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total)) # 3*8
    W = np.zeros((24, rank_total)) # 8*3
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U
        U[:, col_start:col_end] = U_block
        
        # V
        # V_block has shape (6, rank_block). Rows are j*2 + k_local.
        # Full V has shape (24, rank_total). Rows are j*8 + k_global.
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        # W
        # W_block has shape (6, rank_block). Rows are k_local*3 + i.
        # Full W has shape (24, rank_total). Rows are k_global*3 + i.
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()
