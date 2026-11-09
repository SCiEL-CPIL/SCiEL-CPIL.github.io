import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36
    
    # Build the target tensor T for 2x3x6
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = schoolbook_rank

    # Helper to verify
    def verify(U, V, W, T_target, tol=1e-5):
        try:
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            if np.max(np.abs(T_recon - T_target)) < tol:
                return True
        except:
            pass
        return False

    # Helper to clean
    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-6:
                    clean_row.append(int(round(val)))
                elif abs(val*2 - round(val*2)) < 1e-6:
                    clean_row.append(round(val*2)/2.0)
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result

    # Try to find rank 34 decomposition using two rank 17 2x3x3 decompositions
    # Target: M_{2,3,3}
    n_sub, m_sub, p_sub = 2, 3, 3
    T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub), dtype=float)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1.0
    
    def find_rank_17_2x3x3():
        # Returns U, V, W or None
        # U: 6x17, V: 9x17, W: 6x17
        target_rank = 17
        # Variables: U (6*17), V (9*17), W (6*17)
        # Total vars: 17 * (6+9+6) = 357
        
        def loss(params):
            U_flat, V_flat, W_flat = params[:6*target_rank], params[6*target_rank:6*target_rank+9*target_rank], params[6*target_rank+9*target_rank:]
            U = U_flat.reshape(6, target_rank)
            V = V_flat.reshape(9, target_rank)
            W = W_flat.reshape(6, target_rank)
            T_recon = np.einsum('xr,yr,zr->xyz', U, V, W)
            return np.sum((T_recon - T_sub)**2)
        
        # Try a few random initializations
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            random.seed(seed)
            np.random.seed(seed)
            # Initialize with small random values
            init_params = np.random.randn(6*target_rank + 9*target_rank + 6*target_rank) * 0.1
            
            # Use L-BFGS-B
            try:
                res = minimize(loss, init_params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
                if res.fun < 1e-6:
                    # Success
                    U_flat, V_flat, W_flat = res.x[:6*target_rank], res.x[6*target_rank:6*target_rank+9*target_rank], res.x[6*target_rank+9*target_rank:]
                    U = U_flat.reshape(6, target_rank)
                    V = V_flat.reshape(9, target_rank)
                    W = W_flat.reshape(6, target_rank)
                    return U, V, W
            except Exception:
                pass
        return None

    U_sub, V_sub, W_sub = find_rank_17_2x3x3()
    
    if U_sub is not None:
        # Construct 2x3x6 solution
        # We use two copies of the sub-decomposition
        # Block 1: k in {0,1,2}
        # Block 2: k in {3,4,5}
        
        U_total = np.hstack([U_sub, U_sub]) # 6 x 34
        # V_sub is 9x17. Rows correspond to (j, k_local) where k_local in {0,1,2}.
        # We need to map to V_total which is 18x34.
        # V_total rows: j*6 + k_global.
        # For block 1: k_global = k_local.
        # For block 2: k_global = k_local + 3.
        
        V_total = np.zeros((18, 34))
        W_total = np.zeros((12, 34))
        
        # Fill Block 1 (cols 0-16 of U/V/W total? No, cols 0-16 are the first 17 terms)
        # U is same for both, so U_total is just [U_sub, U_sub].
        
        # V mapping:
        # V_sub row index: j*3 + k_local
        # V_total row index: j*6 + k_global
        for j in range(3):
            for k_local in range(3):
                v_idx_sub = j*3 + k_local
                v_idx_total_1 = j*6 + k_local
                v_idx_total_2 = j*6 + (k_local + 3)
                
                # First 17 cols of V_total come from V_sub
                V_total[v_idx_total_1, 0:17] = V_sub[v_idx_sub, :]
                # Second 17 cols of V_total come from V_sub (same A, different B part)
                # Wait, the decomposition is for A * B.
                # If we use the SAME U_sub for both blocks, does it work?
                # C1 = A B1 = sum U_r V1_r W1_r
                # C2 = A B2 = sum U_r V2_r W2_r
                # If we set U_total = [U_sub, U_sub], then:
                # Term r in 0..16: U_sub[:,r] * V_total[:,r] * W_total[:,r]
                # Term r in 17..33: U_sub[:,r-17] * V_total[:,r] * W_total[:,r]
                
                # We need V_total[:, 17:34] to correspond to B2.
                # Since B2 is just B1 shifted in k, and the tensor structure is identical,
                # we can use the same V_sub pattern, just mapped to the new k indices.
                V_total[v_idx_total_2, 17:34] = V_sub[v_idx_sub, :]
        
        # W mapping:
        # W_sub row index: k_local*2 + i
        # W_total row index: k_global*2 + i
        for k_local in range(3):
            for i in range(2):
                w_idx_sub = k_local*2 + i
                w_idx_total_1 = k_local*2 + i
                w_idx_total_2 = (k_local+3)*2 + i
                
                W_total[w_idx_total_1, 0:17] = W_sub[w_idx_sub, :]
                W_total[w_idx_total_2, 17:34] = W_sub[w_idx_sub, :]
        
        if verify(U_total, V_total, W_total, T):
            best_U, best_V, best_W = U_total, V_total, W_total
            best_rank = 34

    # Return best found
    U_out = to_clean_list(best_U)
    V_out = to_clean_list(best_V)
    W_out = to_clean_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
