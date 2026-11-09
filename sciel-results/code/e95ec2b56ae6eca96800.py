import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0 # 110s to leave 10s for safety
    
    # 1. Construct Schoolbook (Rank 63) as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
        
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # 2. Try to find a lower rank decomposition
    # We test a few candidate ranks
    candidate_ranks = [57, 60, 62] # Try a few below 63
    
    # Generate random test matrices to validate the decomposition
    # We need to check if C' == AB for random A, B
    rng = np.random.default_rng(42)
    num_tests = 5
    A_tests = [rng.standard_normal((n, m)) for _ in range(num_tests)]
    B_tests = [rng.standard_normal((m, p)) for _ in range(num_tests)]
    
    def objective(x):
        """
        x is flattened U, V, W.
        Returns the squared error.
        """
        # Unpack
        u_len = n * m
        v_len = m * p
        w_len = p * n
        R = x.size // (u_len + v_len + w_len) # This is fixed, but we pass R in closure
        
        U_flat = x[:R*u_len].reshape(R, u_len).T # Shape (u_len, R)
        V_flat = x[R*u_len:2*R*u_len+R*v_len] # Wait, sizes are different
        # Correct unpacking
        offset_u = 0
        offset_v = R * u_len
        offset_w = R * (u_len + v_len)
        
        U = x[offset_u:offset_v].reshape(u_len, R)
        V = x[offset_v:offset_w].reshape(v_len, R)
        W = x[offset_w:].reshape(w_len, R)
        
        err = 0.0
        for A in A_tests:
            for B in B_tests:
                # Reconstruct C
                # C_ik = sum_r W_{k*n+i, r} * (sum_j A_{i,j} U_{i*m+j, r}) * (sum_j' B_{j',k} V_{j'*p+k, r})
                
                # Vectorize for speed
                # Term A_part: (n, m) x (m, R) -> (n, R)
                # A is (n, m). U is (n*m, R). 
                # U reshaped to (n, m, R).
                U_reshaped = U.reshape(n, m, R)
                # Sum over j: A[i,j] * U[i,j,r]
                # np.einsum('ij,ijr->ir', A, U_reshaped)
                A_part = np.einsum('ij,ijr->ir', A, U_reshaped)
                
                # Term B_part: (m, p) x (m, p, R) -> (p, R)
                # B is (m, p). V is (m*p, R).
                V_reshaped = V.reshape(m, p, R)
                # Sum over j: B[j,k] * V[j,k,r]
                B_part = np.einsum('jk,jkr->kr', B, V_reshaped)
                
                # Combine:
                # C_ik = sum_r W_{k*n+i, r} * A_part[i, r] * B_part[k, r]
                # W is (p*n, R). Reshape to (p, n, R).
                W_reshaped = W.reshape(p, n, R)
                
                # C_recon[i, k] = sum_r W_reshaped[k, i, r] * A_part[i, r] * B_part[k, r]
                # Use einsum: 'kr, ir, kir -> ik'?
                # A_part is (n, R). B_part is (p, R). W_reshaped is (p, n, R).
                C_recon = np.einsum('ir, kr, kir -> ik', A_part, B_part, W_reshaped)
                
                C_true = A @ B
                err += np.sum((C_recon - C_true) ** 2)
        
        return err

    # We will try to optimize for specific ranks
    for R in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Initialize
        u_len = n * m
        v_len = m * p
        w_len = p * n
        total_size = R * (u_len + v_len + w_len)
        
        # Random initialization
        x0 = rng.standard_normal(total_size)
        
        # Objective wrapper
        def obj_func(x):
            return objective(x)
            
        try:
            # Use Nelder-Mead or L-BFGS-B?
            # L-BFGS-B is better for smooth functions.
            res = minimize(obj_func, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
            
            if res.success and res.fun < 1e-8:
                # Valid decomposition found!
                offset_u = 0
                offset_v = R * u_len
                offset_w = R * (u_len + v_len)
                
                U_opt = res.x[offset_u:offset_v].reshape(u_len, R)
                V_opt = res.x[offset_v:offset_w].reshape(v_len, R)
                W_opt = res.x[offset_w:].reshape(w_len, R)
                
                # Verify one more time with different seeds
                A_check = rng.standard_normal((n, m))
                B_check = rng.standard_normal((m, p))
                
                U_r = U_opt.reshape(n, m, R)
                V_r = V_opt.reshape(m, p, R)
                W_r = W_opt.reshape(p, n, R)
                
                A_part = np.einsum('ij,ijr->ir', A_check, U_r)
                B_part = np.einsum('jk,jkr->kr', B_check, V_r)
                C_recon = np.einsum('ir, kr, kir -> ik', A_part, B_part, W_r)
                C_true = A_check @ B_check
                
                if np.allclose(C_recon, C_true, atol=1e-8):
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = R
                    print(f"Found rank {R} decomposition!")
                    # We can stop if we find a very good rank, or continue to find even lower?
                    # For now, break to save time, as lower is better.
                    break
                    
        except Exception as e:
            pass # Continue to next rank or fallback
            
    return best_result
