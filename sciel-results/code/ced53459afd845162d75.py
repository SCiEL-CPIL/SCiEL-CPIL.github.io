import numpy as np
import time

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s

    # Dimensions
    n_U = n * m  # 6
    n_V = m * p  # 18
    n_W = p * n  # 12
    n_eq = n * p * n * m * m * p  # 648 constraints

    # Schoolbook baseline (rank 36)
    r_sb = n * m * p
    U_sb = np.zeros((n_U, r_sb))
    V_sb = np.zeros((n_V, r_sb))
    W_sb = np.zeros((n_W, r_sb))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U_sb[ii * m + jj, col] = 1
                V_sb[jj * p + kk, col] = 1
                W_sb[kk * n + ii, col] = 1
                col += 1

    best_rank = r_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def build_system(U, V, r):
        """Build A (n_eq x n_W*r) and b (n_eq,) such that A @ W_flat = b."""
        A = np.zeros((n_eq, n_W * r))
        b = np.zeros(n_eq)
        eq = 0
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b_idx in range(m):
                        for bp in range(m):
                            for c in range(p):
                                b[eq] = 1.0 if (a == i and b_idx == bp and c == k) else 0.0
                                for l in range(r):
                                    col = (i * p + k) * r + l
                                    A[eq, col] = U[a * m + b_idx, l] * V[bp * p + c, l]
                                eq += 1
        return A, b

    def verify(U, V, W, atol=1e-7):
        """Vectorized verification."""
        r = U.shape[1]
        A, b = build_system(U, V, r)
        W_flat = W.flatten()
        residual = np.linalg.norm(A @ W_flat - b)
        return residual < atol * np.sqrt(n_eq)

    def solve_for_W(U, V, r):
        """Given U,V, solve for W via least squares."""
        A, b = build_system(U, V, r)
        W_flat, residuals, rank_mat, sv = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape(n_W, r)
        if residuals and len(residuals) > 0:
            res = residuals[0]
        else:
            res = np.linalg.norm(A @ W_flat - b)
        return W, res

    # Alternative approach: exploit the p=6 structure by splitting into two 2x3x3 blocks
    # This gives a block-diagonal-like structure that may allow rank reduction
    
    # For the 2x3x3 case, the minimal rank is known to be 7 (Strassen-like)
    # For 2x3x6, we can try to combine two 2x3x3 multiplications
    
    # Let's try a more systematic search using the fact that the tensor has structure
    
    # Approach: Use the known rank-7 decomposition for 2x3x3 and combine
    # 2x3x3 multiplication: C[2x3] = A[2x3] * B[3x3]
    # Strassen-like rank 7 decomposition for 2x3x3:
    
    # Actually, let's try a direct numerical optimization approach
    # We'll use a gradient-free method (Nelder-Mead) on the residual
    
    from scipy.optimize import minimize
    
    def objective(params, r):
        """Objective function: residual of the decomposition."""
        # Unpack parameters
        n_U_params = n_U * r
        n_V_params = n_V * r
        
        U_flat = params[:n_U_params]
        V_flat = params[n_U_params:n_U_params + n_V_params]
        
        U = U_flat.reshape(n_U, r)
        V = V_flat.reshape(n_V, r)
        
        W, res = solve_for_W(U, V, r)
        return res
    
    def random_init_params(r, seed=0):
        """Generate random initial parameters."""
        np.random.seed(seed)
        U = np.random.randn(n_U, r) * 0.5
        V = np.random.randn(n_V, r) * 0.5
        params = np.concatenate([U.flatten(), V.flatten()])
        return params
    
    # Try rank 33, 34, 35 with optimization
    for target_r in [33, 34, 35]:
        if time.time() - start_time > deadline - 30:
            break
        
        found = False
        n_restarts = 20 if target_r <= 34 else 10
        
        for restart in range(n_restarts):
            if time.time() - start_time > deadline - 10:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Start from schoolbook-like structure
                U = np.zeros((n_U, target_r))
                V = np.zeros((n_V, target_r))
                n_fill = min(target_r, n * m * p)
                col = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            if col >= target_r:
                                break
                            U[ii * m + jj, col] = 1
                            V[jj * p + kk, col] = 1
                            col += 1
                for l in range(col, target_r):
                    U[:, l] = np.random.randn(n_U) * 0.1
                    V[:, l] = np.random.randn(n_V) * 0.1
            elif restart == 1:
                # Block structure: exploit p=6 as two 3-blocks
                U = np.zeros((n_U, target_r))
                V = np.zeros((n_V, target_r))
                # First 18 columns for k=0,1,2
                col = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(3):
                            if col >= target_r:
                                break
                            U[ii * m + jj, col] = 1
                            V[jj * p + kk, col] = 1
                            col += 1
                # Remaining for k=3,4,5
                for l in range(col, target_r):
                    U[:, l] = np.random.randn(n_U) * 0.3
                    V[:, l] = np.random.randn(n_V) * 0.3
            else:
                # Random
                np.random.seed(restart * 1000)
                U = np.random.randn(n_U, target_r) * 0.5
                V = np.random.randn(n_V, target_r) * 0.5
            
            # Coordinate descent refinement
            best_res = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), None
            
            W, res = solve_for_W(U, V, target_r)
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(60):
                if time.time() - start_time > deadline - 15:
                    break
                
                # Update random column of U
                l = np.random.randint(0, target_r)
                U_trial = U.copy()
                U_trial[:, l] = np.random.randn(n_U) * 0.3 + U[:, l] * 0.5
                W_trial, res_trial = solve_for_W(U_trial, V, target_r)
                if res_trial < best_res * (1 - 1e-8):
                    U = U_trial
                    W = W_trial
                    best_res = res_trial
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Update random column of V
                l = np.random.randint(0, target_r)
                V_trial = V.copy()
                V_trial[:, l] = np.random.randn(n_V) * 0.3 + V[:, l] * 0.5
                W_trial, res_trial = solve_for_W(U, V_trial, target_r)
                if res_trial < best_res * (1 - 1e-8):
                    V = V_trial
                    W = W_trial
                    best_res = res_trial
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_res < 1e-10:
                    break
            
            if best_res < 1e-8:
                if verify(best_U, best_V, best_W):
                    found = True
                    best_rank = target_r
                    best_U, best_V, best_W = best_U.copy(), best_V.copy(), best_W.copy()
                    break
        
        if found:
            break
    
    # Final verification
    if verify(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
