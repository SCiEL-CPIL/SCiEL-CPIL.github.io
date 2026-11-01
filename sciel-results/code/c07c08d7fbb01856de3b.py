import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin

    n, m, p = DIMENSIONS  # 2, 2, 6

    # Build the target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        S = np.einsum('as,bs,cs->abc', U, V, W)
        return np.allclose(S, T, atol=1e-9)

    def snap_half(U, V, W):
        """Snap to nearest half-integer and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    # Fallback: Rank 21 via 3 independent Strassen
    rank_strassen = 21
    U_str = np.zeros((n*m, rank_strassen))
    V_str = np.zeros((m*p, rank_strassen))
    W_str = np.zeros((p*n, rank_strassen))

    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    for b in range(3):  # 3 blocks of 2 columns
        for s_idx in range(7):
            col = b * 7 + s_idx
            u_vec, v_vec, w_vec = strassen_terms[s_idx]
            for idx in range(4):
                U_str[idx, col] = u_vec[idx]
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                V_str[v_indices[idx], col] = v_vec[idx]
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                W_str[w_indices[idx], col] = w_vec[idx]

    best_rank = 21
    best_U, best_V, best_W = U_str, V_str, W_str

    # Try to find lower rank
    # Ranges to try: 12, 13, 14, 15, 16, 17, 18, 19, 20
    # Start with lower ranks as they are more valuable
    ranks_to_try = [12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    np.random.seed(42)
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break

        def make_residuals(R):
            def residuals(params):
                # params: [U (4 x R), V (12 x R), W (12 x R)] flattened
                U_s = params[0:R*4].reshape(R, 4).T  # 4 x R
                V_s = params[R*4:R*16].reshape(R, 12).T  # 12 x R
                W_s = params[R*16:].reshape(R, 12).T  # 12 x R
                S = np.einsum('as,bs,cs->abc', U_s, V_s, W_s)
                return (S - T).flatten()
            return residuals

        residuals = make_residuals(target_rank)
        
        # Generate initializations
        inits = []
        
        # 1. Random initializations
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            U_rand = np.random.randn(4, target_rank) * 0.5
            V_rand = np.random.randn(12, target_rank) * 0.5
            W_rand = np.random.randn(12, target_rank) * 0.5
            init_params = np.concatenate([
                U_rand.T.flatten(),
                V_rand.T.flatten(),
                W_rand.T.flatten()
            ])
            inits.append(init_params)
            
        # 2. Initialized from Rank-21 solution (truncate + perturb)
        # Take the first target_rank columns of the Strassen solution
        if target_rank <= 21:
            U_init = U_str[:, :target_rank]
            V_init = V_str[:, :target_rank]
            W_init = W_str[:, :target_rank]
            # Add small noise to escape local minima
            U_init += np.random.randn(4, target_rank) * 0.01
            V_init += np.random.randn(12, target_rank) * 0.01
            W_init += np.random.randn(12, target_rank) * 0.01
            init_params = np.concatenate([
                U_init.T.flatten(),
                V_init.T.flatten(),
                W_init.T.flatten()
            ])
            inits.append(init_params)
            
            # Another variant: Permute columns to mix blocks
            perm = np.random.permutation(21)[:target_rank]
            U_init2 = U_str[:, perm]
            V_init2 = V_str[:, perm]
            W_init2 = W_str[:, perm]
            U_init2 += np.random.randn(4, target_rank) * 0.01
            V_init2 += np.random.randn(12, target_rank) * 0.01
            W_init2 += np.random.randn(12, target_rank) * 0.01
            init_params2 = np.concatenate([
                U_init2.T.flatten(),
                V_init2.T.flatten(),
                W_init2.T.flatten()
            ])
            inits.append(init_params2)

        found = False
        for init in inits:
            if time.time() > deadline - 5:
                break
            try:
                # Use 'trf' method for robustness with bounds if needed, or 'lm' if fast
                # 'lm' is faster but can fail if Jacobian is singular or rank deficient
                # 'trf' is slower but more robust
                result = least_squares(residuals, init, method='trf', max_nfev=500)
                
                # Check if residual is close to zero
                if result.cost < 1e-8:
                    R = target_rank
                    U_sol = result.x[0:R*4].reshape(R, 4).T
                    V_sol = result.x[R*4:R*16].reshape(R, 12).T
                    W_sol = result.x[R*16:].reshape(R, 12).T

                    valid, U_snap, V_snap, W_snap = snap_half(U_sol, V_sol, W_sol)
                    if valid:
                        best_rank = R
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
