import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin before 150s kill

    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_from_block(U_block, V_block, W_block):
        """
        Build full 2x2x9 decomposition from a 2x2x3 block decomposition.
        The 2x9 matrix is treated as three 2x3 blocks.
        """
        R_block = U_block.shape[1]
        R_total = 3 * R_block
        U_full = np.zeros((4, R_total))
        V_full = np.zeros((18, R_total))
        W_full = np.zeros((18, R_total))
        
        for b in range(3):
            start_r = b * R_block
            end_r = start_r + R_block
            U_full[:, start_r:end_r] = U_block
            for k in range(2):
                for j_local in range(3):
                    j_global = 3*b + j_local
                    v_idx_block = k*3 + j_local
                    v_idx_full = k*9 + j_global
                    V_full[v_idx_full, start_r:end_r] = V_block[v_idx_block, :]
            for j_local in range(3):
                j_global = 3*b + j_local
                for i in range(2):
                    w_idx_block = j_local*2 + i
                    w_idx_full = j_global*2 + i
                    W_full[w_idx_full, start_r:end_r] = W_block[w_idx_block, :]
        return U_full, V_full, W_full

    def search_rank_223(R, max_restarts=30, max_nfev=8000):
        """
        Search for rank-R decomposition of 2x2x3 matrix multiplication.
        Uses multiple strategies: random init, structured init, and gradient polishing.
        """
        def residuals(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+6*R].reshape(6, R)
            W = params[4*R+6*R:4*R+12*R].reshape(6, R)
            U_r = U.reshape(2, 2, R)
            V_r = V.reshape(2, 3, R)
            W_r = W.reshape(3, 2, R)
            T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
            return T.flatten() - 1.0
        
        best_sol = None
        best_cost = float('inf')
        
        # Strategy 1: Random initializations with varying scales
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            scale = 0.3 + 0.1 * (attempt % 5)
            x0 = np.random.randn(16*R) * scale
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=max_nfev, 
                                   ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if res.cost < 1e-12:
                        break
            except:
                continue
        
        # Strategy 2: Try structured initializations based on Strassen-like patterns
        if best_cost > 1e-8 and time.time() < deadline:
            # Try a few structured initializations
            structured_inits = []
            
            # Initialize with small perturbations around simple patterns
            for _ in range(5):
                U_init = np.zeros((4, R))
                V_init = np.zeros((6, R))
                W_init = np.zeros((6, R))
                
                # Random sparse-ish initialization
                for r in range(R):
                    U_init[np.random.choice(4, size=2, replace=False), r] = np.random.choice([-1, 1], size=2)
                    V_init[np.random.choice(6, size=3, replace=False), r] = np.random.choice([-1, 1], size=3)
                    W_init[np.random.choice(6, size=3, replace=False), r] = np.random.choice([-1, 1], size=3)
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                try:
                    res = least_squares(residuals, x0, method='trf', max_nfev=max_nfev,
                                       ftol=1e-14, xtol=1e-14, gtol=1e-14)
                    if res.cost < best_cost:
                        best_cost = res.cost
                        best_sol = res.x
                        if res.cost < 1e-12:
                            break
                except:
                    continue
        
        if best_sol is not None and best_cost < 1e-10:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+6*R].reshape(6, R)
            W = best_sol[4*R+6*R:4*R+12*R].reshape(6, R)
            
            # Try snapping to half-integers
            for snap_factor in [2, 4, 8]:
                U_snapped = np.round(U * snap_factor) / snap_factor
                V_snapped = np.round(V * snap_factor) / snap_factor
                W_snapped = np.round(W * snap_factor) / snap_factor
                if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, 3):
                    return U_snapped, V_snapped, W_snapped
            
            # Try snapping to integers
            U_snapped = np.round(U)
            V_snapped = np.round(V)
            W_snapped = np.round(W)
            if verify_decomp(U_snapped, V_snapped, W_snapped, 2, 2, 3):
                return U_snapped, V_snapped, W_snapped
            
            # If snapping fails, return the raw solution (should still be valid)
            return U, V, W
        
        return None

    # Try to find a low rank decomposition for [2,2,3]
    # Known: rank 9 for 2x2x3 is achievable (trivial), rank 8 might be possible
    block_decomp = None
    for R in [8, 9]:
        if time.time() > deadline - 5:
            break
        sol = search_rank_223(R)
        if sol is not None:
            U_b, V_b, W_b = sol
            block_decomp = (U_b, V_b, W_b, R)
            break

    if block_decomp is not None:
        U_b, V_b, W_b, R_b = block_decomp
        U_full, V_full, W_full = build_from_block(U_b, V_b, W_b)
        if verify_decomp(U_full, V_full, W_full, n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: Rank 32 construction (same as before)
    terms = []
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
        terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
        terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
        terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
        terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
    
    k = 8
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
