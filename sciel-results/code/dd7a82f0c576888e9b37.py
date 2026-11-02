import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9

    def verify_decomp(U, V, W, n, m, p, atol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)

    def build_rank32():
        """Rank-32: 4 Strassen blocks (7 each) + 1 schoolbook block (4)."""
        terms = []
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            # Strassen for 2x2 @ 2x2
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        # Schoolbook for k=8
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        return terms_to_matrices(terms)

    def build_rank28():
        """Rank-28: 4 Laderman blocks (7 each) for columns 0-7, schoolbook for col 8 (4).
        Wait, that's 28+4=32. Let me reconsider.
        Laderman is for 4x4. Not directly applicable to 2x2.
        Actually, let's try a different approach: use the structure of the 2x2x9 tensor
        to find a rank-28 decomposition directly.
        
        Alternative: Group columns into 3 groups of 3.
        If we can find a rank-9 decomposition for 2x2x3, then 3*9=27.
        If rank-8, then 3*8=24.
        Let's try to find a valid rank-9 or rank-8 for 2x2x3 and tile it.
        """
        # This is a placeholder for now; the search below will try to find it.
        return None

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx, t] = val
            for idx, val in Vd.items():
                V[idx, t] = val
            for idx, val in Wd.items():
                W[idx, t] = val
        return U, V, W

    def build_from_block(U_block, V_block, W_block, n_groups=3):
        """Build full 2x2x9 decomposition from 2x2x3 block decomposition."""
        R_block = U_block.shape[1]
        R_total = n_groups * R_block
        U_full = np.zeros((4, R_total))
        V_full = np.zeros((18, R_total))
        W_full = np.zeros((18, R_total))
        
        for b in range(n_groups):
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

    # Try to find a low-rank decomposition for 2x2x3
    # If we find rank R for 2x2x3, we get rank 3R for 2x2x9.
    # Rank 8 -> 24, Rank 9 -> 27, Rank 10 -> 30.
    
    # Since numerical search is unreliable and slow, let's try to construct
    # a rank-9 or rank-8 decomposition analytically or via a simpler search.
    
    # Actually, the known rank of [2,2,3] is 8.
    # Let's try to find it using a more robust method or a known construction.
    
    # Known rank-8 construction for 2x2x3:
    # This is non-trivial. Let's try a simple search for rank 8 and 9.
    
    def search_rank_223(R, max_restarts=50, deadline=None):
        """Search for rank-R decomposition of 2x2x3."""
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
        
        for attempt in range(max_restarts):
            if deadline is not None and time.time() > deadline:
                break
            x0 = np.random.randn(16*R) * (0.5 + attempt * 0.05)
            try:
                from scipy.optimize import least_squares
                res = least_squares(residuals, x0, method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if res.cost < 1e-10:
                        break
            except:
                continue
        
        if best_sol is not None and best_cost < 1e-8:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+6*R].reshape(6, R)
            W = best_sol[4*R+6*R:4*R+12*R].reshape(6, R)
            # Snap to half-integers or integers
            for snap in [2, 4, 8]:
                U_s = np.round(U * snap) / snap
                V_s = np.round(V * snap) / snap
                W_s = np.round(W * snap) / snap
                if verify_decomp(U_s, V_s, W_s, 2, 2, 3):
                    return U_s, V_s, W_s
            return U, V, W
        return None

    # Try to find rank-8 or rank-9 decomposition for 2x2x3
    # But this is slow and might not succeed within time limits.
    # Let's stick with the known rank-32 construction as a fallback.
    # And try a quick search for rank-9 (total 27) or rank-8 (total 24).
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    # Try rank 8 for 2x2x3 (total 24)
    if time.time() < deadline - 20:
        sol = search_rank_223(8, max_restarts=20, deadline=deadline - 20)
        if sol is not None:
            U_b, V_b, W_b = sol
            U_full, V_full, W_full = build_from_block(U_b, V_b, W_b)
            if verify_decomp(U_full, V_full, W_full, n, m, p):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Try rank 9 for 2x2x3 (total 27)
    if time.time() < deadline - 10:
        sol = search_rank_223(9, max_restarts=20, deadline=deadline - 10)
        if sol is not None:
            U_b, V_b, W_b = sol
            U_full, V_full, W_full = build_from_block(U_b, V_b, W_b)
            if verify_decomp(U_full, V_full, W_full, n, m, p):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: Rank 32
    U, V, W = build_rank32()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
