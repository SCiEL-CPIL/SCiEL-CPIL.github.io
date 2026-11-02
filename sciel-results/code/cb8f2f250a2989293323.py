"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Precompute T in different orientations for efficiency
    T_vec_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # For optimizing W
    T_vec_u = T.reshape(nm, mp * pn)  # For optimizing U
    T_vec_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # For optimizing V
    
    # Kronecker structure: m = 4 = 2 * 2, so j = j1*2 + j2
    # This means the middle dimension has a product structure
    # We can exploit this for structured initializations
    
    def kronecker_init(r, trial):
        """Initialize using Kronecker product structure."""
        m1, m2 = 2, 2  # factor 4 as 2*2
        
        if trial % 4 == 0:
            # Pure Kronecker structure
            # Each rank element d corresponds to (i, j1, j2, k) pattern
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for d in range(r):
                # Cycle through structured indices
                i_d = d % n
                j1_d = (d // n) % m1
                j2_d = (d // (n * m1)) % m2
                k_d = (d // (n * m1 * m2)) % p
                j_d = j1_d * m2 + j2_d
                
                U[i_d * m + j_d, d] = 1.0
                V[j_d * p + k_d, d] = 1.0
                W[k_d * n + i_d, d] = 1.0
            return U, V, W
        elif trial % 4 == 1:
            # Kronecker with sign patterns
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for d in range(r):
                i_d = d % n
                j1_d = (d // n) % m1
                j2_d = (d // (n * m1)) % m2
                k_d = (d // (n * m1 * m2)) % p
                j_d = j1_d * m2 + j2_d
                
                # Use Kronecker-like sign patterns
                s_u = 1.0 if (i_d + j1_d) % 2 == 0 else -1.0
                s_v = 1.0 if (j2_d + k_d) % 2 == 0 else -1.0
                s_w = 1.0 if (k_d + i_d) % 2 == 0 else -1.0
                
                U[i_d * m + j_d, d] = s_u
                V[j_d * p + k_d, d] = s_v
                W[k_d * n + i_d, d] = s_w
            return U, V, W
        elif trial % 4 == 2:
            # Mixed: Kronecker base + random perturbation
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for d in range(r):
                i_d = d % n
                j1_d = (d // n) % m1
                j2_d = (d // (n * m1)) % m2
                k_d = (d // (n * m1 * m2)) % p
                j_d = j1_d * m2 + j2_d
                
                U[i_d * m + j_d, d] = 1.0 + np.random.randn() * 0.1
                V[j_d * p + k_d, d] = 1.0 + np.random.randn() * 0.1
                W[k_d * n + i_d, d] = 1.0 + np.random.randn() * 0.1
            return U, V, W
        else:
            # Block Kronecker: exploit j1, j2 separation
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for d in range(r):
                i_d = d % n
                j1_d = (d // n) % m1
                j2_d = (d // (n * m1)) % m2
                k_d = (d // (n * m1 * m2)) % p
                
                # U: contribute to both j1 and j2 blocks
                U[i_d * m + j1_d * m2, d] = 1.0
                U[i_d * m + j1_d * m2 + 1, d] = 0.5
                U[i_d * m + (j1_d + 1) * m2, d] = 0.5
                U[i_d * m + (j1_d + 1) * m2 + 1, d] = 0.25
                
                # V: similar block structure
                V[j1_d * m2 * p + k_d, d] = 1.0
                V[j1_d * m2 * p + p + k_d, d] = 0.5
                V[(j1_d + 1) * m2 * p + k_d, d] = 0.5
                V[(j1_d + 1) * m2 * p + p + k_d, d] = 0.25
                
                W[k_d * n + i_d, d] = 1.0
            return U, V, W
    
    def random_init(r, trial):
        """Standard random initialization."""
        if trial % 5 == 0:
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
        elif trial % 5 == 1:
            # Sparse-like with ±1 entries
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for d in range(r):
                idx_u = np.random.choice(nm, size=max(1, nm//3), replace=False)
                idx_v = np.random.choice(mp, size=max(1, mp//3), replace=False)
                idx_w = np.random.choice(pn, size=max(1, pn//3), replace=False)
                U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
        elif trial % 5 == 2:
            # Hybrid
            U = np.random.randn(nm, r) * 0.3
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            for d in range(r):
                U[d % nm, d] += 1.0
                V[d % mp, d] += 1.0
                W[d % pn, d] += 1.0
        elif trial % 5 == 3:
            # Structured with full i,j,k coverage
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for d in range(r):
                i_d = d % n
                j_d = (d // n) % m
                k_d = (d // (n * m)) % p
                U[i_d * m + j_d, d] = 1.0
                V[j_d * p + k_d, d] = 1.0
                W[k_d * n + i_d, d] = 1.0
        else:
            # Gaussian with scale
            scale = np.sqrt(nm * mp * pn / r)
            U = np.random.randn(nm, r) * scale
            V = np.random.randn(mp, r) * scale
            W = np.random.randn(pn, r) * scale
        return U, V, W
    
    def alternating_minimize(U, V, W, max_iter=600):
        """Run alternating minimization to convergence."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                return U, V, W, False
            
            # Optimize W: T_vec_w[c, :] ≈ W[c, :] @ X.T where X[a*mp+b, d] = U[a,d]*V[b,d]
            X = np.einsum('ad,bd->abd', U, V).reshape(nm * mp, r)
            XTX = X.T @ X + 1e-10 * np.eye(r)
            XTT = X.T @ T_vec_w.T
            try:
                W_new = np.linalg.solve(XTX, XTT).T
            except np.linalg.LinAlgError:
                W_new = W
            
            # Optimize U: T_vec_u[a, :] ≈ U[a, :] @ Y.T where Y[b*pn+c, d] = V[b,d]*W[c,d]
            Y = np.einsum('bd,cd->bcd', V, W_new).reshape(mp * pn, r)
            YTY = Y.T @ Y + 1e-10 * np.eye(r)
            YTT = Y.T @ T_vec_u.T
            try:
                U_new = np.linalg.solve(YTY, YTT).T
            except np.linalg.LinAlgError:
                U_new = U
            
            # Optimize V: T_vec_v[b, :] ≈ V[b, :] @ Z.T where Z[a*pn+c, d] = U[a,d]*W[c,d]
            Z = np.einsum('ad,cd->acd', U_new, W_new).reshape(nm * pn, r)
            ZTZ = Z.T @ Z + 1e-10 * np.eye(r)
            ZTT = Z.T @ T_vec_v.T
            try:
                V_new = np.linalg.solve(ZTZ, ZTT).T
            except np.linalg.LinAlgError:
                V_new = V
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            T_rec = tensor_from_factors(U, V, W)
            err = np.linalg.norm(T_rec - T)
            if err < 1e-10:
                return U, V, W, True
            if abs(prev_err - err) < 1e-12 and it > 20:
                return U, V, W, err < 1e-8
            prev_err = err
        
        T_rec = tensor_from_factors(U, V, W)
        err = np.linalg.norm(T_rec - T)
        return U, V, W, err < 1e-8
    
    def try_rank(r, n_trials=15, max_iter=600):
        """Try to find a decomposition at rank r."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Choose initialization strategy
            if trial % 2 == 0:
                U, V, W = kronecker_init(r, trial)
            else:
                U, V, W = random_init(r, trial)
            
            U, V, W, converged = alternating_minimize(U, V, W, max_iter)
            
            if converged:
                T_rec = tensor_from_factors(U, V, W)
                if np.allclose(T_rec, T, atol=1e-8, rtol=1e-8):
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    T_rec_r = tensor_from_factors(U_r, V_r, W_r)
                    if np.allclose(T_rec_r, T, atol=1e-8, rtol=1e-8):
                        best_U, best_V, best_W = U_r, V_r, W_r
                    else:
                        # Try rounding to integers
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        T_rec_i = tensor_from_factors(U_i, V_i, W_i)
                        if np.allclose(T_rec_i, T, atol=1e-8, rtol=1e-8):
                            best_U, best_V, best_W = U_i, V_i, W_i
                        else:
                            best_U, best_V, best_W = U, V, W
                    best_rank = r
                    return True
        return False
    
    # Try ranks from lower bound upward
    # Theoretical lower bound for matrix multiplication is max(n, m, p) = 7
    # But practical lower bound is higher. Let's try from reasonable range.
    # For 2x4 x 4x7, known bounds suggest rank could be significantly below 56.
    # Try ranks: focus on 20-40 range first, then expand
    
    # First, try a few key ranks with more trials
    key_ranks = [28, 32, 35, 36, 40, 42, 44, 48, 49, 50, 52, 54, 55]
    for r in key_ranks:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        success = try_rank(r, n_trials=8, max_iter=800)
        if success:
            break
    
    # If not found in key ranks, sweep through remaining
    if best_rank == rank_sb:
        for r in range(20, best_rank):
            if time.time() > deadline:
                break
            if r in key_ranks:
                continue
            success = try_rank(r, n_trials=5, max_iter=400)
            if success:
                break
    
    # Final polish: if we found something, try to refine
    if best_rank < rank_sb:
        if time.time() < deadline - 30:
            # Try to reduce rank further
            for r in range(max(7, best_rank - 5), best_rank):
                if time.time() > deadline:
                    break
                success = try_rank(r, n_trials=10, max_iter=1000)
                if success:
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
