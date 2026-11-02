"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # safety margin

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1
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

    # Kronecker-structured initialization:
    # View middle dimension j (size 4) as (j1, j2) with j1, j2 in {0,1}
    # This gives a block structure that can be exploited for lower rank.
    # For each (i, k), the sum over j = sum over (j1,j2) of A[i, j1*2+j2] * B[j1*2+j2, k]
    # We can try to factor this using Kronecker products.
    def kronecker_init(r, seed=None):
        """Initialize with Kronecker-block structure."""
        rng = np.random.RandomState(seed)
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        # Try to distribute rank across Kronecker blocks
        # For m=4=2*2, we can think of 4 blocks of size 2x2 in the middle
        # Assign some rank dimensions to each block pattern
        
        # Strategy: use rank dimensions that correspond to (j1, j2) patterns
        # and (i, k) patterns
        n_blocks = 4  # 2x2 blocks in middle dimension
        
        # Assign rank dimensions cyclically to blocks
        for d in range(r):
            block = d % n_blocks
            j1 = block // 2
            j2 = block % 2
            
            # U: for each i, set values at positions corresponding to this block
            for i in range(n):
                j = j1 * 2 + j2
                U[i * m + j, d] = rng.choice([-1, 1])
            
            # V: for each k, set values at positions corresponding to this block
            for k in range(p):
                j = j1 * 2 + j2
                V[j * p + k, d] = rng.choice([-1, 1])
            
            # W: for each i, k, set value
            for i in range(n):
                for k in range(p):
                    W[k * n + i, d] = rng.choice([-1, 1])
        
        return U, V, W

    def try_rank_alt(r, n_trials=10, max_iter=300):
        nonlocal best_U, best_V, best_W, best_rank
        for trial in range(n_trials):
            if time.time() > deadline:
                return False
            
            # Choose initialization strategy
            if trial % 4 == 0:
                # Kronecker-structured initialization
                U, V, W = kronecker_init(r, seed=trial * 1000)
            elif trial % 4 == 1:
                # Standard random
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif trial % 4 == 2:
                # Sparse-like initialization
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for d in range(r):
                    idx_u = np.random.choice(nm, size=max(1, nm//4), replace=False)
                    idx_v = np.random.choice(mp, size=max(1, mp//4), replace=False)
                    idx_w = np.random.choice(pn, size=max(1, pn//4), replace=False)
                    U[idx_u, d] = np.random.choice([-1, 1], size=len(idx_u))
                    V[idx_v, d] = np.random.choice([-1, 1], size=len(idx_v))
                    W[idx_w, d] = np.random.choice([-1, 1], size=len(idx_w))
            else:
                # Block-structured initialization with Kronecker flavor
                U = np.random.randn(nm, r) * 0.5
                V = np.random.randn(mp, r) * 0.5
                W = np.random.randn(pn, r) * 0.5
                # Add Kronecker-block structure
                for d in range(r):
                    block = d % 4
                    j1 = block // 2
                    j2 = block % 2
                    for i in range(n):
                        U[i * m + j1*2 + j2, d] += 1.0
                    for k in range(p):
                        V[(j1*2 + j2) * p + k, d] += 1.0
                    for i in range(n):
                        for k in range(p):
                            W[k * n + i, d] += 0.5

            prev_err = np.inf
            for it in range(max_iter):
                if time.time() > deadline:
                    return False
                
                # Build X for optimizing W: X[a*mp+b, d] = U[a,d]*V[b,d]
                X = np.zeros((nm * mp, r))
                for d in range(r):
                    X[:, d] = (U[:, d][:, None] * V[:, d][None, :]).flatten()
                
                # Solve for W
                XTX = X.T @ X + 1e-10 * np.eye(r)
                XTT = X.T @ T_vec_w.T
                try:
                    W_new = np.linalg.solve(XTX, XTT).T
                except np.linalg.LinAlgError:
                    W_new = W
                
                # Build Y for optimizing U: Y[b*pn+c, d] = V[b,d]*W[c,d]
                Y = np.zeros((mp * pn, r))
                for d in range(r):
                    Y[:, d] = (V[:, d][:, None] * W_new[:, d][None, :]).flatten()
                
                # Solve for U
                YTY = Y.T @ Y + 1e-10 * np.eye(r)
                YTT = Y.T @ T_vec_u.T
                try:
                    U_new = np.linalg.solve(YTY, YTT).T
                except np.linalg.LinAlgError:
                    U_new = U
                
                # Build Z for optimizing V: Z[a*pn+c, d] = U[a,d]*W[c,d]
                Z = np.zeros((nm * pn, r))
                for d in range(r):
                    Z[:, d] = (U_new[:, d][:, None] * W_new[:, d][None, :]).flatten()
                
                # Solve for V
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
                    break
                if abs(prev_err - err) < 1e-12 and it > 10:
                    break
                prev_err = err
            
            # Final check
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
                    best_U, best_V, best_W = U, V, W
                best_rank = r
                return True
        return False

    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # But Kronecker structure may allow lower ranks
    # Try a few candidate ranks based on structural insights
    candidate_ranks = []
    
    # Add lower bound
    lower_bound = max(n*m, m*p, p*n)
    
    # Add some structured candidates
    # For m=4=2*2, try ranks that are multiples of 2 or 4
    for r in range(lower_bound, best_rank):
        candidate_ranks.append(r)
    
    # Sort candidates: try lower ranks first, but also try some structured ones
    # Prioritize ranks that are multiples of small numbers (Kronecker-friendly)
    def rank_priority(r):
        # Prefer lower ranks
        score = r
        # Bonus for Kronecker-friendly ranks (multiples of 2, 4)
        if r % 4 == 0:
            score -= 2
        elif r % 2 == 0:
            score -= 1
        return score
    
    candidate_ranks.sort(key=rank_priority)
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        success = try_rank_alt(r, n_trials=8)
        if success:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
