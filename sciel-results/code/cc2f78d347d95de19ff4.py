"""Matrix multiplication 3,3,5 tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Build the multiplication tensor T[a, b, c]:
    # a = i*m + j (row of A, col of A)
    # b = j*p + k (row of B, col of B)
    # c = k*n + i (row of output, col of output)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    def verify_decomposition(U, V, W, tol=1e-7):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return bool(np.allclose(T_recon, T, atol=tol))

    def try_round(U, V, W, scales=(2, 1, 4, 0.5)):
        """Try rounding to various half-integer/integer scales."""
        for scale in scales:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        return None

    def als_step(U, V, W, T, reg=1e-10):
        """One ALS iteration."""
        r = U.shape[1]
        # Update U
        TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
        h_U = np.einsum('abc,bcd->ad', T, TW)
        G_U = np.einsum('bcd,bce->de', TW, TW)
        try:
            U_new = np.linalg.solve(G_U + reg * np.eye(r), h_U.T).T
        except np.linalg.LinAlgError:
            U_new = h_U
        
        # Update V
        UW = U_new[:, np.newaxis, :] * W[np.newaxis, :, :]
        h_V = np.einsum('abc,acd->bd', T, UW)
        G_V = np.einsum('acd,ace->de', UW, UW)
        try:
            V_new = np.linalg.solve(G_V + reg * np.eye(r), h_V.T).T
        except np.linalg.LinAlgError:
            V_new = h_V
        
        # Update W
        UV = U_new[:, np.newaxis, :] * V_new[np.newaxis, :, :]
        h_W = np.einsum('abc,abd->cd', T, UV)
        G_W = np.einsum('abd,abe->de', UV, UV)
        try:
            W_new = np.linalg.solve(G_W + reg * np.eye(r), h_W.T).T
        except np.linalg.LinAlgError:
            W_new = h_W
        
        return U_new, V_new, W_new

    def run_als(U, V, W, r, max_iters=100, check_every=10):
        """Run ALS and try to round."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            U, V, W = als_step(U, V, W, T)
            
            if it % check_every == check_every - 1:
                result = try_round(U, V, W)
                if result is not None:
                    return result
        
        # Final attempt
        result = try_round(U, V, W)
        if result is not None:
            return result
        
        # Try with smaller regularization
        U, V, W = als_step(U, V, W, T, reg=1e-12)
        U, V, W = als_step(U, V, W, T, reg=1e-12)
        result = try_round(U, V, W)
        if result is not None:
            return result
        
        return None

    def structured_init(r, seed):
        """Create structured initialization based on tensor structure."""
        np.random.seed(seed)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # For each term in the tensor, assign to a rank dimension
        # with some probability of sharing
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        # Try to group terms that share indices
        for idx, (i, j, k) in enumerate(terms):
            d = idx % r
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a, d] += 1.0
            V[b, d] += 1.0
            W[c, d] += 1.0
        
        # Add some random perturbation to break symmetry
        U += np.random.randn(n * m, r) * 0.1
        V += np.random.randn(m * p, r) * 0.1
        W += np.random.randn(p * n, r) * 0.1
        
        return U, V, W

    def random_init(r, seed):
        """Random initialization."""
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        return U, V, W

    def try_rank(r, seed):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        # Try structured init
        U, V, W = structured_init(r, seed)
        result = run_als(U, V, W, r, max_iters=80)
        if result is not None:
            return result
        
        if time.time() > deadline - 10:
            return None
        
        # Try random init
        U, V, W = random_init(r, seed)
        result = run_als(U, V, W, r, max_iters=80)
        if result is not None:
            return result
        
        if time.time() > deadline - 10:
            return None
        
        # Try another random init with different scale
        np.random.seed(seed + 1000)
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        result = run_als(U, V, W, r, max_iters=80)
        if result is not None:
            return result
        
        return None

    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But we can do better with structure
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower bound to schoolbook
    ranks_to_try = list(range(lower_bound, min(rank_sb + 1, lower_bound + 12)))
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 2024, 31415, 27182, 16180, 9999, 8888, 7777, 6666, 5555, 4444, 3333, 2222, 1111]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            
            result = try_rank(r, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if r < best_rank:
                    best_rank = r
                    best_U = U_f.copy()
                    best_V = V_f.copy()
                    best_W = W_f.copy()
                found = True
                break
        
        if found:
            continue
    
    # Convert to lists of floats
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
