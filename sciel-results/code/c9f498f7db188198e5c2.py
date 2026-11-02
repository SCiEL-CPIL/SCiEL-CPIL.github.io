import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def _construct_target_tensor(n, m, p):
    """
    Returns a sparse representation of the target multiplication tensor.
    T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0.
    We return a list of (a, b, c, value) tuples for non-zero entries.
    """
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c, 1.0))
    return non_zeros

def _als_step(U, V, W, target, n, m, p, rank):
    """
    Perform one step of Alternating Least Squares.
    U: (nm, rank), V: (mp, rank), W: (pn, rank)
    target: list of (a, b, c, val)
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the linear systems for U
    # For each row a of U, we solve for U[a, :]
    # System: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # This is A_a * U_a = t_a
    # A_a is (mp * pn) x rank
    # t_a is (mp * pn)
    
    # We can build A for all a at once?
    # A[a, b*pn+c, r] = V[b, r] * W[c, r]
    # This is a 4D tensor.
    
    # Let's just loop over rows of U, V, W for simplicity and clarity, 
    # as dimensions are small.
    
    new_U = np.zeros_like(U)
    new_V = np.zeros_like(V)
    new_W = np.zeros_like(W)
    
    # Update U
    for a in range(nm):
        # Build A (mp * pn, rank) and t (mp * pn)
        A = np.zeros((mp * pn, rank))
        t = np.zeros(mp * pn)
        
        # Fill t from target
        for (ta, tb, tc, tv) in target:
            if ta == a:
                idx = tb * pn + tc
                t[idx] = tv
        
        # Fill A
        for r in range(rank):
            # A[:, r] = V[:, r] * W[:, r] (reshaped)
            # V[:, r] is (mp,), W[:, r] is (pn,)
            # Outer product is (mp, pn)
            outer = np.outer(V[:, r], W[:, r])
            A[:, r] = outer.flatten()
            
        # Solve A * U_a = t
        # Use least squares
        try:
            new_U[a, :] = np.linalg.lstsq(A, t, rcond=None)[0]
        except:
            new_U[a, :] = U[a, :] # Keep old if fail
            
    # Update V
    for b in range(mp):
        A = np.zeros((nm * pn, rank))
        t = np.zeros(nm * pn)
        
        for (ta, tb, tc, tv) in target:
            if tb == b:
                idx = ta * pn + tc
                t[idx] = tv
                
        for r in range(rank):
            outer = np.outer(U[:, r], W[:, r])
            A[:, r] = outer.flatten()
            
        try:
            new_V[b, :] = np.linalg.lstsq(A, t, rcond=None)[0]
        except:
            new_V[b, :] = V[b, :]
            
    # Update W
    for c in range(pn):
        A = np.zeros((nm * mp, rank))
        t = np.zeros(nm * mp)
        
        for (ta, tb, tc, tv) in target:
            if tc == c:
                idx = ta * mp + tb
                t[idx] = tv
                
        for r in range(rank):
            outer = np.outer(U[:, r], V[:, r])
            A[:, r] = outer.flatten()
            
        try:
            new_W[c, :] = np.linalg.lstsq(A, t, rcond=None)[0]
        except:
            new_W[c, :] = W[c, :]
            
    return new_U, new_V, new_W

def _calculate_error(U, V, W, target, n, m, p):
    """
    Calculate the error of the decomposition.
    """
    nm, mp, pn = n * m, m * p, p * n
    error = 0.0
    for (a, b, c, val) in target:
        computed = 0.0
        for r in range(U.shape[1]):
            computed += U[a, r] * V[b, r] * W[c, r]
        error += (computed - val) ** 2
    return np.sqrt(error)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Fallback: Rank 21 Strassen
    def get_rank_21():
        rank = 21
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        strassen = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
            ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
            ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
            ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
            ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
        ]
        
        r = 0
        for b in range(3):
            for t in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[t]
                for idx, val in u_coeffs.items():
                    U[idx][r] = val
                for l, val in v_coeffs.items():
                    if l == 0: v_idx = 0 * p + 2 * b
                    elif l == 1: v_idx = 0 * p + 2 * b + 1
                    elif l == 2: v_idx = 1 * p + 2 * b
                    elif l == 3: v_idx = 1 * p + 2 * b + 1
                    V[v_idx][r] = val
                for (i, k_local), val in w_coeffs.items():
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    target = _construct_target_tensor(n, m, p)
    
    # Try to find rank 20
    best_result = get_rank_21()
    best_rank = 21
    best_error = 0.0
    
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    for target_rank in [20, 19]:
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Normalize to avoid scaling issues
            for r in range(target_rank):
                norm_u = np.linalg.norm(U[:, r])
                norm_v = np.linalg.norm(V[:, r])
                norm_w = np.linalg.norm(W[:, r])
                if norm_u > 0 and norm_v > 0 and norm_w > 0:
                    # Scale such that product of norms is 1? Or just normalize each to 1?
                    # Normalizing each to 1 might be too restrictive.
                    # Let's just run ALS.
                    pass
            
            # Run ALS
            for it in range(100):
                if time.time() > deadline:
                    break
                U, V, W = _als_step(U, V, W, target, n, m, p, target_rank)
                
                # Periodic normalization to prevent divergence
                if it % 10 == 9:
                    for r in range(target_rank):
                        nu = np.linalg.norm(U[:, r])
                        nv = np.linalg.norm(V[:, r])
                        nw = np.linalg.norm(W[:, r])
                        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                            # Scale to keep product of norms constant?
                            # Just normalize to unit norm
                            U[:, r] /= nu
                            V[:, r] /= nv
                            W[:, r] /= nw
                            # But this changes the magnitude of the product.
                            # We need to compensate? No, ALS solves for the values.
                            # Normalizing might break the solution if not careful.
                            # Actually, if we normalize U, V, W, the values in U, V, W change.
                            # The next ALS step will re-optimize.
                            # This is a common trick to handle scaling.
                            
            error = _calculate_error(U, V, W, target, n, m, p)
            
            if error < 1e-5:
                # Convert to list of lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                
    return best_result
