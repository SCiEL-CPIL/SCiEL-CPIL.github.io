import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build rank-44 solution
    U44, V44, W44 = build_rank44(n, m, p)
    
    # Precompute triples for fast checking
    triples = np.array([(i * m + j, j * p + k, k * n + i) 
                        for i in range(n) for j in range(m) for k in range(p)])
    
    # Try to find rank-43 by removing one column and re-solving
    # We will try each column removal, and for each, run a more robust ALS + local search.
    # To improve the chance of finding rank-43, we will:
    # 1. Use a better initialization for ALS (e.g., random or based on SVD of the tensor).
    # 2. Use a more robust local search (e.g., simulated annealing or tabu search).
    # 3. Try multiple random restarts if time permits.
    
    # First, try the current approach (removing each column and running ALS)
    for col in range(44):
        if time.time() > deadline - 10:
            break
        result = try_reduce_to_rank(U44, V44, W44, col, n, m, p, deadline, triples)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If the current approach fails, try a more aggressive search with random restarts
    # We will try to find a rank-43 decomposition from scratch using random initialization.
    
    # Define a function to check if a decomposition is valid
    def check_decomposition(U, V, W, triples):
        U_sel = U[triples[:, 0]]
        V_sel = V[triples[:, 1]]
        W_sel = W[triples[:, 2]]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1)) < 1e-6
    
    # Define a function to round to half-integers
    def round_to_half_int(X):
        return np.round(X * 2) / 2
    
    # Define a function to run ALS with random restarts
    def run_als_random(U_init, V_init, W_init, n, m, p, triples, deadline, max_restarts=5):
        R = U_init.shape[1]
        nU = n * m
        nV = m * p
        nW = p * n
        
        best_U = None
        best_V = None
        best_W = None
        best_err = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values if restart > 0
            if restart == 0:
                U = U_init.copy()
                V = V_init.copy()
                W = W_init.copy()
            else:
                U = np.random.randn(nU, R) * 0.5
                V = np.random.randn(nV, R) * 0.5
                W = np.random.randn(nW, R) * 0.5
            
            # ALS iterations
            max_als = 30
            for it in range(max_als):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                            U[i * m + j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                            V[j * p + k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                            W[k * n + i, :] = sol
                        except:
                            pass
                
                # Check if converged
                vals = np.sum(U[triples[:, 0]] * V[triples[:, 1]] * W[triples[:, 2]], axis=1)
                err = np.max(np.abs(vals - 1))
                if err < 1e-8:
                    break
            
            # Round to half-integers
            U_r = round_to_half_int(U)
            V_r = round_to_half_int(V)
            W_r = round_to_half_int(W)
            
            # Check if valid
            vals = np.sum(U_r[triples[:, 0]] * V_r[triples[:, 1]] * W_r[triples[:, 2]], axis=1)
            err = np.max(np.abs(vals - 1))
            
            if err < best_err:
                best_err = err
                best_U = U_r
                best_V = V_r
                best_W = W_r
            
            if err < 1e-6:
                return U_r, V_r, W_r
        
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try to find rank-43 from scratch using random restarts
    R = 43
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Initialize with the rank-44 solution, removing a random column
    for col in range(44):
        if time.time() > deadline - 2:
            break
        U_init = np.delete(U44, col, axis=1).copy()
        V_init = np.delete(V44, col, axis=1).copy()
        W_init = np.delete(W44, col, axis=1).copy()
        
        result = run_als_random(U_init, V_init, W_init, n, m, p, triples, deadline, max_restarts=3)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return rank-44
    return {"U": U44.tolist(), "V": V44.tolist(), "W": W44.tolist()}


def build_rank44(n, m, p):
    rank = 44
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)

    return U, V, W


def vectorized_check(U, V, W, triples):
    """Check all 48 tensor entries at once. Returns array of 48 values."""
    n, m, p = DIMENSIONS
    a_idx = triples[:, 0]
    b_idx = triples[:, 1]
    c_idx = triples[:, 2]
    
    U_sel = U[a_idx]      # (48, R)
    V_sel = V[b_idx]      # (48, R)
    W_sel = W[c_idx]      # (48, R)
    
    # Element-wise product then sum over rank
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    return vals


def try_reduce_to_rank(U, V, W, remove_col, n, m, p, deadline, triples):
    R = U.shape[1] - 1  # 43
    U43 = np.delete(U, remove_col, axis=1).copy()
    V43 = np.delete(V, remove_col, axis=1).copy()
    W43 = np.delete(W, remove_col, axis=1).copy()
    
    # Precompute index arrays for ALS
    nU = n * m
    nV = m * p
    nW = p * n
    
    # ALS iterations
    max_als = 50
    for it in range(max_als):
        if time.time() > deadline - 15:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, R))
                for k in range(p):
                    M[k, :] = V43[j * p + k, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U43[i * m + j, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, R))
                for i in range(n):
                    M[i, :] = U43[i * m + j, :] * W43[k * n + i, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V43[j * p + k, :] = sol
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U43[i * m + j, :] * V43[j * p + k, :]
                try:
                    sol, *_ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W43[k * n + i, :] = sol
                except:
                    pass
        
        # Check if converged
        vals = vectorized_check(U43, V43, W43, triples)
        if np.max(np.abs(vals - 1)) < 1e-8:
            break
    
    # Round to half-integers
    U43 = np.round(U43 * 2) / 2
    V43 = np.round(V43 * 2) / 2
    W43 = np.round(W43 * 2) / 2
    
    vals = vectorized_check(U43, V43, W43, triples)
    if np.max(np.abs(vals - 1)) < 1e-6:
        return U43, V43, W43
    
    # Local search: try flipping individual half-integer values
    U_part = [np.where(triples[:, 0] == a)[0] for a in range(nU)]
    V_part = [np.where(triples[:, 1] == b)[0] for b in range(nV)]
    W_part = [np.where(triples[:, 2] == c)[0] for c in range(nW)]
    
    def fast_check_partial(U, V, W, tri_indices):
        """Check only specific triple indices."""
        if len(tri_indices) == 0:
            return True
        a_idx = triples[tri_indices, 0]
        b_idx = triples[tri_indices, 1]
        c_idx = triples[tri_indices, 2]
        U_sel = U[a_idx]
        V_sel = V[b_idx]
        W_sel = W[c_idx]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.max(np.abs(vals - 1)) < 1e-6
    
    # Greedy local search
    for ls_iter in range(200):
        if time.time() > deadline - 2:
            break
        
        vals = vectorized_check(U43, V43, W43, triples)
        max_err = np.max(np.abs(vals - 1))
        if max_err < 1e-6:
            return U43, V43, W43
        
        # Find the most violated triple
        worst_tri = np.argmax(np.abs(vals - 1))
        a, b, c = triples[worst_tri]
        
        # Try flipping entries in the worst triple's rows
        candidates = []
        
        # U row a
        for r in range(R):
            candidates.append(('U', a, r, 0.5))
            candidates.append(('U', a, r, -0.5))
        
        # V row b
        for r in range(R):
            candidates.append(('V', b, r, 0.5))
            candidates.append(('V', b, r, -0.5))
        
        # W row c
        for r in range(R):
            candidates.append(('W', c, r, 0.5))
            candidates.append(('W', c, r, -0.5))
        
        # Shuffle for diversity
        np.random.shuffle(candidates)
        
        improved = False
        for mat_name, row, col_r, delta in candidates:
            if mat_name == 'U':
                old = U43[row, col_r]
                U43[row, col_r] = old + delta
                if fast_check_partial(U43, V43, W43, U_part[row]):
                    # Also verify full
                    vals_new = vectorized_check(U43, V43, W43, triples)
                    if np.max(np.abs(vals_new - 1)) < 1e-6:
                        return U43, V43, W43
                U43[row, col_r] = old
            elif mat_name == 'V':
                old = V43[row, col_r]
                V43[row, col_r] = old + delta
                if fast_check_partial(U43, V43, W43, V_part[row]):
                    vals_new = vectorized_check(U43, V43, W43, triples)
                    if np.max(np.abs(vals_new - 1)) < 1e-6:
                        return U43, V43, W43
                V43[row, col_r] = old
            else:  # W
                old = W43[row, col_r]
                W43[row, col_r] = old + delta
                if fast_check_partial(U43, V43, W43, W_part[row]):
                    vals_new = vectorized_check(U43, V43, W43, triples)
                    if np.max(np.abs(vals_new - 1)) < 1e-6:
                        return U43, V43, W43
                W43[row, col_r] = old
    
    return None
