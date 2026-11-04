import numpy as np
import time
from fractions import Fraction
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Build the matrix multiplication tensor T[u,v,w]
    # u = i*m + j (input: row i, col j of A), v = j*p + k (input: row j, col k of B), w = k*n + i (output: row i, col k)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    schoolbook_rank = n * m * p
    min_rank = max(n * p, m * p, n * m)
    
    # Schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1
                V_sb[v, col] = 1
                W_sb[w, col] = 1
                col += 1
    
    def check_solution(U, V, W, tol=1e-9):
        T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
        err = np.linalg.norm(T_calc - T)
        return err < tol
    
    def snap_to_half_int(U, V, W):
        U_int = np.round(U * 2) / 2.0
        V_int = np.round(V * 2) / 2.0
        W_int = np.round(W * 2) / 2.0
        return U_int, V_int, W_int
    
    def snap_to_int(U, V, W):
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        return U_int, V_int, W_int
    
    def rational_reconstruct(vals, max_den=32):
        """Try to find rational numbers with small denominators close to vals."""
        result = []
        for v in vals:
            best_frac = None
            best_diff = float('inf')
            for den in range(1, max_den + 1):
                num = round(v * den)
                frac_val = Fraction(num, den)
                diff = abs(float(frac_val) - v)
                if diff < best_diff:
                    best_diff = diff
                    best_frac = float(frac_val)
            result.append(best_frac)
        return np.array(result)
    
    def polish_solution(U, V, W, max_iters=200, tol=1e-12):
        """Refine solution with alternating least squares until convergence."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update W
            A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
            T_W = T.reshape(n * m * m * p, p * n)
            try:
                W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                W = W_T.T
            except:
                continue
            
            # Update U
            A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
            T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
            try:
                U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                U = U_T.T
            except:
                continue
            
            # Update V
            A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
            T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
            try:
                V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                V = V_T.T
            except:
                continue
            
            if it % 20 == 0:
                T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                err = np.linalg.norm(T_calc - T)
                if err < tol:
                    break
        
        return U, V, W
    
    def try_strassen_hierarchy(r):
        """
        Structured tensor network ansatz based on Strassen algorithm hierarchy.
        
        For 4x5x5, we can think of this as multiplying blocks.
        We use a hierarchical approach where we decompose the problem into
        smaller sub-problems and use Strassen-like cancellations.
        
        Key idea: Use linear combinations of entries to create cancellations.
        For example, to compute (a+b)(c+d) - a*c - b*d = a*d + b*c, 
        we save one multiplication.
        
        We build the decomposition using a structured pattern based on
        partitioning the matrices into blocks and applying Strassen recursion.
        """
        best_result = None
        best_err = float('inf')
        
        # Define Strassen-like patterns for different block sizes
        # For 2x2x2: rank 7 instead of 8
        # For 3x3x3: rank 23 instead of 27 (or 21 with advanced methods)
        
        # We'll use a combination of structured initializations based on
        # Strassen patterns applied to sub-blocks
        
        num_patterns = min(15, max(3, (int(deadline - time.time()) * 2) // 10))
        
        for pattern_idx in range(num_patterns):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(2000 + pattern_idx)
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Strategy: Use different structured patterns
            if pattern_idx % 3 == 0:
                # Pattern 1: Block-based Strassen
                # Divide into 2x2 blocks where possible
                # For 4x5x5, think of A as 2x2 blocks of size 2x2.5, etc.
                
                # Create structured entries based on block structure
                for t in range(r):
                    # Assign based on a structured pattern
                    block_row = t // 5
                    block_col = t % 5
                    
                    # U entries: based on row-block and col-block
                    for i in range(n):
                        for j in range(m):
                            if (i // 2) == (block_row // 2) and (j // 2) == (block_col // 2):
                                U[i * m + j, t] = np.random.choice([-1, 1])
                    
                    # V entries
                    for j in range(m):
                        for k in range(p):
                            if (j // 2) == (block_col // 2) and (k // 2) == (block_row % 2):
                                V[j * p + k, t] = np.random.choice([-1, 1])
                    
                    # W entries
                    for k in range(p):
                        for i in range(n):
                            if (k // 2) == (block_row % 2) and (i // 2) == (block_row // 2):
                                W[k * n + i, t] = np.random.choice([-1, 1])
                
                # Add some noise to break symmetry
                U += 0.1 * np.random.randn(n * m, r)
                V += 0.1 * np.random.randn(m * p, r)
                W += 0.1 * np.random.randn(p * n, r)
                
            elif pattern_idx % 3 == 1:
                # Pattern 2: Diagonal Strassen
                # Use diagonal patterns with cancellations
                
                for t in range(r):
                    # Create a sparse pattern with signs for cancellation
                    n_ones = np.random.randint(2, 5)
                    
                    # U: select random positions with signs
                    for _ in range(n_ones):
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        sign = np.random.choice([-1, 1])
                        U[i * m + j, t] = sign
                    
                    # V: correlated with U
                    for _ in range(n_ones):
                        j = np.random.randint(m)
                        k = np.random.randint(p)
                        sign = np.random.choice([-1, 1])
                        V[j * p + k, t] = sign
                    
                    # W: correlated
                    for _ in range(n_ones):
                        k = np.random.randint(p)
                        i = np.random.randint(n)
                        sign = np.random.choice([-1, 1])
                        W[k * n + i, t] = sign
                
                # Add structured noise
                U += 0.05 * np.random.randn(n * m, r)
                V += 0.05 * np.random.randn(m * p, r)
                W += 0.05 * np.random.randn(p * n, r)
                
            else:
                # Pattern 3: Hierarchical Strassen
                # Use a tree-like structure for cancellations
                
                for t in range(r):
                    # Level 1: coarse block
                    coarse_i = t // (m * p)
                    coarse_j = (t // p) % m
                    coarse_k = t % p
                    
                    # U: entries in the coarse block
                    for di in range(min(2, n - coarse_i)):
                        for dj in range(min(2, m - coarse_j)):
                            if coarse_i + di < n and coarse_j + dj < m:
                                U[(coarse_i + di) * m + (coarse_j + dj), t] = \
                                    np.random.choice([-1, 1])
                    
                    # V: correlated
                    for dj in range(min(2, m - coarse_j)):
                        for dk in range(min(2, p - coarse_k)):
                            if coarse_j + dj < m and coarse_k + dk < p:
                                V[(coarse_j + dj) * p + (coarse_k + dk), t] = \
                                    np.random.choice([-1, 1])
                    
                    # W: correlated
                    for dk in range(min(2, p - coarse_k)):
                        for di in range(min(2, n - coarse_i)):
                            if coarse_k + dk < p and coarse_i + di < n:
                                W[(coarse_k + dk) * n + (coarse_i + di), t] = \
                                    np.random.choice([-1, 1])
                
                # Add small noise
                U += 0.05 * np.random.randn(n * m, r)
                V += 0.05 * np.random.randn(m * p, r)
                W += 0.05 * np.random.randn(p * n, r)
            
            # Run ALS optimization
            for it in range(80):
                if time.time() > deadline - 5:
                    break
                
                # Update W
                A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
                T_W = T.reshape(n * m * m * p, p * n)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                    W = W_T.T
                except:
                    continue
                
                # Update U
                A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
                T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                    U = U_T.T
                except:
                    continue
                
                # Update V
                A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
                T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                    V = V_T.T
                except:
                    continue
                
                if it % 10 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-7:
                        # Polish and try to snap
                        U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=100)
                        T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
                        err_pol = np.linalg.norm(T_calc_pol - T)
                        
                        if err_pol < 1e-9:
                            # Try various snapping strategies
                            for snap_func in [snap_to_half_int, snap_to_int]:
                                U_s, V_s, W_s = snap_func(U_pol, V_pol, W_pol)
                                if check_solution(U_s, V_s, W_s):
                                    return U_s, V_s, W_s
                            
                            # Try rational reconstruction
                            U_rat = rational_reconstruct(U_pol.flatten()).reshape(U_pol.shape)
                            V_rat = rational_reconstruct(V_pol.flatten()).reshape(V_pol.shape)
                            W_rat = rational_reconstruct(W_pol.flatten()).reshape(W_pol.shape)
                            if check_solution(U_rat, V_rat, W_rat):
                                return U_rat, V_rat, W_rat
                        
                        if err_pol < best_err:
                            best_err = err_pol
                            best_result = (U_pol.copy(), V_pol.copy(), W_pol.copy())
        
        # If we have a good result, try to polish it further
        if best_result is not None and best_err < 1e-6:
            U, V, W = best_result
            U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=150)
            T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
            err_pol = np.linalg.norm(T_calc_pol - T)
            
            if err_pol < 1e-9:
                for snap_func in [snap_to_half_int, snap_to_int]:
                    U_s, V_s, W_s = snap_func(U_pol, V_pol, W_pol)
                    if check_solution(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                
                U_rat = rational_reconstruct(U_pol.flatten()).reshape(U_pol.shape)
                V_rat = rational_reconstruct(V_pol.flatten()).reshape(V_pol.shape)
                W_rat = rational_reconstruct(W_pol.flatten()).reshape(W_pol.shape)
                if check_solution(U_rat, V_rat, W_rat):
                    return U_rat, V_rat, W_rat
        
        return None
    
    def solve_als(r, max_iters=100, restarts=10, seed_offset=0):
        """Alternating least squares for rank-r tensor decomposition."""
        best_result = None
        best_err = float('inf')
        
        for restart in range(restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + seed_offset + restart)
            
            # Random initialization
            U = np.random.uniform(-1, 1, (n * m, r))
            V = np.random.uniform(-1, 1, (m * p, r))
            W = np.random.uniform(-1, 1, (p * n, r))
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Update W
                A_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(n * m * m * p, r)
                T_W = T.reshape(n * m * m * p, p * n)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(A_UV, T_W, rcond=None)
                    W = W_T.T
                except:
                    continue

                # Update U
                A_VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(m * p * p * n, r)
                T_U = T.transpose(1, 2, 0).reshape(m * p * p * n, n * m)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(A_VW, T_U, rcond=None)
                    U = U_T.T
                except:
                    continue

                # Update V
                A_UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(n * m * p * n, r)
                T_V = T.transpose(0, 2, 1).reshape(n * m * p * n, m * p)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(A_UW, T_V, rcond=None)
                    V = V_T.T
                except:
                    continue

                if it % 10 == 0:
                    T_calc = np.einsum('ut,vt,wt->uvw', U, V, W, optimize=True)
                    err = np.linalg.norm(T_calc - T)
                    if err < best_err:
                        best_err = err
                        best_result = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-7:
                        U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=80)
                        T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
                        err_pol = np.linalg.norm(T_calc_pol - T)
                        
                        if err_pol < 1e-9:
                            for snap_func in [snap_to_half_int, snap_to_int]:
                                U_s, V_s, W_s = snap_func(U_pol, V_pol, W_pol)
                                if check_solution(U_s, V_s, W_s):
                                    return U_s, V_s, W_s
                            
                            U_rat = rational_reconstruct(U_pol.flatten()).reshape(U_pol.shape)
                            V_rat = rational_reconstruct(V_pol.flatten()).reshape(V_pol.shape)
                            W_rat = rational_reconstruct(W_pol.flatten()).reshape(W_pol.shape)
                            if check_solution(U_rat, V_rat, W_rat):
                                return U_rat, V_rat, W_rat
                        
                        if err_pol < best_err:
                            best_err = err_pol
                            best_result = (U_pol.copy(), V_pol.copy(), W_pol.copy())

        if best_result is not None and best_err < 1e-6:
            U, V, W = best_result
            U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iters=100)
            T_calc_pol = np.einsum('ut,vt,wt->uvw', U_pol, V_pol, W_pol, optimize=True)
            err_pol = np.linalg.norm(T_calc_pol - T)
            
            if err_pol < 1e-9:
                for snap_func in [snap_to_half_int, snap_to_int]:
                    U_s, V_s, W_s = snap_func(U_pol, V_pol, W_pol)
                    if check_solution(U_s, V_s, W_s):
                        return U_s, V_s, W_s
                
                U_rat = rational_reconstruct(U_pol.flatten()).reshape(U_pol.shape)
                V_rat = rational_reconstruct(V_pol.flatten()).reshape(V_pol.shape)
                W_rat = rational_reconstruct(W_pol.flatten()).reshape(W_pol.shape)
                if check_solution(U_rat, V_rat, W_rat):
                    return U_rat, V_rat, W_rat
        
        return None
    
    # Search ranks from lower bound up to schoolbook
    # For 4x5x5, try to find rank below 100
    search_start = max(min_rank, 40)
    
    # First try Strassen hierarchy for promising ranks
    for r in range(search_start, min(schoolbook_rank + 1, 80)):
        if time.time() > deadline - 15:
            break
        
        # Try Strassen-like structured approach first
        result = try_strassen_hierarchy(r)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        if time.time() > deadline - 10:
            break
        
        # Then try standard ALS
        restarts = 8 if r < 55 else 5
        max_iters = 80 if r < 55 else 50
        
        result = solve_als(r, max_iters=max_iters, restarts=restarts, seed_offset=0)
        if result:
            U_int, V_int, W_int = result
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
