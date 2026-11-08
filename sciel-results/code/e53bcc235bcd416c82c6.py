import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def snap_to_half(arr):
        return np.round(arr * 2) / 2
    
    def is_half_integer(arr, tol=1e-6):
        doubled = arr * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)
    
    def try_snap_validate(U, V, W):
        # Try half-integers
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    def solve_linear_system_for_rank_reduction(U46, V46, W46, cols_to_keep, target_rank):
        """
        Try to find a rank reduction by fixing V to canonical patterns and solving for U, W.
        cols_to_keep: list of column indices to keep from the rank-46 solution
        We'll try to absorb the removed columns' contributions into the kept ones.
        """
        kept = cols_to_keep
        removed = [c for c in range(46) if c not in kept]
        
        if len(kept) != target_rank:
            return None, None, None
        
        # The tensor must equal sum over kept columns
        # T = sum_{r in kept} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We have freedom to adjust U[:,r] and W[:,r] for r in kept
        # V[:,r] is fixed to V46[:,r]
        
        # Build linear system: for each (a,b,c) where T[a,b,c] != 0,
        # sum_{r in kept} U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        # This is bilinear in U and W, so it's not linear.
        # Instead, we can fix W to W46[:,r] and solve for U, or vice versa.
        
        # Strategy: Fix W to W46[:,r] for r in kept, solve for U
        # T[a,b,c] = sum_{r in kept} U[a,r] * V[b,r] * W46[c,r]
        # For fixed b,c: sum_{r in kept} U[a,r] * (V[b,r] * W46[c,r]) = T[a,b,c]
        # This is linear in U[a,r] for each (a,b,c)
        
        # Actually, let's try a different approach: use the structure of the problem.
        # For matrix multiplication, each tensor element T[a,b,c] = 1 if and only if
        # the indices correspond to valid (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
        
        # Let's try: fix V columns to be the same as rank-46, and solve for U and W
        # using least squares with regularization, then check if half-integer
        
        R = target_rank
        V_fixed = V46[:, kept].copy()
        
        # For each kept column r, we need U[:,r] and W[:,r] such that
        # sum_r U[:,r] ⊗ V_fixed[:,r] ⊗ W[:,r] = T
        
        # This is equivalent to: for each r, U[:,r] ⊗ W[:,r] must satisfy
        # sum_r (U[:,r] ⊗ W[:,r]) * V_fixed[:,r][b] = T[:,b,:]
        
        # Let's use an iterative approach: alternate between solving for U and W
        np.random.seed(42)
        U = U46[:, kept].copy() + np.random.randn(U_dim, R) * 0.01
        W = W46[:, kept].copy() + np.random.randn(W_dim, R) * 0.01
        
        for iteration in range(200):
            if time.time() > deadline - 15:
                break
            
            # Solve for U given W
            # For each b: sum_r U[:,r] * (V_fixed[b,r] * W[:,r]) = T[:,b,:]
            # Reshape: U[:,r] * W[:,r] is outer product, but we need
            # sum_r U[a,r] * V_fixed[b,r] * W[c,r] = T[a,b,c]
            
            # For fixed b,c: sum_r U[a,r] * (V_fixed[b,r] * W[c,r]) = T[a,b,c]
            # This is a linear system in U[a,:] for each (b,c)
            
            U_new = U.copy()
            for a in range(U_dim):
                # For each a, we need U[a,:] such that for all (b,c):
                # sum_r U[a,r] * V_fixed[b,r] * W[c,r] = T[a,b,c]
                # Let M[a] be a matrix where M[a][bc, r] = V_fixed[b,r] * W[c,r]
                # Then M[a] @ U[a,:] = T[a,:,:].reshape(-1)
                
                # Build M
                M = np.zeros((V_dim * W_dim, R))
                for b in range(V_dim):
                    for c in range(W_dim):
                        M[b * W_dim + c, :] = V_fixed[:, b] * W[:, c]
                
                rhs = T[a, :, :].reshape(-1)
                
                # Solve least squares
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    U_new[a, :] = sol
                except:
                    pass
            
            U = U_new
            
            # Solve for W given U
            W_new = W.copy()
            for c in range(W_dim):
                # For each c, we need W[c,:] such that for all (a,b):
                # sum_r U[a,r] * V_fixed[b,r] * W[c,r] = T[a,b,c]
                # Let M[c] be a matrix where M[c][ab, r] = U[a,r] * V_fixed[b,r]
                # Then M[c] @ W[c,:] = T[:,:,c].reshape(-1)
                
                M = np.zeros((U_dim * V_dim, R))
                for a in range(U_dim):
                    for b in range(V_dim):
                        M[a * V_dim + b, :] = U[:, a] * V_fixed[:, b]
                
                rhs = T[:, :, c].reshape(-1)
                
                try:
                    sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    W_new[:, c] = sol
                except:
                    pass
            
            W = W_new
            
            # Check convergence
            prod = np.einsum('ar,br,cr->abc', U, V_fixed, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < 1e-10:
                break
        
        # Check if solution is half-integer
        U_s = snap_to_half(U)
        W_s = snap_to_half(W)
        V_s = snap_to_half(V_fixed)
        
        if validate(U_s, V_s, W_s):
            return U_s, V_s, W_s
        
        # Try integer
        U_i = np.round(U)
        W_i = np.round(W)
        V_i = np.round(V_fixed)
        if validate(U_i, V_i, V_i):
            return U_i, V_i, W_i
        
        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to standard rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_result = None
    best_rank = 46
    
    # Compute column importance
    col_contributions = []
    for c in range(46):
        contrib = np.einsum('a,b,c->abc', U46[:, c], V46[:, c], W46[:, c], optimize=True)
        col_contributions.append(np.sum(contrib * contrib))
    
    support_U = [set(np.nonzero(U46[:, c])[0]) for c in range(46)]
    support_V = [set(np.nonzero(V46[:, c])[0]) for c in range(46)]
    support_W = [set(np.nonzero(W46[:, c])[0]) for c in range(46)]
    
    col_scores = []
    for c in range(46):
        support_size = len(support_U[c]) + len(support_V[c]) + len(support_W[c])
        overlap_count = 0
        for c2 in range(46):
            if c2 == c:
                continue
            overlap = len(support_U[c] & support_U[c2]) + len(support_V[c] & support_V[c2]) + len(support_W[c] & support_W[c2])
            if overlap > 2:
                overlap_count += 1
        score = col_contributions[c] * 0.1 + support_size * 0.5 + overlap_count * 0.3
        col_scores.append(score)
    
    col_order = np.argsort(col_scores)
    
    # Strategy 1: Try removing single columns (rank 45)
    for attempt in range(min(20, 46)):
        if time.time() > deadline - 60:
            break
            
        col_to_zero = col_order[attempt]
        kept = [i for i in range(46) if i != col_to_zero]
        
        U_opt, V_opt, W_opt = solve_linear_system_for_rank_reduction(U46, V46, W46, kept, 45)
        
        if U_opt is not None:
            best_result = (U_opt, V_opt, W_opt)
            best_rank = 45
            break
    
    # Strategy 2: If rank 45 failed, try removing pairs (rank 44)
    if best_result is None and time.time() < deadline - 70:
        pair_candidates = []
        for i in range(len(col_order)):
            for j in range(i+1, len(col_order)):
                c1 = col_order[i]
                c2 = col_order[j]
                overlap = (len(support_U[c1] & support_U[c2]) + 
                          len(support_V[c1] & support_V[c2]) + 
                          len(support_W[c1] & support_W[c2]))
                combined_score = col_scores[i] + col_scores[j]
                pair_candidates.append((overlap, combined_score, c1, c2))
        
        pair_candidates.sort(key=lambda x: (-x[0], x[1]))
        
        for idx in range(min(15, len(pair_candidates))):
            if time.time() > deadline - 65:
                break
            _, _, c1, c2 = pair_candidates[idx]
            kept = [i for i in range(46) if i not in (c1, c2)]
            
            U_opt, V_opt, W_opt = solve_linear_system_for_rank_reduction(U46, V46, W46, kept, 44)
            
            if U_opt is not None:
                best_result = (U_opt, V_opt, W_opt)
                best_rank = 44
                break
    
    # Strategy 3: Try rank 43
    if best_result is None and time.time() < deadline - 75:
        triple_candidates = []
        for i in range(min(12, len(col_order))):
            for j in range(i+1, min(12, len(col_order))):
                for k in range(j+1, min(12, len(col_order))):
                    c1, c2, c3 = col_order[i], col_order[j], col_order[k]
                    overlap = (len(support_U[c1] & support_U[c2]) + len(support_V[c1] & support_V[c2]) + len(support_W[c1] & support_W[c2]) +
                              len(support_U[c1] & support_U[c3]) + len(support_V[c1] & support_V[c3]) + len(support_W[c1] & support_W[c3]) +
                              len(support_U[c2] & support_U[c3]) + len(support_V[c2] & support_V[c3]) + len(support_W[c2] & support_W[c3]))
                    triple_candidates.append((overlap, c1, c2, c3))
        
        triple_candidates.sort(reverse=True)
        
        for idx in range(min(10, len(triple_candidates))):
            if time.time() > deadline - 70:
                break
            _, c1, c2, c3 = triple_candidates[idx]
            kept = [i for i in range(46) if i not in (c1, c2, c3)]
            
            U_opt, V_opt, W_opt = solve_linear_system_for_rank_reduction(U46, V46, W46, kept, 43)
            
            if U_opt is not None:
                best_result = (U_opt, V_opt, W_opt)
                best_rank = 43
                break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
