"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 4, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 16, 24, 24

    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = delta
    # We seek rank-r decomposition: T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # i.e., T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]

    def build_tensor():
        """Build the multiplication tensor as a 3D array."""
        T = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    def verify_decomposition(U, V, W, T, tol=1e-6):
        """Verify that U,V,W decompose T."""
        # T[a,b,c] should equal sum_t U[a,t]*V[b,t]*W[c,t]
        # Check a sample of entries
        rank = U.shape[1]
        # Use matrix multiplication trick: for each t, outer product
        # T_approx[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,c]
        # = sum_t (U[:,t] * W[:,t]) @ V[:,t]^T reshaped... 
        # Simpler: check all entries via einsum
        T_approx = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def try_rank(r, T, max_iter=500, tol=1e-6, seed=42):
        """Try to find rank-r decomposition using iterative optimization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with random matrices
        U = rng.randn(nm, r) * 0.1
        V = rng.randn(mp, r) * 0.1
        W = rng.randn(pn, r) * 0.1
        
        # Normalize initial columns
        for t in range(r):
            norm_u = np.linalg.norm(U[:, t])
            norm_v = np.linalg.norm(V[:, t])
            norm_w = np.linalg.norm(W[:, t])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                U[:, t] /= scale
                V[:, t] /= scale
                W[:, t] /= scale
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Compute gradient-based updates
            # T_approx[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # dT/du[a,t] = V[b,t]*W[c,t] for the (a,b,c) entry
            # Gradient of ||T - T_approx||^2 w.r.t. U[a,t]:
            # = -2 * sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) * V[b,t] * W[c,t]
            
            # Efficient computation using matrix operations
            # T_approx = sum_t U[:,t] * V[:,t] * W[:,t] (outer product)
            
            # For each t, the contribution is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
            # Gradient w.r.t. U[:,t]: -2 * (T - T_approx) contracted with V[:,t] and W[:,t]
            
            # Compute T_approx
            T_approx = np.einsum('at,bt,ct->abc', U, V, W)
            err = T - T_approx
            
            # Gradient w.r.t. U: dL/du[a,t] = -2 * sum_{b,c} err[a,b,c] * V[b,t] * W[c,t]
            # = -2 * (err @ (V[:,t] * W[:,t]))[a]  ... not quite
            # Actually: sum_{b,c} err[a,b,c] * V[b,t] * W[c,t]
            # = sum_b (sum_c err[a,b,c] * W[c,t]) * V[b,t]
            # = sum_b (err[a,:,:] * W[:,t])[b] * V[b,t]  -- no
            # Let's do it step by step:
            # G_U[a,t] = sum_{b,c} err[a,b,c] * V[b,t] * W[c,t]
            # = sum_b V[b,t] * (sum_c err[a,b,c] * W[c,t])
            # = sum_b V[b,t] * (err[a,:,:] dot W[:,t])
            
            # Use einsum for efficiency
            # G_U = -2 * einsum('abc,bt,ct->at', err, V, W)
            # G_V = -2 * einsum('abc,at,ct->bt', err, U, W)  
            # G_W = -2 * einsum('abc,at,bt->ct', err, U, V)
            
            G_U = np.einsum('abc,bt,ct->at', err, V, W)
            G_V = np.einsum('abc,at,ct->bt', err, U, W)
            G_W = np.einsum('abc,at,bt->ct', err, U, V)
            
            # Learning rate with decay
            lr = 0.01 / (1 + it * 0.001)
            
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
            
            # Renormalize columns to prevent drift
            for t in range(r):
                norm_u = np.linalg.norm(U[:, t])
                norm_v = np.linalg.norm(V[:, t])
                norm_w = np.linalg.norm(W[:, t])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    scale = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                    U[:, t] /= scale
                    V[:, t] /= scale
                    W[:, t] /= scale
            
            # Check convergence
            T_approx = np.einsum('at,bt,ct->abc', U, V, W)
            cur_err = np.linalg.norm(T - T_approx)
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if cur_err < tol:
                break
        
        # Final verification
        if verify_decomposition(best_U, best_V, best_W, T, tol=1e-5):
            # Convert to integer/half-integer if possible
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    def strassen_decomposition():
        """Fallback: recursive Strassen, rank 77."""
        def build_2x2x1():
            rank = 4
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(2)]
            W = [[0]*rank for _ in range(2)]
            U[0][0]=1; V[0][0]=1; W[0][0]=1
            U[1][1]=1; V[1][1]=1; W[0][1]=1
            U[2][2]=1; V[0][2]=1; W[1][2]=1
            U[3][3]=1; V[1][3]=1; W[1][3]=1
            return U, V, W

        def build_2x2x2_strassen():
            rank = 7
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(4)]
            W = [[0]*rank for _ in range(4)]
            # M1
            U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
            # M2
            U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
            # M3
            U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
            # M4
            U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
            # M5
            U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
            # M6
            U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
            # M7
            U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
            return U, V, W

        def build_2x2x3():
            rank = 11
            U = [[0]*rank for _ in range(4)]
            V = [[0]*rank for _ in range(6)]
            W = [[0]*rank for _ in range(6)]
            U1, V1, W1 = build_2x2x1()
            for c in range(4):
                for r in range(4):
                    U[r][c] += U1[r][c]
                for r in range(2):
                    V[r*3][c] += V1[r][c]
                for r in range(2):
                    W[r][c] += W1[r][c]
            U2, V2, W2 = build_2x2x2_strassen()
            base = 4
            v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
            w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
            for c in range(7):
                for r in range(4):
                    U[r][base+c] += U2[r][c]
                for r in range(4):
                    V[v2_map[r]][base+c] += V2[r][c]
                for r in range(4):
                    W[w2_map[r]][base+c] += W2[r][c]
            return U, V, W

        def build_4x4x6():
            n, m, p = 4, 4, 6
            U_sub, V_sub, W_sub = build_2x2x3()
            rank_sub = len(U_sub[0])
            total_rank = 7 * rank_sub
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            terms = [
                {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
                {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
                {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
                {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
                {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
                {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
                {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
            ]
            u_offset = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
            v_offset = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
            w_offset = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

            for t_idx, term in enumerate(terms):
                base_col = t_idx * rank_sub
                for c in range(rank_sub):
                    for u_local in range(4):
                        u_val = U_sub[u_local][c]
                        if u_val == 0: continue
                        i_local = u_local // 2
                        j_local = u_local % 2
                        for block, coeff in term['A']:
                            ro, co = u_offset[block]
                            g_u = (i_local + ro) * m + (j_local + co)
                            U[g_u][base_col + c] += u_val * coeff
                    for v_local in range(6):
                        v_val = V_sub[v_local][c]
                        if v_val == 0: continue
                        j_local = v_local // 3
                        k_local = v_local % 3
                        for block, coeff in term['B']:
                            ro, co = v_offset[block]
                            g_v = (j_local + ro) * p + (k_local + co)
                            V[g_v][base_col + c] += v_val * coeff
                    for w_local in range(6):
                        w_val = W_sub[w_local][c]
                        if w_val == 0: continue
                        k_local = w_local // 2
                        i_local = w_local % 2
                        for block, coeff in term['C']:
                            ro, co = w_offset[block]
                            g_w = (k_local + ro) * n + (i_local + co)
                            W[g_w][base_col + c] += w_val * coeff
            return U, V, W

        U, V, W = build_4x4x6()
        return U, V, W

    # Strategy: try to find a lower-rank decomposition
    # Known results suggest rank 60 might be achievable
    T = build_tensor()
    
    best_result = None
    best_rank = 77  # Strassen baseline
    
    # Try various ranks from low to high
    for r in [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for this rank
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 5:
                break
            
            U, V, W, err = try_rank(r, T, max_iter=200, tol=1e-6, seed=seed)
            if U is not None:
                best_result = (U, V, W, r)
                best_rank = r
                break
        
        if best_result is not None:
            break
    
    # If no better result found, use Strassen
    if best_result is None:
        U_str, V_str, W_str = strassen_decomposition()
        return {"U": U_str, "V": V_str, "W": W_str}
    else:
        U, V, W, r = best_result
        # Convert numpy arrays to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
