import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def _als_tensor_decomposition(n, m, p, max_rank, deadline, seed=42):
    """
    Use Alternating Least Squares to find a low-rank decomposition of the
    matrix multiplication tensor T[i,j,k] = delta_{sum} for C = A*B.
    
    The tensor is: T_{(i,j),(j,k),(k,i)} = 1 if indices match, 0 otherwise.
    We want U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = T.
    
    This is equivalent to finding rank-r factor matrices such that:
    sum_r U[:,r] V[:,r]^T (elementwise) ... actually let's think in terms of
    the 3-way tensor unfolding.
    
    The multiplication tensor has entries:
    T[a, b, c] where a = (i,j) in [0,nm), b = (j,k) in [0,mp), c = (k,i) in [0,pn)
    T[a,b,c] = 1 if the three indices are consistent (j matches between a and b,
    k matches between b and c, i matches between c and a), else 0.
    
    For a rank-r decomposition:
    T[a,b,c] = sum_{r=0}^{r-1} U[a,r] * V[b,r] * W[c,r]
    
    ALS: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W.
    Each solve is a linear system.
    """
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    best_result = None
    best_rank = max_rank + 1
    
    # Try ranks from low to high
    for rank in range(7, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 15 if rank <= 40 else 8
        max_iters = 200 if rank <= 30 else 100
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with random small values
            U = rng.randn(nm, rank) * 0.5
            V = rng.randn(mp, rank) * 0.5
            W = rng.randn(pn, rank) * 0.5
            
            prev_err = np.inf
            converged = False
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Solve for U: for each a, sum_r U[a,r] * (V[:,r] * W[:,r])[b,c] = T[a,b,c]
                # This is: U[a,:] @ M_U[a] = T[a,:,:] where M_U[a] depends on V,W
                # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # For fixed V,W: define Z[b,c] outer product structure
                # U[a,r] appears multiplied by (V[:,r] ⊗ W[:,r])[b,c]
                # So: U[a,:] @ (V * W.T) reshaped... 
                # Let's define: for each r, the contribution to T from term r is
                # U[:,r] * V[:,r] * W[:,r] (outer product)
                # So T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                # For U: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # = sum_r U[a,r] * (V[:,r] * W[:,r].T)  [as a matrix indexed by (b,c)]
                # So: T[a] = U[a,:] @ M where M[r, (b,c)] = V[b,r] * W[c,r]
                # M has shape (rank, mp*pn)
                
                # Build M_U: shape (rank, mp*pn)
                # M_U[r, b*pn+c] = V[b,r] * W[c,r]
                M_U = (V[:, :, None] * W[None, :, :]).T.reshape(rank, mp * pn)
                # M_U[r, idx] where idx = b*pn + c, value = V[b,r]*W[c,r]
                
                # T_flat[a] = T[a,:,:].reshape(-1), shape (mp*pn,)
                T_flat = T.reshape(nm, mp * pn)
                
                # Solve: U @ M_U.T = T_flat  =>  U[a,:] @ M_U.T = T_flat[a,:]
                # This is: for each a, solve (M_U @ M_U.T) U[a,:]^T = M_U @ T_flat[a,:]^T
                # Actually: U[a,:] @ M_U.T = T_flat[a,:]
                # => M_U @ U[a,:]^T = T_flat[a,:]^T
                # => (M_U.T @ M_U) U[a,:]^T = M_U.T @ T_flat[a,:]^T
                
                try:
                    gram_U = M_U @ M_U.T  # (rank, rank)
                    rhs_U = M_U @ T_flat.T  # (rank, nm)
                    # Solve gram_U @ U.T = rhs_U
                    U_new = np.linalg.solve(gram_U + 1e-10 * np.eye(rank), rhs_U).T
                except np.linalg.LinAlgError:
                    try:
                        U_new = np.linalg.lstsq(M_U, T_flat.T, rcond=None)[0].T
                    except:
                        break
                
                # Similarly for V:
                # T[a,b,:] = sum_r V[b,r] * (U[a,r] * W[c,r])
                # M_V[r, a*pn+c] = U[a,r] * W[c,r]
                M_V = (U[:, :, None] * W[None, :, :]).T.reshape(rank, nm * pn)
                T_flat_V = T.reshape(mp, nm * pn)
                # T_flat_V[b, a*pn+c] = T[a,b,c]
                # Need: T[a,b,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                # For fixed b: sum_r V[b,r] * (U[:,r] * W[:,r])[a,c] = T[:,b,:].reshape(-1)
                # M_V[r, a*pn+c] = U[a,r]*W[c,r]
                # V[b,:] @ M_V.T = T[:,b,:].reshape(-1)
                # M_V @ V[b,:]^T = T[:,b,:].reshape(-1)^T
                
                try:
                    gram_V = M_V @ M_V.T
                    rhs_V = M_V @ T_flat_V.T
                    V_new = np.linalg.solve(gram_V + 1e-10 * np.eye(rank), rhs_V).T
                except np.linalg.LinAlgError:
                    try:
                        V_new = np.linalg.lstsq(M_V, T_flat_V.T, rcond=None)[0].T
                    except:
                        break
                
                # Similarly for W:
                # T[a,b,c] = sum_r W[c,r] * (U[a,r] * V[b,r])
                # M_W[r, a*mp+b] = U[a,r] * V[b,r]
                M_W = (U[:, :, None] * V[None, :, :]).T.reshape(rank, nm * mp)
                T_flat_W = T.reshape(pn, nm * mp)
                # T_flat_W[c, a*mp+b] = T[a,b,c]
                # W[c,:] @ M_W.T = T_flat_W[c,:]
                
                try:
                    gram_W = M_W @ M_W.T
                    rhs_W = M_W @ T_flat_W.T
                    W_new = np.linalg.solve(gram_W + 1e-10 * np.eye(rank), rhs_W).T
                except np.linalg.LinAlgError:
                    try:
                        W_new = np.linalg.lstsq(M_W, T_flat_W.T, rcond=None)[0].T
                    except:
                        break
                
                U, V, W = U_new, V_new, W_new
                
                # Compute error
                # err = ||T - sum_r U[:,r]⊗V[:,r]⊗W[:,r]||_F
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(recon - T)
                
                if err < prev_err * (1 - 1e-6) and it > 5:
                    # Converged
                    if err < 1e-6:
                        converged = True
                        break
                    if err < 1e-4:
                        # Keep going a bit more
                        pass
                if err < 1e-8:
                    converged = True
                    break
                
                prev_err = err
            
            if converged or (err < 1e-6 if 'err' in dir() else False):
                # Verify exactly
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                max_err = np.max(np.abs(recon - T))
                if max_err < 1e-9:
                    # Round to half-integers if close
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    recon_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                    if np.max(np.abs(recon_r - T)) < 1e-9:
                        U, V, W = U_round, V_round, W_round
                    best_result = (U.copy(), V.copy(), W.copy())
                    best_rank = rank
                    break  # Found a valid decomposition at this rank
            elif err < 1e-4 and best_rank > rank:
                # Near convergence, keep as candidate
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                max_err = np.max(np.abs(recon - T))
                if max_err < 1e-6:
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    recon_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                    if np.max(np.abs(recon_r - T)) < 1e-9:
                        best_result = (U_round.copy(), V_round.copy(), W_round.copy())
                        best_rank = rank
                        break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    
    # First try ALS-based tensor decomposition for potentially lower rank
    # The block-based method gives rank 84; try to beat that
    try:
        als_result = _als_tensor_decomposition(n, m, p, max_rank=83, deadline=start_time + 80, seed=42)
        if als_result is not None:
            rank_als = len(als_result["U"][0])
            # Verify
            U_arr = np.array(als_result["U"])
            V_arr = np.array(als_result["V"])
            W_arr = np.array(als_result["W"])
            recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            T_check = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i*m+j, j*p+k, k*n+i] = 1.0
            if np.max(np.abs(recon - T_check)) < 1e-9:
                return als_result
    except Exception:
        pass

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def build_decomposition():
        if time.time() > deadline:
            return None

        strassen_terms = strassen_2x2x2_terms()
        
        # Partition indices
        i_groups = [[0, 1], [2]]
        j_groups = [[0, 1], [2, 3], [4]]
        k_groups = [[0, 1], [2, 3], [4, 5]]
        
        U_rows = n * m
        V_rows = m * p
        W_rows = p * n
        
        U = []
        V = []
        W = []
        
        # We build column by column (rank dimension)
        # But it's easier to build the lists of non-zeros and then assemble
        
        # Let's just append to the lists
        # Initialize with zeros
        # We don't know total rank in advance, so we'll collect terms
        
        terms_list = []
        
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si = len(ig)
                    sj = len(jg)
                    sk = len(kg)
                    
                    if si == 2 and sj == 2 and sk == 2:
                        # Strassen
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            u_term = {}
                            v_term = {}
                            w_term = {}
                            
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = ig[li], jg[lj]
                                u_term[(gi, gj)] = val_u
                                
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = jg[lj], kg[lk]
                                v_term[(gj, gk)] = val_v
                                
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = kg[lk], ig[li]
                                w_term[(gk, gi)] = val_w
                                
                            terms_list.append((u_term, v_term, w_term))
                    else:
                        # Schoolbook
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi = ig[li]
                                    gj = jg[lj]
                                    gk = kg[lk]
                                    
                                    u_term = {(gi, gj): 1}
                                    v_term = {(gj, gk): 1}
                                    w_term = {(gk, gi): 1}
                                    
                                    terms_list.append((u_term, v_term, w_term))
        
        rank = len(terms_list)
        
        # Assemble U, V, W
        U = [[0] * rank for _ in range(U_rows)]
        V = [[0] * rank for _ in range(V_rows)]
        W = [[0] * rank for _ in range(W_rows)]
        
        for col, (u_term, v_term, w_term) in enumerate(terms_list):
            for (i, j), val in u_term.items():
                U[i * m + j][col] = val
            for (j, k), val in v_term.items():
                V[j * p + k][col] = val
            for (k, i), val in w_term.items():
                W[k * n + i][col] = val
                
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    result = build_decomposition()
    
    if result is not None:
        rank = len(result["U"][0])
        if verify(result["U"], result["V"], result["W"], rank):
            return result
            
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
