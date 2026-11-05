"""Matrix multiplication 4,4,9 tensor decomposition - rank reduction via linear algebra."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build target tensor (sparse)
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1.0
    
    target_flat = target.flatten()
    
    def verify(U, V, W, rank):
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T.flatten(), target_flat, atol=1e-9)
    
    # Build rank-114 solution: 2x Strassen 4x4x4 (49 each) + schoolbook 4x4x1 (16)
    def build_rank_114():
        U_cols, V_cols, W_cols = [], [], []
        
        def add_term(u, v, w):
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        # Strassen 2x2x2: 7 terms
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        s2x2 = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
        ]
        
        def add_strassen_4x4(B_col_off):
            def add_inner(A_blocks, B_blocks, C_blocks):
                for uc, vc, wc in s2x2:
                    u_vec = [0.0] * 16
                    v_vec = [0.0] * 36
                    w_vec = [0.0] * 36
                    
                    for loc, coeff in uc.items():
                        li, lj = loc // 2, loc % 2
                        for br, bc, bc_c in A_blocks:
                            gi, gj = br*2 + li, bc*2 + lj
                            u_vec[gi*m + gj] += coeff * bc_c
                    
                    for loc, coeff in vc.items():
                        lj, lk = loc // 2, loc % 2
                        for br, bc, bc_c in B_blocks:
                            gj, gk = br*2 + lj, B_col_off + bc*2 + lk
                            v_vec[gj*p + gk] += coeff * bc_c
                    
                    for loc, coeff in wc.items():
                        li, lk = loc // 2, loc % 2
                        for br, bc, bc_c in C_blocks:
                            gi, gk = br*2 + li, B_col_off + bc*2 + lk
                            w_vec[gk*n + gi] += coeff * bc_c
                    
                    add_term(u_vec, v_vec, w_vec)
            
            # 7 outer Strassen combinations
            add_inner([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
            add_inner([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])
            add_inner([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
            add_inner([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
            add_inner([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
            add_inner([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
            add_inner([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
        
        add_strassen_4x4(0)
        add_strassen_4x4(4)
        
        # Schoolbook for last column (k=8)
        for i in range(4):
            for j in range(4):
                u_vec = [0.0] * 16
                v_vec = [0.0] * 36
                w_vec = [0.0] * 36
                u_vec[i*m + j] = 1.0
                v_vec[j*p + 8] = 1.0
                w_vec[8*n + i] = 1.0
                add_term(u_vec, v_vec, w_vec)
        
        rank = len(U_cols)
        U = np.array(U_cols).T  # (16, rank)
        V = np.array(V_cols).T  # (36, rank)
        W = np.array(W_cols).T  # (36, rank)
        return U, V, W, rank
    
    U, V, W, rank = build_rank_114()
    
    # Verify the base solution
    if not verify(U, V, W, rank):
        # Fallback: pure schoolbook rank 144
        U_fb = np.zeros((16, 144))
        V_fb = np.zeros((36, 144))
        W_fb = np.zeros((36, 144))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_fb[i*4+j, idx] = 1
                    V_fb[j*9+k, idx] = 1
                    W_fb[k*4+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 1: Try to eliminate linearly dependent columns
    # The tensor equation is: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # This is linear in W given U,V. But also: for each (a,b) pair, 
    # sum_r U[a,r]*V[b,r]*W[:,r] = T[a,b,:]
    # So we can write: (U ⊙ V) @ W = T_reshaped where U⊙V is (16*36, rank)
    
    # Check if we can reduce by finding null space of the UV product matrix
    UV = np.kron(U, V)  # This isn't quite right; need element-wise per column
    
    # Actually: D[a*36+b, r] = U[a,r] * V[b,r]
    D = np.zeros((16*36, rank))
    for r in range(rank):
        D[:, r] = np.kron(U[:, r], V[:, r])
    
    T_mat = target.reshape(16*36, 36)
    
    # Check rank of D
    try:
        D_rank = np.linalg.matrix_rank(D, tol=1e-10)
    except:
        D_rank = rank
    
    if D_rank < rank:
        # There's redundancy. Find a basis for the column space.
        Q, R = np.linalg.qr(D, mode='reduced')
        # Columns of D that are linearly independent
        pivots = np.argsort(-np.abs(np.diag(R)))
        independent = sorted(pivots[:D_rank])
        
        if D_rank < best_rank:
            U_new = U[:, independent]
            V_new = V[:, independent]
            # Solve for W_new
            D_new = np.zeros((16*36, D_rank))
            for r in range(D_rank):
                D_new[:, r] = np.kron(U_new[:, r], V_new[:, r])
            W_new, _, _, _ = np.linalg.lstsq(D_new, T_mat.T, rcond=None)
            W_new = W_new.T
            
            if verify(U_new, V_new, W_new, D_rank):
                best_rank = D_rank
                best_U, best_V, best_W = U_new, V_new, W_new
    
    # Strategy 2: Try removing one column at a time and re-solving
    # For each column r, check if it can be expressed as a combination of others
    for attempt in range(rank):
        if time.time() > deadline - 20:
            break
        
        # Try removing column 'attempt'
        keep = [i for i in range(best_rank) if i != attempt]
        if len(keep) >= best_rank - 1:
            U_sub = best_U[:, keep]
            V_sub = best_V[:, keep]
            
            D_sub = np.zeros((16*36, len(keep)))
            for r in range(len(keep)):
                D_sub[:, r] = np.kron(U_sub[:, r], V_sub[:, r])
            
            try:
                W_sub, residuals, _, _ = np.linalg.lstsq(D_sub, T_mat.T, rcond=None)
                W_sub = W_sub.T
                
                if verify(U_sub, V_sub, W_sub, len(keep)):
                    best_rank = len(keep)
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
            except:
                pass
    
    # Strategy 3: Random support search for lower rank
    # Use the structure from the 114 solution as a guide
    rng = np.random.RandomState(42)
    
    # Get active indices from current best
    u_active = set(np.where(np.any(best_U != 0, axis=1))[0])
    v_active = set(np.where(np.any(best_V != 0, axis=1))[0])
    w_active = set(np.where(np.any(best_W != 0, axis=1))[0])
    
    u_pool = list(u_active)
    v_pool = list(v_active)
    w_pool = list(w_active)
    
    # Try ranks from best_rank-1 down to max(49, best_rank-10)
    for target_rank in range(best_rank - 1, max(48, best_rank - 15), -1):
        if time.time() > deadline - 15:
            break
        
        max_iters = 30
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            U_try = np.zeros((16, target_rank))
            V_try = np.zeros((36, target_rank))
            
            for r in range(target_rank):
                # Random sparse supports
                n_u = rng.randint(2, 5)
                n_v = rng.randint(2, 5)
                
                u_supp = rng.choice(u_pool, size=n_u, replace=False)
                v_supp = rng.choice(v_pool, size=n_v, replace=False)
                
                for pos in u_supp:
                    U_try[pos, r] = rng.choice([-1.0, 1.0])
                for pos in v_supp:
                    V_try[pos, r] = rng.choice([-1.0, 1.0])
            
            D_try = np.zeros((16*36, target_rank))
            for r in range(target_rank):
                D_try[:, r] = np.kron(U_try[:, r], V_try[:, r])
            
            try:
                W_try, _, _, _ = np.linalg.lstsq(D_try, T_mat.T, rcond=None)
                W_try = W_try.T
                
                # Check exactness
                T_check = np.einsum('ar,br,cr->abc', U_try, V_try, W_try)
                if np.allclose(T_check.flatten(), target_flat, atol=1e-9):
                    # Check if values are close to half-integers
                    all_vals = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
                    scaled = all_vals * 2
                    if np.allclose(scaled, np.round(scaled), atol=1e-6):
                        U_exact = np.round(U_try * 2) / 2.0
                        V_exact = np.round(V_try * 2) / 2.0
                        W_exact = np.round(W_try * 2) / 2.0
                        
                        if verify(U_exact, V_exact, W_exact, target_rank):
                            best_rank = target_rank
                            best_U, best_V, best_W = U_exact, V_exact, W_exact
                            break  # Found lower rank, try even lower
            except:
                pass
    
    # Strategy 4: Try to compress the schoolbook part with Strassen-like sharing
    # The last column (k=8) uses 16 rank-1 terms. Can we share with the Strassen parts?
    # This is harder, but let's try a targeted approach:
    # Replace some Strassen columns with combined columns that handle both
    
    if time.time() < deadline - 25:
        # Try: use Laderman-style rank-16 for 4x4x4 if we can find it
        # For now, try a hybrid: reduce the schoolbook part by using 
        # pre-computed sums from the Strassen parts
        
        # Alternative: try rank = 2*16 + 16 = 48 using Laderman for each 4x4x4 block
        # Laderman's algorithm for 4x4x4 uses 16 multiplications
        # Let me implement it properly
        
        def laderman_4x4x4(B_col_off):
            """Add 16 Laderman terms for 4x4 * 4x4 -> 4x4."""
            terms = []
            
            # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1) - each 2x2
            # B blocks: B00, B01, B10, B11
            # C blocks: C00, C01, C10, C11
            
            # Laderman's 16 products:
            # P1  = A00 * B00
            # P2  = A00 * B11  
            # P3  = A11 * B00
            # P4  = A11 * B11
            # P5  = A00 * (B01 + B11)
            # P6  = A11 * (B00 + B10)
            # P7  = (A00 + A10) * (B00 - B11)
            # P8  = (A01 + A11) * (B01 - B00)
            # P9  = (A00 + A01) * B10
            # P10 = A10 * (B01 + B11)
            # P11 = (A01 + A11) * B00
            # P12 = (A10 + A11) * B01
            # P13 = (A00 + A01) * (B10 - B00)
            # P14 = (A10 + A11) * (B11 - B01)
            # P15 = (A01 - A10) * (B10 + B01)
            # P16 = (A00 + A11) * (B00 + B11)
            
            # Reconstruction:
            # C00 = P1 + P5 - P7 + P9 - P13 + P15 - P16
            # C01 = P2 - P8 + P10 + P12 - P14 - P15 + P16  
            # C10 = P3 - P6 + P7 + P11 - P13 - P14 + P16
            # C11 = P4 + P6 - P8 - P9 + P13 + P14 - P16
            
            # This is complex. Let me use a verified formulation.
            # Actually, let me skip this and use the Strassen approach which we know works.
            pass
        
        # Instead, let's try a different reduction:
        # Can we share columns between the two 4x4x4 Strassen blocks?
        # The A-side is the same for both blocks!
        # So U vectors for block 1 and block 2 could potentially be shared.
        
        # In our construction, the U vectors for the first 49 terms use A indices 0-3 (cols 0-3 of B)
        # and the U vectors for the next 49 terms use the same A indices but B cols 4-7.
        # Wait, no - the U vectors are the same! Only V and W differ.
        
        # Let me check: in add_strassen_4x4, the U vector depends only on A_blocks,
        # which is the same for both calls. So U[:, 0:49] == U[:, 49:98]!
        
        if np.allclose(best_U[:, :49], best_U[:, 49:98]):
            # We can potentially merge these!
            # But W vectors are different, so we can't simply merge.
            # However, we can factor: if U1 == U2, then:
            # sum_{r=0}^{48} U_r ⊗ V1_r ⊗ W1_r + sum_{r=0}^{48} U_r ⊗ V2_r ⊗ W2_r
            # = sum_{r=0}^{48} U_r ⊗ (V1_r ⊗ W1_r + V2_r ⊗ W2_r)
            # But this doesn't reduce rank directly.
            
            # Actually, the rank is determined by the number of terms, not by U being shared.
            # Each term still contributes one rank. So no reduction here.
            pass
    
    # Final: ensure we have a valid solution
    if best_rank < 144:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out, V_out, W_out = U, V, W
    
    # Round to half-integers for cleanliness
    U_out = np.round(U_out * 2) / 2.0
    V_out = np.round(V_out * 2) / 2.0
    W_out = np.round(W_out * 2) / 2.0
    
    final_rank = U_out.shape[1]
    
    # Final verification
    if not verify(U_out, V_out, W_out, final_rank):
        # Fallback to the original 114
        U_out, V_out, W_out = U, V, W
        final_rank = 114
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
