import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices into blocks
    # i: 3 -> [2, 1]
    # j: 4 -> [2, 2]
    # k: 6 -> [2, 4]
    
    # Define blocks
    i_blocks = [2, 1]
    j_blocks = [2, 2]
    k_blocks = [2, 4]
    
    rank = 0
    U = []
    V = []
    W = []
    
    # Helper to add a block decomposition
    def add_block(i_start, j_start, k_start, ni, nj, nk):
        nonlocal rank
        # Determine rank of this block
        # If ni==2, nj==2, nk==2: Rank 7 (Strassen)
        # Else: Rank ni*nj (using sum-of-ones factorization)
        
        if ni == 2 and nj == 2 and nk == 2:
            r = 7
        else:
            r = ni * nj
            
        # Generate U, V, W columns for this block
        # Local indices: ii in [0, ni), jj in [0, nj), kk in [0, nk)
        
        u_cols = []
        v_cols = []
        w_cols = []
        
        if ni == 2 and nj == 2 and nk == 2:
            # Strassen decomposition for 2x2x2
            # Standard Strassen rank-7 decomposition
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # U (n x m part)
            # M1: a+d -> [1, 1; 1, 1]
            # M2: c+d -> [0, 1; 1, 1]
            # M3: a   -> [1, 0; 0, 0]
            # M4: d   -> [0, 0; 0, 1]
            # M5: a+b -> [1, 1; 0, 0]
            # M6: c-a -> [-1, 1; 1, 0]
            # M7: b-d -> [0, 1; -1, 1]
            
            U_local = [
                [1, 0, 1, 0, 1, -1, 0],
                [1, 1, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, -1],
                [0, 1, 0, 1, 0, 0, 1]
            ]
            
            # V (m x p part)
            # M1: e+h -> [1, 1; 1, 1]
            # M2: e   -> [1, 0; 1, 0]
            # M3: f-h -> [1, -1; 0, 0]
            # M4: g-e -> [-1, 0; 1, 0]
            # M5: h   -> [0, 1; 0, 1]
            # M6: e+f -> [1, 1; 1, 1]
            # M7: g+h -> [1, 1; 1, 1]
            
            V_local = [
                [1, 1, 1, -1, 0, 1, 1],
                [1, 0, -1, 0, 1, 1, 1],
                [1, 1, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1]
            ]
            
            # W (p x n part)
            # C11 = M1 - M4 + M5 - M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M6 - M7
            
            W_local = [
                [1, 0, 0, -1, 1, 0, -1],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 0],
                [1, -1, 0, 0, 0, -1, -1]
            ]
            
            # Note: W indices are k, i.
            # W[k*n + i, c]
            # My W_local is [k, i] or [i, k]?
            # Task says W[k*n+i, r].
            # So row index is k*n + i.
            # For 2x2x2, n=2.
            # Rows: 0,1,2,3 corresponding to (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1).
            # C11 (i=0,k=0) -> Row 0
            # C12 (i=0,k=1) -> Row 2? No.
            # k*n + i.
            # i=0, k=0 -> 0
            # i=1, k=0 -> 1
            # i=0, k=1 -> 2
            # i=1, k=1 -> 3
            
            # My W_local above was [i, k] or [k, i]?
            # C11 is i=0, k=0.
            # C12 is i=0, k=1.
            # C21 is i=1, k=0.
            # C22 is i=1, k=1.
            
            # Let's reorder W_local to match k*n+i.
            # Row 0 (k=0, i=0): C11 coeffs -> [1, 0, 0, -1, 1, 0, -1]
            # Row 1 (k=0, i=1): C21 coeffs -> [0, 1, 0, 1, 0, 1, 0]
            # Row 2 (k=1, i=0): C12 coeffs -> [0, 0, 1, 0, 1, 0, 0]
            # Row 3 (k=1, i=1): C22 coeffs -> [1, -1, 0, 0, 0, -1, -1]
            
            W_ordered = [
                [1, 0, 0, -1, 1, 0, -1],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [1, -1, 0, 0, 0, -1, -1]
            ]
            
            u_cols = U_local
            v_cols = V_local
            w_cols = W_ordered
            
        else:
            # Sum of ones factorization
            # Rank = ni * nj
            # u_{(ii,jj), i} = delta(i, ii)
            # v_{(ii,jj), j} = delta(j, jj)
            # w_{(ii,jj), k} = 1
            
            for ii in range(ni):
                for jj in range(nj):
                    u_col = [0] * ni
                    u_col[ii] = 1
                    u_cols.append(u_col)
                    
                    v_col = [0] * nj
                    v_col[jj] = 1
                    v_cols.append(v_col)
                    
                    w_col = [1] * nk
                    w_cols.append(w_col)
                    
        # Append to global lists
        # U rows: i*m + j.
        # V rows: j*p + k.
        # W rows: k*n + i.
        
        for c in range(r):
            u_global = [0] * (n * m)
            v_global = [0] * (m * p)
            w_global = [0] * (p * n)
            
            # Fill U
            for ii in range(ni):
                for jj in range(nj):
                    global_idx = (i_start + ii) * m + (j_start + jj)
                    u_global[global_idx] = u_cols[c][ii * nj + jj] # Wait, u_cols is flattened?
                    # My u_cols are lists of length ni*nj?
                    # In Strassen case, U_local is 4x7.
                    # In sum-of-ones case, u_cols are lists of length ni.
                    # I need to be consistent.
            
            # Let's restructure to avoid confusion.
            # Build full matrices for the block, then place.
            
            U_block = np.zeros((ni * nj, r))
            V_block = np.zeros((nj * nk, r))
            W_block = np.zeros((nk * ni, r))
            
            if ni == 2 and nj == 2 and nk == 2:
                U_block = np.array(U_local)
                V_block = np.array(V_local)
                W_block = np.array(W_ordered)
            else:
                idx = 0
                for ii in range(ni):
                    for jj in range(nj):
                        U_block[idx, idx] = 1 # Diagonal?
                        # No, rank is ni*nj.
                        # Each rank-1 term corresponds to one (ii,jj).
                        # Term t = ii*nj + jj.
                        # U_{(ii,jj), t} = 1 if t == ii*nj+jj.
                        # V_{(jj,kk), t} = 1 if jj == jj_t.
                        # W_{(kk,ii), t} = 1.
                        
                        # This logic is getting messy.
                        pass

    # Simplified approach:
    # Just generate the full U, V, W matrices directly.
    
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    def append_block(i_s, j_s, k_s, ni, nj, nk):
        nonlocal U, V, W
        
        if ni == 2 and nj == 2 and nk == 2:
            r = 7
            U_b = np.array([
                [1, 0, 1, 0, 1, -1, 0],
                [1, 1, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, -1],
                [0, 1, 0, 1, 0, 0, 1]
            ])
            V_b = np.array([
                [1, 1, 1, -1, 0, 1, 1],
                [1, 0, -1, 0, 1, 1, 1],
                [1, 1, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1]
            ])
            W_b = np.array([
                [1, 0, 0, -1, 1, 0, -1],
                [0, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [1, -1, 0, 0, 0, -1, -1]
            ])
        else:
            r = ni * nj
            U_b = np.zeros((ni * nj, r))
            V_b = np.zeros((nj * nk, r))
            W_b = np.zeros((nk * ni, r))
            
            for t in range(r):
                ii = t // nj
                jj = t % nj
                
                # U: row ii*nj+jj is 1
                U_b[ii * nj + jj, t] = 1
                
                # V: row jj*nk + kk is 1 for all kk?
                # No, V_b row index is jj*nk + kk.
                # We want V_{jj, kk} to be 1 for all kk?
                # In sum-of-ones, v_{jj} = 1.
                # So V_b[jj * nk + kk, t] = 1 for all kk.
                for kk in range(nk):
                    V_b[jj * nk + kk, t] = 1
                    
                # W: row kk*ni + ii is 1 for all kk?
                # w_{kk} = 1.
                # So W_b[kk * ni + ii, t] = 1 for all kk.
                for kk in range(nk):
                    W_b[kk * ni + ii, t] = 1

        # Place into global
        U_cols = np.zeros((n * m, r))
        V_cols = np.zeros((m * p, r))
        W_cols = np.zeros((p * n, r))
        
        # U placement
        for ii in range(ni):
            for jj in range(nj):
                global_idx = (i_s + ii) * m + (j_s + jj)
                local_idx = ii * nj + jj
                U_cols[global_idx, :] = U_b[local_idx, :]
                
        # V placement
        for jj in range(nj):
            for kk in range(nk):
                global_idx = (j_s + jj) * p + (k_s + kk)
                local_idx = jj * nk + kk
                V_cols[global_idx, :] = V_b[local_idx, :]
                
        # W placement
        for kk in range(nk):
            for ii in range(ni):
                global_idx = (k_s + kk) * n + (i_s + ii)
                local_idx = kk * ni + ii
                W_cols[global_idx, :] = W_b[local_idx, :]
                
        U = np.hstack([U, U_cols])
        V = np.hstack([V, V_cols])
        W = np.hstack([W, W_cols])

    # Partition:
    # i: 0-1 (2), 2 (1)
    # j: 0-1 (2), 2-3 (2)
    # k: 0-1 (2), 2-5 (4)
    
    blocks = [
        (0, 0, 0, 2, 2, 2),
        (0, 2, 0, 2, 2, 2),
        (0, 0, 2, 2, 2, 4),
        (0, 2, 2, 2, 2, 4),
        (2, 0, 0, 1, 2, 2),
        (2, 2, 0, 1, 2, 2),
        (2, 0, 2, 1, 2, 4),
        (2, 2, 2, 1, 2, 4)
    ]
    
    for b in blocks:
        append_block(*b)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
