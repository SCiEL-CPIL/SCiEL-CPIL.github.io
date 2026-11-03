"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p  # = 105
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    # Define Strassen 2x2x2 block vectors (generic indices 0,1)
    # These are (u, v, w) vectors of size 2x2, 2x2, 2x2 flattened?
    # No, we need them in the context of the full dimensions to place them.
    # But for verification, we can use small vectors.
    
    # Let's define the 7 Strassen terms for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11) -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00) -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7
    
    # We represent each term as (u_vec, v_vec, w_vec) where u_vec is length 4 (2x2), etc.
    # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
    
    s_terms = []
    
    # M1
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[0] = 1.0; u[3] = 1.0 # A00, A11
    v[0] = 1.0; v[3] = 1.0 # B00, B11
    w[0] = 1.0; w[3] = 1.0 # C00, C11
    s_terms.append((u, v, w))
    
    # M2
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[2] = 1.0; u[3] = 1.0 # A10, A11
    v[0] = 1.0 # B00
    w[2] = 1.0; w[3] = -1.0 # C10, C11
    s_terms.append((u, v, w))
    
    # M3
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[0] = 1.0 # A00
    v[1] = 1.0; v[3] = -1.0 # B01, B11
    w[1] = 1.0; w[3] = 1.0 # C01, C11
    s_terms.append((u, v, w))
    
    # M4
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[3] = 1.0 # A11
    v[2] = 1.0; v[0] = -1.0 # B10, B00
    w[0] = 1.0; w[2] = 1.0 # C00, C10
    s_terms.append((u, v, w))
    
    # M5
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[0] = 1.0; u[1] = 1.0 # A00, A01
    v[3] = 1.0 # B11
    w[0] = -1.0; w[1] = 1.0 # C00, C01
    s_terms.append((u, v, w))
    
    # M6
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[2] = 1.0; u[0] = -1.0 # A10, A00
    v[0] = 1.0; v[1] = 1.0 # B00, B01
    w[3] = 1.0 # C11
    s_terms.append((u, v, w))
    
    # M7
    u = np.zeros(4); v = np.zeros(4); w = np.zeros(4)
    u[1] = 1.0; u[3] = -1.0 # A01, A11
    v[2] = 1.0; v[3] = 1.0 # B10, B11
    w[0] = 1.0 # C00
    s_terms.append((u, v, w))
    
    # Verify Strassen block once
    def verify_strassen_block():
        R = 7
        Ur = np.array([t[0] for t in s_terms]).T # 4x7
        Vr = np.array([t[1] for t in s_terms]).T # 4x7
        Wr = np.array([t[2] for t in s_terms]).T # 4x7
        
        # Reshape to 2x2x7
        Ur = Ur.reshape(2, 2, R)
        Vr = Vr.reshape(2, 2, R)
        Wr = Wr.reshape(2, 2, R)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        T = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for c in range(R):
                        s += Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
                    T[i,j,k] = s
        return np.allclose(T, 1.0, atol=1e-10)

    if not verify_strassen_block():
        # Fallback to schoolbook if Strassen is broken (should not happen)
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Search for disjoint Strassen blocks
    # A block is defined by (i0, i1, j0, j1, k0, k1)
    # We want to maximize the number of blocks.
    # Each block covers 2x2x2 = 8 (i,j,k) tuples.
    # Total tuples = 105. Max blocks = 13 (104 tuples).
    # Rank = 105 - 13 = 92.
    
    all_i_pairs = [(0,1), (0,2), (1,2)]
    all_j_pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    all_k_pairs = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (1,2), (1,3), (1,4), (1,5), (1,6), (2,3), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6), (4,5), (4,6), (5,6)]
    
    # Greedy selection of blocks
    # We need to ensure that no (i,j,k) is covered by more than one block.
    
    def greedy_block_selection(time_limit=30):
        start_time = time.time()
        best_blocks = []
        covered = set()
        
        # Sort potential blocks by some heuristic? Just iterate.
        # To find the *maximum* number of disjoint blocks, we can use a simple backtracking or greedy.
        # Greedy: pick the block that covers the most uncovered points? All cover 8.
        # So we just pick the first valid one?
        # To be smarter, we can try to pick blocks that leave a "better" structure for future blocks.
        # But for now, let's just try to find a maximal set.
        
        # We can iterate through all possible blocks and add them if valid.
        # To improve the count, we might need to swap.
        # Let's try a simple greedy first, then maybe a local search.
        
        # Generate all possible blocks
        all_blocks = []
        for ip in all_i_pairs:
            for jp in all_j_pairs:
                for kp in all_k_pairs:
                    all_blocks.append((ip, jp, kp))
        
        # Shuffle to get different results? Or just iterate.
        # Let's iterate and add if valid.
        for block in all_blocks:
            if time.time() - start_time > time_limit:
                break
            ip, jp, kp = block
            I = {ip[0], ip[1]}
            J = {jp[0], jp[1]}
            K = {kp[0], kp[1]}
            
            # Check overlap
            valid = True
            for i in I:
                for j in J:
                    for k in K:
                        if (i, j, k) in covered:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: continue
            
            # Add block
            best_blocks.append(block)
            for i in I:
                for j in J:
                    for k in K:
                        covered.add((i, j, k))
                        
        return best_blocks

    # Run greedy
    best_blocks = greedy_block_selection(time_limit=20)
    
    # Try to improve by removing one block and seeing if we can fit more?
    # Simple local search: try to swap one block for another that allows adding a new one.
    # This is complex. Let's stick with the greedy result for now and see the rank.
    
    # Build the decomposition
    def build_decomposition(blocks):
        U_list, V_list, W_list = [], [], []
        covered = set()
        
        for ip, jp, kp in blocks:
            i0, i1 = sorted(ip)
            j0, j1 = sorted(jp)
            k0, k1 = sorted(kp)
            
            # Map local indices (0,1) to global indices
            i_map = {0: i0, 1: i1}
            j_map = {0: j0, 1: j1}
            k_map = {0: k0, 1: k1}
            
            for u_loc, v_loc, w_loc in s_terms:
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                
                # Place u
                for li in range(2):
                    for lj in range(2):
                        idx = li * 2 + lj
                        if u_loc[idx] != 0:
                            gi = i_map[li]
                            gj = j_map[lj]
                            u[gi * m + gj] = u_loc[idx]
                
                # Place v
                for lj in range(2):
                    for lk in range(2):
                        idx = lj * 2 + lk
                        if v_loc[idx] != 0:
                            gj = j_map[lj]
                            gk = k_map[lk]
                            v[gj * p + gk] = v_loc[idx]
                
                # Place w
                for lk in range(2):
                    for li in range(2):
                        idx = lk * 2 + li
                        if w_loc[idx] != 0:
                            gk = k_map[lk]
                            gi = i_map[li]
                            w[gk * n + gi] = w_loc[idx]
                
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
                
            # Mark covered
            for i in [i0, i1]:
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        covered.add((i, j, k))
        
        # Add schoolbook for uncovered
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        u = np.zeros(nm); v = np.zeros(mp); w = np.zeros(pn)
                        u[i * m + j] = 1.0
                        v[j * p + k] = 1.0
                        w[k * n + i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
        
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        return U, V, W

    U, V, W = build_decomposition(best_blocks)
    
    # Final verification (just in case)
    def verify(U, V, W):
        R = U.shape[1]
        Ur = U.reshape(n, m, R)
        Vr = V.reshape(m, p, R)
        Wr = W.reshape(p, n, R)
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=1e-10)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
