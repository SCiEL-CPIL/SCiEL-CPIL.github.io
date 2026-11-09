import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Strassen 2x2x2 terms
    # u, v, w are lists of 4 elements
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Helper to add terms to U, V, W
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # We need to map local indices to global indices
    # Global A: 2x4 (indices 0-7)
    # Global B: 4x6 (indices 0-23)
    # Global C: 2x6 (indices 0-11)
    
    # Part 1: C4 = A * B4 (2x4 * 4x4 -> 2x4)
    # A is same. B4 is B[:, 0:4]. C4 is C[:, 0:4].
    # Split A into AL (2x2) and AR (2x2).
    # Split B4 into BT (2x4) and BB (2x4).
    # C4 = AL * BT + AR * BB.
    # AL * BT is 2x2 * 2x4.
    # AR * BB is 2x2 * 2x4.
    
    # Let's do 2x2 * 2x4 using two 2x2 * 2x2 blocks.
    # BT = [BT1, BT2] (2x2 each).
    # C4_L = AL * BT1 (2x2).
    # C4_R = AL * BT2 (2x2).
    # Similarly for AR.
    
    # This is getting complex to hardcode.
    # Let's write a recursive function or a composition helper.
    
    # Actually, since n=2, m=4, p=6 is small, we can just define the terms explicitly
    # or write a small composition engine.
    
    # Composition Engine:
    # A term is (u, v, w).
    # u is a vector of length n*m.
    # v is a vector of length m*p.
    # w is a vector of length p*n.
    
    # For subproblem 2x2x2:
    # u_sub is length 4.
    # v_sub is length 4.
    # w_sub is length 4.
    
    # To embed 2x2x2 into 2x4x4:
    # A is 2x4. AL is A[:, 0:2].
    # u_global[i] = u_sub[i] for i in 0..3, 0 otherwise.
    # B is 4x4. BT is B[0:2, :].
    # v_global[j*p+k] = v_sub[j*2+k] for j in 0..1, k in 0..3?
    # Wait, B is 4x4. BT is 2x4.
    # v_sub is for 2x2 B.
    # So we need to split BT into two 2x2 blocks.
    
    # Let's just use the 2x4x2 and 2x4x4 decompositions directly.
    
    # 2x4x2 decomposition:
    # A (2x4) * B (4x2) -> C (2x2).
    # C = AL * BT + AR * BB.
    # AL (2x2) * BT (2x2) -> C1 (2x2).
    # AR (2x2) * BB (2x2) -> C2 (2x2).
    # C = C1 + C2.
    
    # Terms for AL * BT:
    # u: AL entries (indices 0-3 of A).
    # v: BT entries (indices 0-7 of B? No, B is 4x2. BT is B[0:2, :]. Indices 0-3).
    # w: C1 entries (indices 0-3 of C).
    
    # Terms for AR * BB:
    # u: AR entries (indices 4-7 of A).
    # v: BB entries (indices 4-7 of B).
    # w: C2 entries (indices 0-3 of C, added to C1).
    
    # So for 2x4x2, we have 7 + 7 = 14 terms.
    
    # 2x4x4 decomposition:
    # A (2x4) * B (4x4) -> C (2x4).
    # C = AL * BT + AR * BB.
    # BT is 2x4. BB is 2x4.
    # AL * BT is 2x2 * 2x4.
    # This can be split into AL * BT1 + AL * BT2.
    # AL * BT1 is 2x2 * 2x2.
    # AL * BT2 is 2x2 * 2x2.
    # So AL * BT has 14 terms.
    # AR * BB has 14 terms.
    # Total 28 terms.
    
    # 2x4x6 decomposition:
    # A (2x4) * B (4x6) -> C (2x6).
    # C = A * B4 + A * B2.
    # A * B4 is 2x4x4 (28 terms).
    # A * B2 is 2x4x2 (14 terms).
    # Total 42 terms.
    
    # Let's implement this mapping.
    
    def map_u(u_sub, offset_a):
        # u_sub is length 4 (2x2)
        # u_global is length 8 (2x4)
        u_global = [0]*8
        for i in range(4):
            u_global[offset_a + i] = u_sub[i]
        return u_global
        
    def map_v_bt(v_sub, offset_b, offset_k):
        # v_sub is length 4 (2x2)
        # v_global is length 24 (4x6)
        # BT is B[0:2, :].
        # v_sub corresponds to B[j, k] for j in 0..1, k in 0..1 (local).
        # Global B index is j*6 + k_global.
        # k_global = offset_k + k_local.
        v_global = [0]*24
        for j in range(2):
            for k in range(2):
                idx_sub = j*2 + k
                idx_global = j*6 + (offset_k + k)
                v_global[idx_global] = v_sub[idx_sub]
        return v_global
        
    def map_v_bb(v_sub, offset_b, offset_k):
        # v_sub is length 4 (2x2)
        # BB is B[2:4, :].
        # j_global = 2 + j_local.
        v_global = [0]*24
        for j in range(2):
            for k in range(2):
                idx_sub = j*2 + k
                idx_global = (2+j)*6 + (offset_k + k)
                v_global[idx_global] = v_sub[idx_sub]
        return v_global
        
    def map_w(w_sub, offset_c):
        # w_sub is length 4 (2x2)
        # w_global is length 12 (2x6)
        # C is 2x6.
        # w_sub corresponds to C[i, k] for i in 0..1, k in 0..1 (local).
        # Global C index is k_global*2 + i.
        # k_global = offset_c + k_local.
        w_global = [0]*12
        for i in range(2):
            for k in range(2):
                idx_sub = k*2 + i
                idx_global = (offset_c + k)*2 + i
                w_global[idx_global] = w_sub[idx_sub]
        return w_global

    # Add 2x4x4 part (C[:, 0:4])
    # AL * BT1 (B[0:2, 0:2])
    for u, v, w in strassen_terms:
        U.append(map_u(u, 0))
        V.append(map_v_bt(v, 0, 0))
        W.append(map_w(w, 0))
        
    # AL * BT2 (B[0:2, 2:4])
    for u, v, w in strassen_terms:
        U.append(map_u(u, 0))
        V.append(map_v_bt(v, 0, 2))
        W.append(map_w(w, 2))
        
    # AR * BB1 (B[2:4, 0:2])
    for u, v, w in strassen_terms:
        U.append(map_u(u, 4))
        V.append(map_v_bb(v, 0, 0))
        W.append(map_w(w, 0))
        
    # AR * BB2 (B[2:4, 2:4])
    for u, v, w in strassen_terms:
        U.append(map_u(u, 4))
        V.append(map_v_bb(v, 0, 2))
        W.append(map_w(w, 2))
        
    # Add 2x4x2 part (C[:, 4:6])
    # AL * BT (B[0:2, 4:6])
    for u, v, w in strassen_terms:
        U.append(map_u(u, 0))
        V.append(map_v_bt(v, 0, 4))
        W.append(map_w(w, 4))
        
    # AR * BB (B[2:4, 4:6])
    for u, v, w in strassen_terms:
        U.append(map_u(u, 4))
        V.append(map_v_bb(v, 0, 4))
        W.append(map_w(w, 4))
        
    return {"U": U, "V": V, "W": W}
