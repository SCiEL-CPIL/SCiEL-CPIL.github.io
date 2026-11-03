import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Base decompositions
    
    # Strassen 2x2x2 (Rank 7)
    # U, V, W are lists of tuples (index, value)
    strassen_terms = [
        # m1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # m2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1)]),
        # m3
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1)]),
        # m4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # m5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # m6
        ([(0, -1), (2, 1)], [(0, 1), (1, 1)], [(3, 1)]),
        # m7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # 2x2x1 (Rank 4)
    # A(2x2) B(2x1) -> C(2x1)
    # W index k*n+i. Here n=2 (local), p=1 (local).
    # Wait, local n=2, m=2, p=1.
    # W index = k * n_local + i.
    # k is 0 (only one column in B).
    # So W index = 0 * 2 + i = i.
    # C_0 -> i=0 -> w=0.
    # C_1 -> i=1 -> w=1.
    mul_2x2x1_terms = [
        ([(0, 1)], [(0, 1)], [(0, 1)]), # A00 B0 -> C0
        ([(1, 1)], [(1, 1)], [(0, 1)]), # A01 B1 -> C0
        ([(2, 1)], [(0, 1)], [(1, 1)]), # A10 B0 -> C1
        ([(3, 1)], [(1, 1)], [(1, 1)]), # A11 B1 -> C1
    ]

    # Helper to add a local term to global matrices
    # u_offset, v_offset, w_offset
    # u_size, v_size, w_size for local block
    # global U, V, W are lists of lists [row][col]
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_local, v_local, w_local, u_off, v_off, w_off):
        nonlocal rank
        # Create zero vectors for this term
        # We need to know the global sizes
        # We will just append to U, V, W columns
        # But we need to fill the rest with 0
        # This is inefficient if we do it per term.
        # Better to collect terms first, then build matrices.
        pass

    # Collect all terms as (u_sparse, v_sparse, w_sparse)
    # u_sparse: list of (global_index, val)
    # v_sparse: list of (global_index, val)
    # w_sparse: list of (global_index, val)
    
    terms = []
    
    def embed_term(local_u, local_v, local_w, u_off, v_off, w_off):
        gu = [(u + u_off, val) for u, val in local_u]
        gv = [(v + v_off, val) for v, val in local_v]
        gw = [(w + w_off, val) for w, val in local_w]
        terms.append((gu, gv, gw))
        
    # Global indices:
    # U: i*m + j. i in 0..1, j in 0..3.
    # V: j*p + k. j in 0..3, k in 0..6.
    # W: k*n + i. k in 0..6, i in 0..1.
    
    # Block 1: A1 (i=0..1, j=0..1), B1 (j=0..1, k=0..6)
    # A1 offset in U: j starts at 0. i starts at 0.
    # u_off = 0.
    # B1 offset in V: j starts at 0. k starts at 0.
    # v_off = 0.
    # Output C1: k=0..6, i=0..1.
    # w_off = 0.
    
    # Sub-block 1a: B1a (j=0..1, k=0..2). 2x2x3.
    # Uses Strassen (2x2x2) for k=0,1 and 2x2x1 for k=2.
    # Strassen part:
    for t in strassen_terms:
        embed_term(t[0], t[1], t[2], u_off=0, v_off=0, w_off=0)
    # 2x2x1 part:
    # Local B index k=0 corresponds to global k=2.
    # Local V index j*p_local + k_local. p_local=1.
    # So local v index = j.
    # Global v index = j*7 + k_global.
    # For k_global=2, v_global = j*7 + 2.
    # Local W index = k_local * n_local + i. n_local=2.
    # k_local=0. w_local = i.
    # Global W index = k_global * n + i = 2*2 + i = 4 + i.
    for t in mul_2x2x1_terms:
        # Map local V indices (j) to global (j*7 + 2)
        # local v index is j (since p=1, v=j*1+0=j).
        # Wait, in mul_2x2x1_terms, V indices are (0,1) and (1,1).
        # These correspond to j=0 and j=1.
        # So we need to shift V indices by 2*7=14? No.
        # v_global = j*7 + 2.
        # local v = j.
        # So v_global = v_local * 7 + 2.
        # This is not a simple offset.
        
        # Let's generalize embed to handle scaling?
        # Or just map manually.
        
        gu = [(u, val) for u, val in t[0]] # u_off 0
        gv = [(v * 7 + 2, val) for v, val in t[1]]
        gw = [(w + 4, val) for w, val in t[2]]
        terms.append((gu, gv, gw))
        
    # Sub-block 1b: B1b (j=0..1, k=3..6). 2x2x4.
    # Uses two Strassen.
    # First Strassen for k=3,4.
    # Second Strassen for k=5,6.
    
    # Strassen 1: k=3,4.
    # v_global = j*7 + k.
    # local v = j*2 + k_local.
    # k_local=0 -> k=3. k_local=1 -> k=4.
    # v_global = j*7 + (3 + k_local) = j*7 + 3 + k_local.
    # This is not a simple offset either.
    
    # It is easier to just generate the terms for 2x2x2 and map indices.
    
    def map_strassen(u_off, k_start, w_off):
        # u_off: offset for U (j_start * ?)
        # k_start: start of k indices
        # w_off: offset for W
        for t in strassen_terms:
            gu = [(u, val) for u, val in t[0]]
            # v_local = j*2 + k_local
            # v_global = j*7 + (k_start + k_local)
            gv = []
            for v_idx, val in t[1]:
                j = v_idx // 2
                k_local = v_idx % 2
                v_global = j * 7 + (k_start + k_local)
                gv.append((v_global, val))
            # w_local = k_local * 2 + i
            # w_global = (k_start + k_local) * 2 + i
            gw = []
            for w_idx, val in t[2]:
                k_local = w_idx // 2
                i = w_idx % 2
                w_global = (k_start + k_local) * 2 + i
                gw.append((w_global, val))
            terms.append((gu, gv, gw))

    # Block 1, Sub-block 1b, Part 1: k=3,4
    map_strassen(u_off=0, k_start=3, w_off=0)
    # Block 1, Sub-block 1b, Part 2: k=5,6
    map_strassen(u_off=0, k_start=5, w_off=0)
    
    # Block 2: A2 (i=0..1, j=2..3), B2 (j=2..3, k=0..6)
    # A2 offset in U: j starts at 2.
    # u_global = i*4 + j.
    # local u = i*2 + j_local.
    # j = 2 + j_local.
    # u_global = i*4 + 2 + j_local.
    
    def map_strassen_block2(u_j_start, k_start):
        for t in strassen_terms:
            gu = []
            for u_idx, val in t[0]:
                i = u_idx // 2
                j_local = u_idx % 2
                j_global = u_j_start + j_local
                u_global = i * 4 + j_global
                gu.append((u_global, val))
            gv = []
            for v_idx, val in t[1]:
                j_local = v_idx // 2
                k_local = v_idx % 2
                j_global = u_j_start + j_local
                k_global = k_start + k_local
                v_global = j_global * 7 + k_global
                gv.append((v_global, val))
            gw = []
            for w_idx, val in t[2]:
                k_local = w_idx // 2
                i = w_idx % 2
                k_global = k_start + k_local
                w_global = k_global * 2 + i
                gw.append((w_global, val))
            terms.append((gu, gv, gw))

    def map_2x2x1_block2(u_j_start, k_start):
        for t in mul_2x2x1_terms:
            gu = []
            for u_idx, val in t[0]:
                i = u_idx // 2
                j_local = u_idx % 2
                j_global = u_j_start + j_local
                u_global = i * 4 + j_global
                gu.append((u_global, val))
            gv = []
            for v_idx, val in t[1]:
                j_local = v_idx
                j_global = u_j_start + j_local
                k_global = k_start
                v_global = j_global * 7 + k_global
                gv.append((v_global, val))
            gw = []
            for w_idx, val in t[2]:
                i = w_idx
                k_global = k_start
                w_global = k_global * 2 + i
                gw.append((w_global, val))
            terms.append((gu, gv, gw))

    # Block 2, Sub-block 2a: B2a (j=2..3, k=0..2).
    # Strassen for k=0,1
    map_strassen_block2(u_j_start=2, k_start=0)
    # 2x2x1 for k=2
    map_2x2x1_block2(u_j_start=2, k_start=2)
    
    # Block 2, Sub-block 2b: B2b (j=2..3, k=3..6).
    # Strassen for k=3,4
    map_strassen_block2(u_j_start=2, k_start=3)
    # Strassen for k=5,6
    map_strassen_block2(u_j_start=2, k_start=5)
    
    # Build matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (gu, gv, gw) in enumerate(terms):
        for idx, val in gu:
            U[idx][r] = val
        for idx, val in gv:
            V[idx][r] = val
        for idx, val in gw:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
