import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start = time.time()
    deadline = start + 115.0

    n, m, p = DIMENSIONS
    
    # Target Rank 69
    # 1x4x6 (24)
    # 2x4x6 ->
    #   2x3x6 ->
    #     2x1x6 (12)
    #     2x2x6 (21)
    #   2x1x6 (12)
    
    total_rank = 24 + 12 + 12 + 21
    U = np.zeros((n * m, total_rank), dtype=np.int64)
    V = np.zeros((m * p, total_rank), dtype=np.int64)
    W = np.zeros((p * n, total_rank), dtype=np.int64)
    
    col = 0

    def add_block(n_off, m_off, p_off, n_block, m_block, p_block, u_vecs, v_vecs, w_vecs):
        nonlocal col
        # u_vecs, v_vecs, w_vecs are lists of tuples (local_idx, value)
        # Or just arrays. Let's use lists of (idx, val).
        # Actually, passing pre-computed local vectors is cleaner.
        
        # For each term r in this block:
        for r in range(len(u_vecs)):
            # u_vecs[r] is list of (local_idx, val)
            for u_idx, val in u_vecs[r]:
                global_u_idx = (n_off + u_idx // m_block) * m + (m_off + u_idx % m_block)
                # Wait, local U index is i*m_block + j.
                # So i = u_idx // m_block, j = u_idx % m_block.
                U[global_u_idx, col] += val
                
            for v_idx, val in v_vecs[r]:
                global_v_idx = (m_off + v_idx // p_block) * p + (p_off + v_idx % p_block)
                V[global_v_idx, col] += val
                
            for w_idx, val in w_vecs[r]:
                global_w_idx = (p_off + w_idx // n_block) * n + (n_off + w_idx % n_block)
                W[global_w_idx, col] += val
                
            col += 1

    # 1. Block 1: 1x4x6 at (0,0,0)
    # Rank 24. Dot product type.
    # n=1, m=4, p=6
    u_vecs = []
    v_vecs = []
    w_vecs = []
    for j in range(4):
        for k in range(6):
            u_vecs.append([(j, 1)]) # Local U idx j
            v_vecs.append([(j * 6 + k, 1)]) # Local V idx j*6+k
            w_vecs.append([(k * 1 + 0, 1)]) # Local W idx k
    add_block(0, 0, 0, 1, 4, 6, u_vecs, v_vecs, w_vecs)

    # 2. Block 2b: 2x1x6 at (1, 3, 0)
    # n_off=1, m_off=3, p_off=0
    # n=2, m=1, p=6
    # Rank 12.
    u_vecs = []
    v_vecs = []
    w_vecs = []
    for i in range(2):
        for k in range(6):
            u_vecs.append([(i * 1 + 0, 1)]) # Local U idx i
            v_vecs.append([(0 * 6 + k, 1)]) # Local V idx k
            w_vecs.append([(k * 2 + i, 1)]) # Local W idx k*2+i
    add_block(1, 3, 0, 2, 1, 6, u_vecs, v_vecs, w_vecs)

    # 3. Block 2a,1: 2x1x6 at (1, 0, 0)
    # n_off=1, m_off=0, p_off=0
    # n=2, m=1, p=6
    # Rank 12.
    u_vecs = []
    v_vecs = []
    w_vecs = []
    for i in range(2):
        for k in range(6):
            u_vecs.append([(i * 1 + 0, 1)])
            v_vecs.append([(0 * 6 + k, 1)])
            w_vecs.append([(k * 2 + i, 1)])
    add_block(1, 0, 0, 2, 1, 6, u_vecs, v_vecs, w_vecs)

    # 4. Block 2a,2: 2x2x6 at (1, 1, 0)
    # This is 3 blocks of 2x2x2.
    # Sub-block 4a: 2x2x2 at (1, 1, 0)
    # Sub-block 4b: 2x2x2 at (1, 1, 2)
    # Sub-block 4c: 2x2x2 at (1, 1, 4)
    
    # Strassen 2x2x2 Vectors
    # Local n=2, m=2, p=2
    # U idx: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V idx: 00, 01, 10, 11 -> 0, 1, 2, 3
    # W idx: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    
    strassen_u = [
        [(0, 1), (3, 1)],       # M1: A00+A11
        [(2, 1), (3, 1)],       # M2: A10+A11
        [(0, 1)],               # M3: A00
        [(3, 1)],               # M4: A11
        [(0, 1), (1, 1)],       # M5: A00+A01
        [(2, 1), (0, -1)],      # M6: A10-A00
        [(1, 1), (3, -1)]       # M7: A01-A11
    ]
    strassen_v = [
        [(0, 1), (3, 1)],       # B00+B11
        [(0, 1)],               # B00
        [(1, 1), (3, -1)],      # B01-B11
        [(2, 1), (0, -1)],      # B10-B00
        [(3, 1)],               # B11
        [(0, 1), (1, 1)],       # B00+B01
        [(2, 1), (3, 1)]        # B10+B11
    ]
    strassen_w = [
        [(0, 1), (3, 1)],       # C00+C11
        [(2, 1), (3, -1)],      # C10-C11
        [(1, 1), (3, 1)],       # C01+C11
        [(0, 1), (2, 1)],       # C00+C10
        [(1, 1), (0, -1)],      # C01-C00
        [(3, 1)],               # C11
        [(0, 1)]                # C00
    ]
    
    for p_sub in [0, 2, 4]:
        add_block(1, 1, p_sub, 2, 2, 2, strassen_u, strassen_v, strassen_w)

    # Check time
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
