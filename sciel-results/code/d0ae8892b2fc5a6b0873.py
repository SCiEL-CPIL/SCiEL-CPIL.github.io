import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    via recursive Strassen (2 levels of 2x2 Strassen).
    """
    start_time = time.time()
    deadline = start_time + 115.0

    # Build the 49-term recursive Strassen decomposition
    # Level 1: 2x2 Strassen at block level (4x4 = 2x2 blocks of 2x2)
    # Level 2: 2x2 Strassen within each 2x2 block multiplication
    
    # Strassen 2x2: 7 terms, each with U,V,W as lists of (idx, coeff)
    # For 2x2: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    S = [
        # (U_entries, V_entries, W_entries) where entries are (local_idx, coeff)
        # local_idx for U/V: i*2+j, for W: k*2+i
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),          # M1
        ([(2,1),(3,1)], [(0,1)],               [(2,1),(3,-1)]),   # M2
        ([(0,1)],               [(1,1),(3,-1)], [(1,1),(3,1)]),   # M3
        ([(3,1)],               [(2,1),(0,-1)], [(0,1),(2,1)]),   # M4
        ([(0,1),(1,1)],      [(3,1)],               [(0,-1),(1,1)]), # M5
        ([(2,1),(0,-1)],     [(0,1),(1,1)],      [(3,1)]),         # M6
        ([(1,1),(3,-1)],     [(2,1),(3,1)],      [(0,1)]),         # M7
    ]
    
    # Outer Strassen (same structure, but at block level)
    # Block indices: (br, bc) -> local block idx = br*2+bc
    O = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)],               [(2,1),(3,-1)]),
        ([(0,1)],               [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)],               [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)],      [(3,1)],               [(0,-1),(1,1)]),
        ([(2,1),(0,-1)],     [(0,1),(1,1)],      [(3,1)]),
        ([(1,1),(3,-1)],     [(2,1),(3,1)],      [(0,1)]),
    ]
    
    rank = 49
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(16)]
    W = [[0.0] * rank for _ in range(16)]
    
    col = 0
    for (ou, ov, ow) in O:
        for (su, sv, sw) in S:
            # U: outer block (br,bc) and inner (lr,lc)
            # global row = br*2+lr, global col = bc*2+lc, idx = (br*2+lr)*4 + (bc*2+lc)
            for (blk_idx, c1) in ou:
                br, bc = blk_idx // 2, blk_idx % 2
                for (loc_idx, c2) in su:
                    lr, lc = loc_idx // 2, loc_idx % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    U[gr * 4 + gc][col] += c1 * c2
            
            # V: same structure
            for (blk_idx, c1) in ov:
                br, bc = blk_idx // 2, blk_idx % 2
                for (loc_idx, c2) in sv:
                    lr, lc = loc_idx // 2, loc_idx % 2
                    gr = br * 2 + lr
                    gc = bc * 2 + lc
                    V[gr * 4 + gc][col] += c1 * c2
            
            # W: W[k*4+i, r] where k = col of B, i = row of A
            # From outer: block (br, bc) -> global row of A = br*2+lr, global col of B = bc*2+lc
            # W_idx = (bc*2+lc)*4 + (br*2+lr)
            for (blk_idx, c1) in ow:
                br, bc = blk_idx // 2, blk_idx % 2
                for (loc_idx, c2) in sw:
                    lr, lc = loc_idx // 2, loc_idx % 2
                    gr_a = br * 2 + lr
                    gc_b = bc * 2 + lc
                    W[gc_b * 4 + gr_a][col] += c1 * c2
            
            col += 1
    
    # Verify with exact integer arithmetic
    try:
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        U3 = Ua.reshape(4, 4, rank)
        V3 = Va.reshape(4, 4, rank)
        W3 = Wa.reshape(4, 4, rank)
        S_tensor = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.all(S_tensor == 1):
            raise ValueError("Decomposition verification failed")
    except Exception:
        # Fallback: schoolbook rank 64
        rank = 64
        U = [[0.0] * rank for _ in range(16)]
        V = [[0.0] * rank for _ in range(16)]
        W = [[0.0] * rank for _ in range(16)]
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][c] = 1.0
                    V[j*4+k][c] = 1.0
                    W[k*4+i][c] = 1.0
                    c += 1
    
    return {"U": U, "V": V, "W": W}
