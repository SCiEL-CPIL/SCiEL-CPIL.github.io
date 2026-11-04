import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 component
    def add_rank1(u_idx_vals, v_idx_vals, w_idx_vals):
        nonlocal col
        for idx, val in u_idx_vals.items():
            U[idx, col] += val
        for idx, val in v_idx_vals.items():
            V[idx, col] += val
        for idx, val in w_idx_vals.items():
            W[idx, col] += val
        col += 1
        
    # Strassen 2x2x2 decomposition
    # Local indices for 2x2 block:
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    add_rank1({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
    
    # M2 = (a10+a11)b00 -> C10 += 1, C11 -= 1
    add_rank1({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
    
    # M3 = a00(b01-b11) -> C01 += 1, C11 += 1
    add_rank1({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
    
    # M4 = a11(b10-b00) -> C00 += 1, C10 += 1
    add_rank1({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
    
    # M5 = (a00+a01)b11 -> C00 -= 1, C01 += 1
    add_rank1({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
    
    # M6 = (a10-a00)(b00+b10) -> C11 += 1
    add_rank1({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1})
    
    # M7 = (a01-a11)(b10+b11) -> C00 += 1
    add_rank1({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    # Schoolbook 2x1x2 decomposition
    # A: col 1 (local index 1 in 2x1? No, 2x1 has indices 0,1)
    # B: row 0 (local index 0 in 1x2? No, 1x2 has indices 0,1)
    # Wait, in 2x1x2:
    # A is 2x1. Indices 0,1.
    # B is 1x2. Indices 0,1.
    # C is 2x2. Indices 0,1,2,3.
    # C00 = A0*B0
    add_rank1({0: 1}, {0: 1}, {0: 1})
    # C01 = A0*B1
    add_rank1({0: 1}, {1: 1}, {1: 1})
    # C10 = A1*B0
    add_rank1({1: 1}, {0: 1}, {2: 1})
    # C11 = A1*B1
    add_rank1({1: 1}, {1: 1}, {3: 1})
    
    # This was for ONE 2x3x2 block.
    # We need to repeat this 3 times for the 3 blocks of p=6.
    # The above add_rank1 calls used LOCAL indices.
    # I need to map them to GLOBAL indices.
    
    # Let's restart the construction properly with global mapping.
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    def add_global_rank1(u_map, v_map, w_map):
        nonlocal col
        for g_idx, val in u_map.items():
            U[g_idx, col] += val
        for g_idx, val in v_map.items():
            V[g_idx, col] += val
        for g_idx, val in w_map.items():
            W[g_idx, col] += val
        col += 1
        
    # Loop over 3 blocks of p (size 2 each)
    for block_k in range(3):
        # p indices for this block: k_start = block_k * 2
        k_start = block_k * 2
        
        # 1. Strassen part for A[0:2, 0:2] * B[0:2, k_start:k_start+2]
        # Local A indices: (i, j) where i in 0,1; j in 0,1
        # Local B indices: (j, k) where j in 0,1; k in 0,1 (relative to block)
        # Local C indices: (k, i) where k in 0,1; i in 0,1
        
        # Global U index: i * 3 + j
        # Global V index: j * 6 + (k_start + k)
        # Global W index: (k_start + k) * 2 + i
        
        def get_u(i, j): return i * 3 + j
        def get_v(j, k_rel): return j * 6 + (k_start + k_rel)
        def get_w(k_rel, i): return (k_start + k_rel) * 2 + i
        
        # M1
        add_global_rank1(
            {get_u(0,0): 1, get_u(1,1): 1},
            {get_v(0,0): 1, get_v(1,1): 1},
            {get_w(0,0): 1, get_w(1,1): 1}
        )
        # M2
        add_global_rank1(
            {get_u(1,0): 1, get_u(1,1): 1},
            {get_v(0,0): 1},
            {get_w(0,1): 1, get_w(1,1): -1}
        )
        # M3
        add_global_rank1(
            {get_u(0,0): 1},
            {get_v(0,1): 1, get_v(1,1): -1},
            {get_w(1,0): 1, get_w(1,1): 1}
        )
        # M4
        add_global_rank1(
            {get_u(1,1): 1},
            {get_v(1,0): 1, get_v(0,0): -1},
            {get_w(0,0): 1, get_w(0,1): 1}
        )
        # M5
        add_global_rank1(
            {get_u(0,0): 1, get_u(0,1): 1},
            {get_v(1,1): 1},
            {get_w(0,0): -1, get_w(1,0): 1}
        )
        # M6
        add_global_rank1(
            {get_u(1,0): 1, get_u(0,0): -1},
            {get_v(0,0): 1, get_v(1,0): 1},
            {get_w(1,1): 1}
        )
        # M7
        add_global_rank1(
            {get_u(0,1): 1, get_u(1,1): -1},
            {get_v(1,0): 1, get_v(1,1): 1},
            {get_w(0,0): 1}
        )
        
        # 2. Schoolbook part for A[0:2, 2:3] * B[2:3, k_start:k_start+2]
        # A index: i * 3 + 2
        # B index: 2 * 6 + (k_start + k_rel)
        # C index: (k_start + k_rel) * 2 + i
        
        def get_u2(i): return i * 3 + 2
        def get_v2(k_rel): return 2 * 6 + (k_start + k_rel)
        def get_w2(k_rel, i): return (k_start + k_rel) * 2 + i
        
        # C00 = A0 * B0
        add_global_rank1(
            {get_u2(0): 1},
            {get_v2(0): 1},
            {get_w2(0,0): 1}
        )
        # C01 = A0 * B1
        add_global_rank1(
            {get_u2(0): 1},
            {get_v2(1): 1},
            {get_w2(1,0): 1}
        )
        # C10 = A1 * B0
        add_global_rank1(
            {get_u2(1): 1},
            {get_v2(0): 1},
            {get_w2(0,1): 1}
        )
        # C11 = A1 * B1
        add_global_rank1(
            {get_u2(1): 1},
            {get_v2(1): 1},
            {get_w2(1,1): 1}
        )
        
    return {"U": U, "V": V, "W": W}
