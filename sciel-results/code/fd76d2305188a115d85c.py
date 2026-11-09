def construct():
    n, m, p = 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42

    # Strassen coefficients for A (2x2)
    # A indices in local 2x2 block: 0, 1, 2, 3 (row-major)
    # Global A index: i*m + j
    
    # Strassen coefficients for B (2x2)
    # B indices in local 2x2 block: 0, 1, 2, 3 (row-major)
    # Global B index: j*p + k
    
    # Strassen coefficients for C (2x2)
    # C indices in local 2x2 block: 0, 1, 2, 3 (row-major)
    # Global C index: k*n + i (column-major for n x p)
    
    # We have 2 blocks for A (x=0,1) and 3 blocks for B/C (y=0,1,2)
    
    # Define the 7 Strassen terms
    # Each term is a list of (coeff, index) for U, V, W
    # U is from A, V is from B, W is from C
    
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_terms = [
        # M1
        {
            'U': [(1, 0), (1, 3)], # a00 + a11
            'V': [(1, 0), (1, 3)], # b00 + b11
            'W': [(1, 0), (1, 3)]  # c00 + c11 (wait, W is output)
            # Output contributions:
            # C00 += 1
            # C11 += 1
        },
        # M2
        {
            'U': [(1, 2), (1, 3)], # a10 + a11
            'V': [(1, 1)],         # b01
            'W': [(1, 1), (1, 2)]  # C01 += 1, C10 += 1
        },
        # M3
        {
            'U': [(1, 0)],         # a00
            'V': [(1, 2), (-1, 3)],# b10 - b11
            'W': [(1, 1), (-1, 3)] # C01 += 1, C11 -= 1
        },
        # M4
        {
            'U': [(1, 3)],         # a11
            'V': [(1, 0), (-1, 1)],# b00 - b01
            'W': [(1, 0), (1, 2)]  # C00 += 1, C10 += 1
        },
        # M5
        {
            'U': [(1, 0), (1, 1)], # a00 + a01
            'V': [(1, 3)],         # b11
            'W': [(-1, 0), (1, 3)] # C00 -= 1, C11 += 1
        },
        # M6
        {
            'U': [(1, 2), (-1, 0)],# a10 - a00
            'V': [(1, 0), (1, 1)], # b00 + b01
            'W': [(1, 3)]          # C11 += 1
        },
        # M7
        {
            'U': [(1, 1), (-1, 3)],# a01 - a11
            'V': [(1, 2), (1, 3)], # b10 + b11
            'W': [(1, 0), (1, 3)]  # C00 += 1, C11 += 1
        }
    ]
    
    # Wait, the W part in the term definition above is not quite right.
    # The term is a rank-1 tensor $u \otimes v \otimes w$.
    # $w$ is a vector in the output space.
    # In Strassen, $M_k$ is a scalar.
    # The output $C$ is reconstructed by summing $M_k \times \text{coeff}$.
    # So for each Strassen term $k$, we have a vector $w_k$ in the output space.
    # $w_k$ has 1s or -1s in the positions corresponding to the output elements.
    
    # Let's re-define the terms properly.
    # Term k:
    # u_k: vector in A-space (size 4 for 2x2 block)
    # v_k: vector in B-space (size 4 for 2x2 block)
    # w_k: vector in C-space (size 4 for 2x2 block)
    
    terms = [
        # M1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # M2: u=(0,0,1,1), v=(0,1,0,0), w=(0,1,1,0)
        {'u': [0, 0, 1, 1], 'v': [0, 1, 0, 0], 'w': [0, 1, 1, 0]},
        # M3: u=(1,0,0,0), v=(0,0,1,-1), w=(0,1,0,-1)
        {'u': [1, 0, 0, 0], 'v': [0, 0, 1, -1], 'w': [0, 1, 0, -1]},
        # M4: u=(0,0,0,1), v=(1,-1,0,0), w=(1,0,1,0)
        {'u': [0, 0, 0, 1], 'v': [1, -1, 0, 0], 'w': [1, 0, 1, 0]},
        # M5: u=(1,1,0,0), v=(0,0,0,1), w=(-1,0,0,1)
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 0, 0, 1]},
        # M6: u=(0,-1,1,0), v=(1,1,0,0), w=(0,0,0,1)
        {'u': [0, -1, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        # M7: u=(0,1,0,-1), v=(0,0,1,1), w=(1,0,0,1)
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 1]}
    ]
    
    c_idx = 0
    for x in range(2): # A block index (0, 1)
        for y in range(3): # B/C block index (0, 1, 2)
            for t in terms:
                # Map local indices to global indices
                # A global index: i*m + j
                # Local i in {0,1}, local j in {0,1}
                # Global i = i_local (since n=2, no offset in i)
                # Global j = x*2 + j_local
                for li in range(2):
                    for lj in range(2):
                        val = t['u'][li*2 + lj]
                        if val != 0:
                            gi = li
                            gj = x*2 + lj
                            U[gi*m + gj][c_idx] += val
                
                # B global index: j*p + k
                # Local j in {0,1} (rows of B block), local k in {0,1} (cols of B block)
                # Global j = x*2 + j_local (Rows of B correspond to rows of A block? 
                # Wait. A is 2x4. B is 4x6.
                # A_i * B_j.
                # A block x uses rows 0..1 of A? No, A is 2x4.
                # We split A columns.
                # A_1 is cols 0,1. A_2 is cols 2,3.
                # So for x=0, we use A cols 0,1.
                # These correspond to B rows 0,1.
                # So global j for B = x*2 + j_local.
                # Global k for B = y*2 + k_local.
                for lj in range(2):
                    for lk in range(2):
                        val = t['v'][lj*2 + lk]
                        if val != 0:
                            gj = x*2 + lj
                            gk = y*2 + lk
                            V[gj*p + gk][c_idx] += val
                
                # C global index: k*n + i
                # Local i in {0,1} (rows of C), local k in {0,1} (cols of C)
                # Global i = li (rows of C are same as rows of A)
                # Global k = y*2 + lk
                for li in range(2):
                    for lk in range(2):
                        val = t['w'][li*2 + lk]
                        if val != 0:
                            gi = li
                            gk = y*2 + lk
                            W[gk*n + gi][c_idx] += val
                
                c_idx += 1

    return {"U": U, "V": V, "W": W}
