"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partitions for indices 0..4
    # Sizes: 2, 2, 1
    # P_A: {0,1}, {2,3}, {4}
    # P_B: {0,1}, {2,3}, {4}
    # P_C: {0,1}, {2,3}, {4}
    
    partitions = [
        [0, 1],
        [2, 3],
        [4]
    ]
    
    # Offsets
    offsets = [0, 2, 4]
    sizes = [2, 2, 1]
    
    # Base decompositions
    # Each is a dict with 'U', 'V', 'W' lists of lists, and 'n', 'm', 'p'
    
    def base_111():
        return {
            'n': 1, 'm': 1, 'p': 1,
            'U': [[1]],
            'V': [[1]],
            'W': [[1]]
        }

    def base_211():
        # A(2x1) B(1x1) -> C(2x1)
        # U(2x1, 2), V(1x1, 2), W(1x2, 2)
        # W index k*n+i. n=2, p=1.
        # C0 (i=0,k=0) -> idx 0
        # C1 (i=1,k=0) -> idx 1
        return {
            'n': 2, 'm': 1, 'p': 1,
            'U': [[1, 0], [0, 1]],
            'V': [[1, 1]],
            'W': [[1, 0], [0, 1]] # W[0,0]=1, W[1,0]=0? No, W is list of rows?
                                   # Shape (p*n, rank).
                                   # Row 0: k=0, i=0 -> 1
                                   # Row 1: k=0, i=1 -> 1
                                   # So W = [[1, 0], [0, 1]] is correct if rows are k*n+i.
        }
        
    def base_121():
        # A(1x2) B(2x1) -> C(1x1)
        # U(1x2, 2), V(2x1, 2), W(1x1, 2)
        return {
            'n': 1, 'm': 2, 'p': 1,
            'U': [[1, 1]],
            'V': [[1, 0], [0, 1]],
            'W': [[1, 1]]
        }

    def base_112():
        # A(1x1) B(1x2) -> C(1x2)
        # U(1x1, 2), V(1x2, 2), W(2x1, 2)
        # W index k*1+i. n=1.
        # C0 (k=0,i=0) -> 0
        # C1 (k=1,i=0) -> 1
        return {
            'n': 1, 'm': 1, 'p': 2,
            'U': [[1, 1]],
            'V': [[1, 0], [0, 1]],
            'W': [[1, 0], [0, 1]]
        }

    def base_221():
        # A(2x2) B(2x1) -> C(2x1)
        # Rank 4
        # U(4x2, 4), V(2x1, 4), W(1x2, 4)
        # W idx: 0 (i=0,k=0), 1 (i=1,k=0)
        return {
            'n': 2, 'm': 2, 'p': 1,
            'U': [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
            'V': [[1,0,1,0], [0,1,0,1]],
            'W': [[1,1,0,0], [0,0,1,1]]
        }
        
    def base_212():
        # A(2x1) B(1x2) -> C(2x2)
        # Rank 4
        # U(2x1, 4), V(1x2, 4), W(2x2, 4)
        # W idx: 0(0,0), 1(1,0), 2(0,1), 3(1,1)
        return {
            'n': 2, 'm': 1, 'p': 2,
            'U': [[1,1,0,0], [0,0,1,1]],
            'V': [[1,0,1,0], [0,1,0,1]],
            'W': [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
        }
        
    def base_122():
        # A(1x2) B(2x2) -> C(1x2)
        # Rank 4
        # U(1x2, 4), V(2x2, 4), W(2x1, 4)
        # W idx: 0(0,0), 1(1,0)
        return {
            'n': 1, 'm': 2, 'p': 2,
            'U': [[1,1,0,0]],
            'V': [[1,0,1,0], [0,1,0,1]],
            'W': [[1,0,1,0], [0,1,0,1]]
        }
        
    def base_222():
        # A(2x2) B(2x2) -> C(2x2)
        # Rank 7 (Strassen)
        # U(4x2, 7), V(2x2, 7), W(2x2, 7)
        # W idx: 0(0,0), 1(1,0), 2(0,1), 3(1,1)
        
        # M1: (A00+A11)(B00+B11) -> C00 + C11
        # U: 1,0,0,1 (A00, A01, A10, A11)
        # V: 1,0,0,1
        # W: 1,0,0,1 (C00, C10, C01, C11) -> idx 0, 1, 2, 3
        # W[0]=1, W[3]=1
        
        # M2: (A10+A11)B00 -> C10 - C11
        # U: 0,0,1,1
        # V: 1,0,0,0
        # W: 0,1,0,-1
        
        # M3: A00(B10-B11) -> C01 + C11
        # U: 1,0,0,0
        # V: 0,0,1,-1
        # W: 0,0,1,1
        
        # M4: A11(B01-B00) -> C00 + C10
        # U: 0,0,0,1
        # V: -1,1,0,0
        # W: 1,1,0,0
        
        # M5: (A00+A01)B11 -> -C00 + C01
        # U: 1,1,0,0
        # V: 0,0,0,1
        # W: -1,0,1,0
        
        # M6: (A10-A00)(B00+B01) -> C11
        # U: -1,0,1,0
        # V: 1,1,0,0
        # W: 0,0,0,1
        
        # M7: (A01-A11)(B10+B11) -> C00
        # U: 0,1,0,-1
        # V: 0,0,1,1
        # W: 1,0,0,0
        
        U = [
            [1, 0, 1, 0, 1, -1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1],  # A01
            [0, 1, 0, 0, 0, 1, 0],  # A10
            [1, 1, 0, 1, 0, 0, -1]  # A11
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0], # B00
            [0, 0, 0, 1, 0, 1, 0],  # B01
            [0, 0, 1, 0, 0, 0, 1],  # B10
            [1, 0, -1, 0, 1, 0, 1]  # B11
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 1, 0, 1, 0, 0, 0],  # C10
            [0, 0, 1, 0, 1, 0, 0],  # C01
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
        return {
            'n': 2, 'm': 2, 'p': 2,
            'U': U,
            'V': V,
            'W': W
        }

    base_decomps = {
        (1,1,1): base_111,
        (2,1,1): base_211,
        (1,2,1): base_121,
        (1,1,2): base_112,
        (2,2,1): base_221,
        (2,1,2): base_212,
        (1,2,2): base_122,
        (2,2,2): base_222
    }

    # Accumulate global U, V, W
    # n=5, m=5, p=5
    total_rank = 0
    # Pre-calculate total rank
    for a in range(3):
        for b in range(3):
            for c in range(3):
                na, mb, pc = sizes[a], sizes[b], sizes[c]
                total_rank += len(base_decomps[(na,mb,pc)]()['U'][0])

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                na, mb, pc = sizes[a], sizes[b], sizes[c]
                dec = base_decomps[(na, mb, pc)]()
                rank = len(dec['U'][0])
                
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                for r in range(rank):
                    # U local (i', j') -> global (i, j)
                    for i_prime in range(na):
                        for j_prime in range(mb):
                            val = dec['U'][i_prime * mb + j_prime][r]
                            if val != 0:
                                i_global = off_a + i_prime
                                j_global = off_b + j_prime
                                U[i_global * m + j_global][col_offset + r] += val
                                
                    # V local (j', k') -> global (j, k)
                    for j_prime in range(mb):
                        for k_prime in range(pc):
                            val = dec['V'][j_prime * pc + k_prime][r]
                            if val != 0:
                                j_global = off_b + j_prime
                                k_global = off_c + k_prime
                                V[j_global * p + k_global][col_offset + r] += val
                                
                    # W local (k', i') -> global (k, i)
                    # W index is k * n + i
                    for k_prime in range(pc):
                        for i_prime in range(na):
                            val = dec['W'][k_prime * na + i_prime][r]
                            if val != 0:
                                k_global = off_c + k_prime
                                i_global = off_a + i_prime
                                W[k_global * n + i_global][col_offset + r] += val
                                                        
                col_offset += rank

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
