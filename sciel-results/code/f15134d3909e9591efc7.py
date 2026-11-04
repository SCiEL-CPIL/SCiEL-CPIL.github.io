"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    columns = []  # Each entry is (U_col, V_col, W_col) as sparse dicts
    
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    def add_laderman_2x3x4(n_start, m_start, p_start):
        """Add Laderman's rank-14 decomposition for 2x3x4 block."""
        def ug(i, j): return (n_start + i) * m + (m_start + j)
        def vg(j, k): return (m_start + j) * p + (p_start + k)
        def wg(i, k): return (p_start + k) * n + (n_start + i)
        
        # 14 bilinear forms for 2x3x4 matrix multiplication
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Products:
        terms = [
            # p0 = A[0,0]*B[0,0]
            ({ug(0,0): 1}, {vg(0,0): 1}, {wg(0,0): 1}),
            # p1 = A[0,0]*B[0,1]
            ({ug(0,0): 1}, {vg(0,1): 1}, {wg(0,1): 1}),
            # p2 = A[0,1]*B[1,0]
            ({ug(0,1): 1}, {vg(1,0): 1}, {wg(0,2): 1}),
            # p3 = A[0,1]*B[1,1]
            ({ug(0,1): 1}, {vg(1,1): 1}, {wg(0,3): 1}),
            # p4 = A[0,2]*B[2,0]
            ({ug(0,2): 1}, {vg(2,0): 1}, {wg(0,0): 1, wg(1,0): 1}),
            # p5 = A[0,2]*B[2,1]
            ({ug(0,2): 1}, {vg(2,1): 1}, {wg(0,1): 1, wg(1,1): 1}),
            # p6 = A[1,0]*B[0,2]
            ({ug(1,0): 1}, {vg(0,2): 1}, {wg(0,2): 1, wg(1,2): 1}),
            # p7 = A[1,0]*B[0,3]
            ({ug(1,0): 1}, {vg(0,3): 1}, {wg(0,3): 1, wg(1,3): 1}),
            # p8 = A[1,1]*B[1,2]
            ({ug(1,1): 1}, {vg(1,2): 1}, {wg(0,2): 1, wg(1,2): 1}),
            # p9 = A[1,1]*B[1,3]
            ({ug(1,1): 1}, {vg(1,3): 1}, {wg(0,3): 1, wg(1,3): 1}),
            # p10 = A[1,2]*B[2,2]
            ({ug(1,2): 1}, {vg(2,2): 1}, {wg(0,2): 1, wg(1,2): 1}),
            # p11 = A[1,2]*B[2,3]
            ({ug(1,2): 1}, {vg(2,3): 1}, {wg(0,3): 1, wg(1,3): 1}),
            # p12 = (A[0,0]+A[1,0]+A[0,1]+A[1,1])*(B[0,0]+B[1,0]+B[0,1]+B[1,1])
            ({ug(0,0): 1, ug(1,0): 1, ug(0,1): 1, ug(1,1): 1},
             {vg(0,0): 1, vg(1,0): 1, vg(0,1): 1, vg(1,1): 1}, {}),
            # p13 = (A[0,2]+A[1,2])*(B[2,0]+B[2,1]+B[2,2]+B[2,3])
            ({ug(0,2): 1, ug(1,2): 1},
             {vg(2,0): 1, vg(2,1): 1, vg(2,2): 1, vg(2,3): 1}, {}),
        ]
        
        # Now we need to express each C[i,k] as a linear combination of the 14 products.
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        #         = p0 + p2 + p4
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        #         = p1 + p3 + p5
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
        # C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3] + A[0,2]*B[2,3]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
        # C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3] + A[1,2]*B[2,3]
        
        # This construction above doesn't work because the w coefficients are wrong.
        # Let me use a different, verified approach.
        
        # Clear the terms and use a known valid rank-14 decomposition.
        # After careful derivation, here's a valid construction:
        
        # I'll fall back to schoolbook for correctness
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    add_term({ug(i, j): 1}, {vg(j, k): 1}, {wg(i, k): 1})
    
    # Block partitioning: n=3 -> [1,2], m=4 -> [3,1], p=8 -> [4,4]
    
    # Row 0 of A (n=0):
    add_schoolblock(0, 0, 0, 1, 3, 4)  # n=[0], m=[0,1,2], p=[0,1,2,3]: 12 terms
    add_schoolblock(0, 3, 0, 1, 1, 4)  # n=[0], m=[3], p=[0,1,2,3]: 4 terms
    add_schoolblock(0, 0, 4, 1, 3, 4)  # n=[0], m=[0,1,2], p=[4,5,6,7]: 12 terms
    add_schoolblock(0, 3, 4, 1, 1, 4)  # n=[0], m=[3], p=[4,5,6,7]: 4 terms
    
    # Rows 1-2 of A (n=[1,2]):
    add_laderman_2x3x4(1, 0, 0)   # n=[1,2], m=[0,1,2], p=[0,1,2,3]: 24 terms (schoolbook fallback)
    add_schoolblock(1, 3, 0, 2, 1, 4)  # n=[1,2], m=[3], p=[0,1,2,3]: 8 terms
    add_laderman_2x3x4(1, 0, 4)   # n=[1,2], m=[0,1,2], p=[4,5,6,7]: 24 terms (schoolbook fallback)
    add_schoolblock(1, 3, 4, 2, 1, 4)  # n=[1,2], m=[3], p=[4,5,6,7]: 8 terms
    
    rank = len(columns)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col_idx, (u_dict, v_dict, w_dict) in enumerate(columns):
        for idx, val in u_dict.items():
            U[idx][col_idx] = val
        for idx, val in v_dict.items():
            V[idx][col_idx] = val
        for idx, val in w_dict.items():
            W[idx][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
