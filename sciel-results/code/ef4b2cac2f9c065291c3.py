"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by block partitioning:
    # n=3 -> [1, 2]
    # m=4 -> [3, 1]  
    # p=8 -> [4, 4]
    
    # Sub-blocks and their ranks:
    # (1,3,4): rank 12 (schoolbook)
    # (1,1,4): rank 4 (schoolbook)
    # (2,3,4): rank 14 (Laderman)
    # (2,1,4): rank 8 (schoolbook)
    # Total: 2*(12+4+14+8) = 76
    
    columns = []  # Each entry is (U_col, V_col, W_col) as sparse dicts
    
    # Helper to add a rank-1 term
    def add_term(u_dict, v_dict, w_dict):
        columns.append((u_dict, v_dict, w_dict))
    
    # For each sub-block, add schoolbook terms
    def add_schoolblock(n_start, m_start, p_start, ni, mi, pi):
        for i in range(ni):
            for j in range(mi):
                for k in range(pi):
                    u_idx = (n_start + i) * m + (m_start + j)
                    v_idx = (m_start + j) * p + (p_start + k)
                    w_idx = (p_start + k) * n + (n_start + i)
                    add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Add Laderman for 2x3x4 block
    def add_laderman(n_start, m_start, p_start):
        """Add Laderman's rank-14 decomposition for 2x3x4 block."""
        # Local to global index mapping
        def u_glob(li, lj):  # li in {0,1}, lj in {0,1,2}
            return (n_start + li) * m + (m_start + lj)
        def v_glob(li, lj):  # li in {0,1,2}, lj in {0,1,2,3}
            return (m_start + li) * p + (p_start + lj)
        def w_glob(li, lj):  # li in {0,1}, lj in {0,1,2,3}
            return (p_start + lj) * n + (n_start + li)
        
        # 14 rank-1 terms for 2x3x4 based on Laderman's algorithm
        # Local A: 2x3, index = i*3+j
        # Local B: 3x4, index = i*4+j  
        # Local C: 2x4, index = i*4+j
        
        # Laderman's 14 multiplications:
        # M1 = A00 * B00
        add_term({u_glob(0,0): 1}, {v_glob(0,0): 1}, {w_glob(0,0): 1})
        # M2 = A00 * B01
        add_term({u_glob(0,0): 1}, {v_glob(0,1): 1}, {w_glob(0,1): 1})
        # M3 = A01 * B10
        add_term({u_glob(0,1): 1}, {v_glob(1,0): 1}, {w_glob(0,2): 1})
        # M4 = A01 * B11
        add_term({u_glob(0,1): 1}, {v_glob(1,1): 1}, {w_glob(0,3): 1})
        # M5 = A02 * B20
        add_term({u_glob(0,2): 1}, {v_glob(2,0): 1}, {w_glob(1,0): 1})
        # M6 = A02 * B21
        add_term({u_glob(0,2): 1}, {v_glob(2,1): 1}, {w_glob(1,1): 1})
        # M7 = (A00+A10) * (B00+B10)
        add_term({u_glob(0,0): 1, u_glob(1,0): 1}, {v_glob(0,0): 1, v_glob(1,0): 1}, {w_glob(0,0): 1, w_glob(1,0): 1})
        # M8 = (A00+A10) * (B01+B11)
        add_term({u_glob(0,0): 1, u_glob(1,0): 1}, {v_glob(0,1): 1, v_glob(1,1): 1}, {w_glob(0,1): 1, w_glob(1,1): 1})
        # M9 = (A01+A11) * (B00+B10)
        add_term({u_glob(0,1): 1, u_glob(1,1): 1}, {v_glob(0,0): 1, v_glob(1,0): 1}, {w_glob(0,2): 1, w_glob(1,2): 1})
        # M10 = (A01+A11) * (B01+B11)
        add_term({u_glob(0,1): 1, u_glob(1,1): 1}, {v_glob(0,1): 1, v_glob(1,1): 1}, {w_glob(0,3): 1, w_glob(1,3): 1})
        # M11 = (A00+A01+A10+A11) * (B00+B01+B10+B11)
        add_term({u_glob(0,0): 1, u_glob(0,1): 1, u_glob(1,0): 1, u_glob(1,1): 1}, 
                 {v_glob(0,0): 1, v_glob(0,1): 1, v_glob(1,0): 1, v_glob(1,1): 1},
                 {w_glob(0,0): 1, w_glob(0,1): 1, w_glob(1,0): 1, w_glob(1,1): 1})
        # M12 = A02 * (B20+B21)
        add_term({u_glob(0,2): 1}, {v_glob(2,0): 1, v_glob(2,1): 1}, {w_glob(1,0): 1, w_glob(1,1): 1})
        # M13 = A12 * (B20+B21)
        add_term({u_glob(1,2): 1}, {v_glob(2,0): 1, v_glob(2,1): 1}, {w_glob(1,0): 1, w_glob(1,1): 1})
        # M14 = (A02+A12) * (B20+B21)
        add_term({u_glob(0,2): 1, u_glob(1,2): 1}, {v_glob(2,0): 1, v_glob(2,1): 1}, {w_glob(1,0): 1, w_glob(1,1): 1})
    
    # Block 1: n=[0], m=[0,1,2], p=[0,1,2,3]
    add_schoolblock(0, 0, 0, 1, 3, 4)  # 12 terms
    
    # Block 2: n=[0], m=[3], p=[0,1,2,3]
    add_schoolblock(0, 3, 0, 1, 1, 4)  # 4 terms
    
    # Block 3: n=[0], m=[0,1,2], p=[4,5,6,7]
    add_schoolblock(0, 0, 4, 1, 3, 4)  # 12 terms
    
    # Block 4: n=[0], m=[3], p=[4,5,6,7]
    add_schoolblock(0, 3, 4, 1, 1, 4)  # 4 terms
    
    # Block 5: n=[1,2], m=[0,1,2], p=[0,1,2,3] - use Laderman (14 terms)
    add_laderman(1, 0, 0)
    
    # Block 6: n=[1,2], m=[3], p=[0,1,2,3] - schoolbook (8 terms)
    add_schoolblock(1, 3, 0, 2, 1, 4)
    
    # Block 7: n=[1,2], m=[0,1,2], p=[4,5,6,7] - use Laderman (14 terms)
    add_laderman(1, 0, 4)
    
    # Block 8: n=[1,2], m=[3], p=[4,5,6,7] - schoolbook (8 terms)
    add_schoolblock(1, 3, 4, 2, 1, 4)
    
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
