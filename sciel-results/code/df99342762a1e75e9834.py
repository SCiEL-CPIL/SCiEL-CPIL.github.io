"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Decompose the 2x3x10 matrix multiplication tensor.
    # Split the p=10 dimension into 5 blocks of p=2.
    # For each block, use Strassen for the 2x2x2 part (rank 7) and
    # 4 outer products for the 2x1x2 part (rank 4), total rank 11 per block.
    # Total rank = 5 * 11 = 55.
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55
    
    # For each block t = 0..4:
    #   Local A indices: i in {0,1}, k in {0,1,2} -> global U index: i*3 + k
    #   Local B indices: k in {0,1,2}, j_local in {0,1} -> global V index: k*10 + (2*t + j_local)
    #   Local C indices: j_local in {0,1}, i in {0,1} -> global W index: (2*t + j_local)*2 + i
    #
    # Strassen for 2x2x2 (A' is 2x2, B' is 2x2, C' is 2x2):
    #   M1 = (A00 + A11)(B00 + B11)
    #   M2 = (A10 + A11) B00
    #   M3 = A00 (B10 - B11)
    #   M4 = A11 (B01 - B00)
    #   M5 = (A00 + A01) B11
    #   M6 = (A10 - A00)(B00 + B01)
    #   M7 = (A01 - A11)(B10 + B11)
    #
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Outer product for a2 * b2^T (A row 2, B col 2):
    #   M8  = A02 * B20
    #   M9  = A02 * B21
    #   M10 = A12 * B20
    #   M11 = A12 * B21
    #
    #   C00 += M8, C01 += M9, C10 += M10, C11 += M11
    
    # Define the 11 terms: each is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (local_A_idx, coeff) where local_A_idx = i*3 + k
    # V_coeffs: list of (local_B_idx, coeff) where local_B_idx = k*10 + j_local (but we use local k, j_local)
    # Actually, let me think in terms of local indices.
    # For a given block t, global V index for (k, j_local) is k*10 + (2*t + j_local)
    # Global W index for (j_local, i) is (2*t + j_local)*2 + i
    
    # Let's define each term by (u_entries, v_entries, w_entries)
    # u_entries: dict mapping global U row -> coefficient
    # v_entries: dict mapping global V row -> coefficient  
    # w_entries: dict mapping global W row -> coefficient
    
    def add_term(col, u_entry, v_entry, w_entry):
        """u_entry, v_entry, w_entry are lists of (index, coeff)"""
        for idx, c in u_entry:
            U[idx][col] += c
        for idx, c in v_entry:
            V[idx][col] += c
        for idx, c in w_entry:
            W[idx][col] += c
    
    for t in range(5):
        base_col = 11 * t
        # Global row offsets
        # U: i*3 + k, i in {0,1}, k in {0,1,2}
        # V: k*10 + (2*t + j_local), k in {0,1,2}, j_local in {0,1}
        # W: (2*t + j_local)*2 + i, j_local in {0,1}, i in {0,1}
        
        # Helper to get global V row for local (k, jl)
        def vrow(k, jl):
            return k * 10 + (2 * t + jl)
        
        # Helper to get global W row for local (jl, i)
        def wrow(jl, i):
            return (2 * t + jl) * 2 + i
        
        # U rows for (i, k)
        def urow(i, k):
            return i * 3 + k
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # A00 = urow(0,0)=0, A11 = urow(1,1)=4
        # B00 = vrow(0,0), B11 = vrow(1,1)
        # C00 = wrow(0,0), C01 = wrow(1,0), C10 = wrow(0,1), C11 = wrow(1,1)
        col = base_col + 0
        add_term(col, 
                 [(0, 1), (4, 1)],
                 [(vrow(0,0), 1), (vrow(1,1), 1)],
                 [(wrow(0,0), 1), (wrow(1,1), 1)])
        
        # Term 2: M2 = (A10 + A11) B00
        # A10 = urow(1,0)=3, A11 = urow(1,1)=4
        # B00 = vrow(0,0)
        # C10 = wrow(0,1)
        col = base_col + 1
        add_term(col,
                 [(3, 1), (4, 1)],
                 [(vrow(0,0), 1)],
                 [(wrow(0,1), 1)])
        
        # Term 3: M3 = A00 (B10 - B11)
        # A00 = urow(0,0)=0
        # B10 = vrow(1,0), B11 = vrow(1,1)
        # C01 = wrow(1,0), C11 = wrow(1,1)
        col = base_col + 2
        add_term(col,
                 [(0, 1)],
                 [(vrow(1,0), 1), (vrow(1,1), -1)],
                 [(wrow(1,0), 1), (wrow(1,1), 1)])
        
        # Term 4: M4 = A11 (B01 - B00)
        # A11 = urow(1,1)=4
        # B01 = vrow(0,1), B00 = vrow(0,0)
        # C00 = wrow(0,0), C10 = wrow(0,1)
        col = base_col + 3
        add_term(col,
                 [(4, 1)],
                 [(vrow(0,1), 1), (vrow(0,0), -1)],
                 [(wrow(0,0), 1), (wrow(0,1), 1)])
        
        # Term 5: M5 = (A00 + A01) B11
        # A00 = urow(0,0)=0, A01 = urow(0,1)=1
        # B11 = vrow(1,1)
        # C00 = wrow(0,0), C01 = wrow(1,0)
        col = base_col + 4
        add_term(col,
                 [(0, 1), (1, 1)],
                 [(vrow(1,1), 1)],
                 [(wrow(0,0), -1), (wrow(1,0), 1)])
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        # A10 = urow(1,0)=3, A00 = urow(0,0)=0
        # B00 = vrow(0,0), B01 = vrow(0,1)
        # C11 = wrow(1,1)
        col = base_col + 5
        add_term(col,
                 [(3, 1), (0, -1)],
                 [(vrow(0,0), 1), (vrow(0,1), 1)],
                 [(wrow(1,1), 1)])
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        # A01 = urow(0,1)=1, A11 = urow(1,1)=4
        # B10 = vrow(1,0), B11 = vrow(1,1)
        # C00 = wrow(0,0)
        col = base_col + 6
        add_term(col,
                 [(1, 1), (4, -1)],
                 [(vrow(1,0), 1), (vrow(1,1), 1)],
                 [(wrow(0,0), 1)])
        
        # Term 8: M8 = A02 * B20
        # A02 = urow(0,2)=2
        # B20 = vrow(2,0)
        # C00 = wrow(0,0)
        col = base_col + 7
        add_term(col,
                 [(2, 1)],
                 [(vrow(2,0), 1)],
                 [(wrow(0,0), 1)])
        
        # Term 9: M9 = A02 * B21
        # A02 = urow(0,2)=2
        # B21 = vrow(2,1)
        # C01 = wrow(1,0)
        col = base_col + 8
        add_term(col,
                 [(2, 1)],
                 [(vrow(2,1), 1)],
                 [(wrow(1,0), 1)])
        
        # Term 10: M10 = A12 * B20
        # A12 = urow(1,2)=5
        # B20 = vrow(2,0)
        # C10 = wrow(0,1)
        col = base_col + 9
        add_term(col,
                 [(5, 1)],
                 [(vrow(2,0), 1)],
                 [(wrow(0,1), 1)])
        
        # Term 11: M11 = A12 * B21
        # A12 = urow(1,2)=5
        # B21 = vrow(2,1)
        # C11 = wrow(1,1)
        col = base_col + 10
        add_term(col,
                 [(5, 1)],
                 [(vrow(2,1), 1)],
                 [(wrow(1,1), 1)])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
