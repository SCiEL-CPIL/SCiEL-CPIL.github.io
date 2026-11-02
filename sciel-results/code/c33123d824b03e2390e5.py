"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-20 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 4 groups of 3. Apply a rank-5 decomposition
    for 2x2 * 2x3 to each group. Total rank = 4 * 5 = 20.
    
    The rank-5 decomposition for 2x2 * 2x3 uses these 5 products:
      P0 = (A00 + A01) * (B00 + B01)
      P1 = (A00 + A01) * (B00 + B02)
      P2 = A00 * B01
      P3 = A00 * B02
      P4 = A01 * B02
      
    Then:
      C00 = P0 - P2 - P3
      C01 = P2 + P3
      C02 = P1 - P3
      C10 = P0 - P1 - P2
      C11 = P1 - P4
      C12 = P4
    """
    start_time = time.time()
    deadline = start_time + 130.0
    n, m, p = DIMENSIONS
    num_groups = p // 3
    rank = num_groups * 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each group of 3 columns, define the 5 products and output contributions
    # Product definitions: (U_entries, V_entries)
    # U_entries: list of (row_index, coeff) where row = i*m + j
    # V_entries: list of (row_index, coeff) where row = j*p + k
    
    for g in range(num_groups):
        base_col = 3 * g
        
        # Product 0: (A00 + A01) * (B00 + B01)
        # U: A00=1, A01=1 -> rows 0, 1
        # V: B00=1, B01=1 -> rows base_col, base_col+1
        u_entries_0 = [(0, 1), (1, 1)]
        v_entries_0 = [(base_col, 1), (base_col + 1, 1)]
        # Output: C00 += -1*P0? No, C00 = P0 - P2 - P3, so P0 contributes +1 to C00
        # C00 index: col=base_col, i=0 -> W[base_col*2 + 0]
        # C10 index: col=base_col, i=1 -> W[base_col*2 + 1]
        # C00 = P0 - P2 - P3 => P0 gives +1 to C00
        # C10 = P0 - P1 - P2 => P0 gives +1 to C10
        w_entries_0 = [(base_col * n + 0, 1), (base_col * n + 1, 1)]
        
        col = g * 5
        for (r, c) in u_entries_0:
            U[r][col] = c
        for (r, c) in v_entries_0:
            V[r][col] = c
        for (r, c) in w_entries_0:
            W[r][col] = c
        
        # Product 1: (A00 + A01) * (B00 + B02)
        # U: A00=1, A01=1
        # V: B00=1, B02=1 -> rows base_col, base_col+2
        u_entries_1 = [(0, 1), (1, 1)]
        v_entries_1 = [(base_col, 1), (base_col + 2, 1)]
        # C02 = P1 - P3 => P1 gives +1 to C02
        # C10 = P0 - P1 - P2 => P1 gives -1 to C10
        # C11 = P1 - P4 => P1 gives +1 to C11
        w_entries_1 = [((base_col + 2) * n + 0, 1), (base_col * n + 1, -1), ((base_col + 1) * n + 1, 1)]
        
        col = g * 5 + 1
        for (r, c) in u_entries_1:
            U[r][col] = c
        for (r, c) in v_entries_1:
            V[r][col] = c
        for (r, c) in w_entries_1:
            W[r][col] = c
        
        # Product 2: A00 * B01
        # U: A00=1
        # V: B01=1 -> row base_col+1
        u_entries_2 = [(0, 1)]
        v_entries_2 = [(base_col + 1, 1)]
        # C00 = P0 - P2 - P3 => P2 gives -1 to C00
        # C01 = P2 + P3 => P2 gives +1 to C01
        # C10 = P0 - P1 - P2 => P2 gives -1 to C10
        w_entries_2 = [(base_col * n + 0, -1), ((base_col + 1) * n + 0, 1), (base_col * n + 1, -1)]
        
        col = g * 5 + 2
        for (r, c) in u_entries_2:
            U[r][col] = c
        for (r, c) in v_entries_2:
            V[r][col] = c
        for (r, c) in w_entries_2:
            W[r][col] = c
        
        # Product 3: A00 * B02
        # U: A00=1
        # V: B02=1 -> row base_col+2
        u_entries_3 = [(0, 1)]
        v_entries_3 = [(base_col + 2, 1)]
        # C00 = P0 - P2 - P3 => P3 gives -1 to C00
        # C01 = P2 + P3 => P3 gives +1 to C01
        # C02 = P1 - P3 => P3 gives -1 to C02
        w_entries_3 = [(base_col * n + 0, -1), ((base_col + 1) * n + 0, 1), ((base_col + 2) * n + 0, -1)]
        
        col = g * 5 + 3
        for (r, c) in u_entries_3:
            U[r][col] = c
        for (r, c) in v_entries_3:
            V[r][col] = c
        for (r, c) in w_entries_3:
            W[r][col] = c
        
        # Product 4: A01 * B02
        # U: A01=1
        # V: B02=1 -> row base_col+2
        u_entries_4 = [(1, 1)]
        v_entries_4 = [(base_col + 2, 1)]
        # C11 = P1 - P4 => P4 gives -1 to C11
        # C12 = P4 => P4 gives +1 to C12
        w_entries_4 = [((base_col + 1) * n + 1, -1), ((base_col + 2) * n + 1, 1)]
        
        col = g * 5 + 4
        for (r, c) in u_entries_4:
            U[r][col] = c
        for (r, c) in v_entries_4:
            V[r][col] = c
        for (r, c) in w_entries_4:
            W[r][col] = c
        
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
