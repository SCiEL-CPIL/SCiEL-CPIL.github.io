"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose M x N x P tensor into smaller blocks.
    # Dimensions: n=2, m=5, p=6
    # Split m=5 into m1=1, m2=4.
    # Part 1: n x m1 x p = 2 x 1 x 6. Rank = 2*1*6 = 12.
    # Part 2: n x m2 x p = 2 x 4 x 6.
    #   Split m2=4 into m2a=2, m2b=2.
    #   Part 2a: 2 x 2 x 6. Split p=6 into p1=2, p2=2, p3=2.
    #     Each 2x2x2 block has rank 7 (Strassen).
    #     Rank(2x2x6) = 3 * 7 = 21.
    #   Part 2b: 2 x 2 x 6. Rank = 21.
    #   Total Rank(2x4x6) = 42.
    # Total Rank = 12 + 42 = 54.
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal current_rank
        if val != 0:
            U[u_idx][current_rank] += val
            V[v_idx][current_rank] += val
            W[w_idx][current_rank] += val
        current_rank += 1

    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a Strassen decomposition for a 2x2x2 block.
        Local indices i,j,k in {0,1}.
        Global indices:
        i_global = i_base + i
        j_global = j_base + j
        k_global = k_base + k
        
        U index: i_global * m + j_global
        V index: j_global * p + k_global
        W index: k_global * n + i_global
        """
        # Strassen terms for 2x2x2
        # C = A * B
        # T1 = (A00 + A11)(B00 + B11) -> C00, C11
        # T2 = (A10 + A11)B00 -> C10, C11
        # T3 = A00(B01 - B11) -> C01, C11
        # T4 = A11(B10 - B00) -> C00, C10
        # T5 = (A00 + A01)B11 -> C00, C01
        # T6 = (A10 - A00)(B00 + B01) -> C11
        # T7 = (A01 - A11)(B10 + B11) -> C00
        
        # Helper to get global indices
        def u_idx(i, j):
            return (i_base + i) * m + (j_base + j)
        def v_idx(j, k):
            return (j_base + j) * p + (k_base + k)
        def w_idx(k, i):
            return (k_base + k) * n + (i_base + i)
            
        # T1
        add_term(u_idx(0,0), v_idx(0,0), w_idx(0,0), 1)
        add_term(u_idx(1,1), v_idx(1,1), w_idx(1,1), 1)
        # Note: The above is NOT Strassen. Strassen combines terms into single multiplications.
        # We must add the linear combinations to U, V, W for the SAME rank index.
        
        # Reset the logic for Strassen.
        # Each Strassen term is ONE rank unit.
        # We need to clear the previous "add_term" usage which incremented rank for each component.
        # I will rewrite the block to handle the 7 terms correctly.
        
    # The previous helper `add_term` was flawed for Strassen because Strassen terms
    # involve sums/differences in U and V that multiply to give sums/differences in W.
    # We need to assign all components of a Strassen term to the SAME rank column.
    
    # Let's restart the filling logic properly.
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def apply_linear_combo(mat, indices_vals, rank_idx):
        for idx, val in indices_vals:
            mat[idx][rank_idx] += val

    def add_strassen_term(u_comps, v_comps, w_comps):
        nonlocal current_rank
        apply_linear_combo(U, u_comps, current_rank)
        apply_linear_combo(V, v_comps, current_rank)
        apply_linear_combo(W, w_comps, current_rank)
        current_rank += 1

    def process_strassen_block(i_base, j_base, k_base):
        """
        Processes a 2x2x2 block using Strassen.
        """
        def u(i, j):
            return (i_base + i) * m + (j_base + j)
        def v(j, k):
            return (j_base + j) * p + (k_base + k)
        def w(k, i):
            return (k_base + k) * n + (i_base + i)
            
        # T1: (A00 + A11)(B00 + B11)
        # U: A00+1, A11+1
        # V: B00+1, B11+1
        # W: C00+1, C11+1
        add_strassen_term(
            [(u(0,0), 1), (u(1,1), 1)],
            [(v(0,0), 1), (v(1,1), 1)],
            [(w(0,0), 1), (w(1,1), 1)]
        )
        
        # T2: (A10 + A11)B00
        # U: A10+1, A11+1
        # V: B00+1
        # W: C10+1, C11-1
        add_strassen_term(
            [(u(1,0), 1), (u(1,1), 1)],
            [(v(0,0), 1)],
            [(w(0,1), 1), (w(1,1), -1)]
        )
        
        # T3: A00(B01 - B11)
        # U: A00+1
        # V: B01+1, B11-1
        # W: C01+1, C11+1
        add_strassen_term(
            [(u(0,0), 1)],
            [(v(0,1), 1), (v(1,1), -1)],
            [(w(1,0), 1), (w(1,1), 1)]
        )
        
        # T4: A11(B10 - B00)
        # U: A11+1
        # V: B10+1, B00-1
        # W: C00+1, C10+1
        add_strassen_term(
            [(u(1,1), 1)],
            [(v(1,0), 1), (v(0,0), -1)],
            [(w(0,0), 1), (w(0,1), 1)]
        )
        
        # T5: (A00 + A01)B11
        # U: A00+1, A01+1
        # V: B11+1
        # W: C00-1, C01+1
        add_strassen_term(
            [(u(0,0), 1), (u(0,1), 1)],
            [(v(1,1), 1)],
            [(w(0,0), -1), (w(1,0), 1)]
        )
        
        # T6: (A10 - A00)(B00 + B01)
        # U: A10+1, A00-1
        # V: B00+1, B01+1
        # W: C11+1
        add_strassen_term(
            [(u(1,0), 1), (u(0,0), -1)],
            [(v(0,0), 1), (v(0,1), 1)],
            [(w(1,1), 1)]
        )
        
        # T7: (A01 - A11)(B10 + B11)
        # U: A01+1, A11-1
        # V: B10+1, B11+1
        # W: C00+1
        add_strassen_term(
            [(u(0,1), 1), (u(1,1), -1)],
            [(v(1,0), 1), (v(1,1), 1)],
            [(w(0,0), 1)]
        )

    # Part 1: 2x1x6 (j=0)
    # Rank 12. Direct terms.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][current_rank] = 1
            V[v_idx][current_rank] = 1
            W[w_idx][current_rank] = 1
            current_rank += 1
            
    # Part 2: 2x4x6 (j=1..4)
    # Split j into [1,2] and [3,4]
    for block_j in range(2):
        j_base = 1 + block_j * 2
        # Split p=6 into [0,1], [2,3], [4,5]
        for block_k in range(3):
            k_base = block_k * 2
            process_strassen_block(0, j_base, k_base)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
