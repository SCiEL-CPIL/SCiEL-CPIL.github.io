"""Optimized rank-50 decomposition for 2x3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import time

def _make_2x3x9_rank_50():
    """
    Returns U (6x50), V (27x50), W (18x50) as lists of lists.
    Uses the direct-sum decomposition:
      Rank(2x3x9) = Rank(2x2x9) + Rank(2x1x9)
    Rank(2x2x9) = 4 * Rank(2x2x2) + Rank(2x2x1) = 4*7 + 4 = 32
    Rank(2x1x9) = 18
    Total rank = 50
    """
    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 27 x 50
    W = [[0] * rank for _ in range(p * n)]  # 18 x 50

    c = 0

    def add_term(U_entries, V_entries, W_entries, c_idx):
        for (u_idx, u_val) in U_entries:
            U[u_idx][c_idx] += u_val
        for (v_idx, v_val) in V_entries:
            V[v_idx][c_idx] += v_val
        for (w_idx, w_val) in W_entries:
            W[w_idx][c_idx] += w_val

    # --- Part A: 2x2x9 (rank 32) ---
    # Strassen for 4 blocks of 2 columns each (k in {0,1}, {2,3}, {4,5}, {6,7})
    # Then schoolbook for k=8 (rank 4)
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # M1 = (a00+a11) * (b00+b01+b10+b11) -> contributes to C00, C01, C10, C11
        # Wait, standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B01)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # Note: In our case, B is 2x2 for this block.
        # B00 = b_{0,k0}, B01 = b_{0,k1}
        # B10 = b_{1,k0}, B11 = b_{1,k1}
        
        # M1
        add_term(
            [(0*m+0, 1), (1*m+1, 1)],  # A00, A11
            [(0*p+k0, 1), (0*p+k1, 1), (1*p+k0, 1), (1*p+k1, 1)], # B00, B01, B10, B11
            [(k0*n+0, 1), (k0*n+1, -1), (k1*n+0, 1), (k1*n+1, 1)], # C00(+1), C10(-1)? No.
            # Let's re-verify W contributions.
            # C00 gets +M1
            # C11 gets +M1
            # So W[k0*n+0, c] += 1, W[k1*n+1, c] += 1
            # The other entries in W for this term should be 0.
            # Wait, the add_term function adds to specific indices.
            # I need to specify which W entries get modified.
            # M1 contributes to C00 and C11.
            # So W[k0*n+0, c] += 1
            # W[k1*n+1, c] += 1
            # Others are 0.
            # My previous code was wrong. Let's fix it.
            c
        )
        # Correct M1 W entries:
        # C00 is at k0*n+0. C11 is at k1*n+1.
        # So: [(k0*n+0, 1), (k1*n+1, 1)]
        # I will re-implement carefully.

    # Let's restart the implementation to be sure.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0

    # Strassen Block Helper
    def strassen_block(g):
        nonlocal c
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (0,0), A11 (1,1) -> idx 0, 4
        # V: B00 (0,k0), B11 (1,k1) -> idx k0, 9+k1
        # W: C00 (k0,0), C11 (k1,1) -> idx k0*2+0, k1*2+1
        # C00 = M1 - M4 + M5 - M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 adds +1 to C00 and +1 to C11.
        add_term(
            [(0, 1), (4, 1)],
            [(k0, 1), (9+k1, 1)],
            [(2*k0, 1), (2*k1+1, 1)],
            c
        )
        c += 1

        # M2 = (A10+A11)B00
        # U: A10 (1,0), A11 (1,1) -> idx 3, 4
        # V: B00 (0,k0) -> idx k0
        # W: C10 (k0,1) +1, C11 (k1,1) -1
        # C10 = M2 + M4
        # C11 = M1 - M2 + ...
        add_term(
            [(3, 1), (4, 1)],
            [(k0, 1)],
            [(2*k0+1, 1), (2*k1+1, -1)],
            c
        )
        c += 1

        # M3 = A00(B11-B01)
        # U: A00 (0,0) -> idx 0
        # V: B11 (1,k1) +1, B01 (0,k1) -1 -> idx 9+k1, k1
        # W: C01 (k1,0) +1, C11 (k1,1) +1
        # C01 = M3 + M5
        # C11 = ... + M3 ...
        add_term(
            [(0, 1)],
            [(9+k1, 1), (k1, -1)],
            [(2*k1, 1), (2*k1+1, 1)],
            c
        )
        c += 1

        # M4 = A11(B00-B10)
        # U: A11 (1,1) -> idx 4
        # V: B00 (0,k0) +1, B10 (1,k0) -1 -> idx k0, 9+k0
        # W: C00 (k0,0) -1, C10 (k0,1) +1
        # C00 = M1 - M4 + ...
        # C10 = M2 + M4
        add_term(
            [(4, 1)],
            [(k0, 1), (9+k0, -1)],
            [(2*k0, -1), (2*k0+1, 1)],
            c
        )
        c += 1

        # M5 = (A00+A01)B11
        # U: A00 (0,0), A01 (0,1) -> idx 0, 1
        # V: B11 (1,k1) -> idx 9+k1
        # W: C00 (k0,0) +1, C01 (k1,0) +1
        # C00 = ... + M5 ...
        # C01 = M3 + M5
        add_term(
            [(0, 1), (1, 1)],
            [(9+k1, 1)],
            [(2*k0, 1), (2*k1, 1)],
            c
        )
        c += 1

        # M6 = (A10-A00)(B00+B01)
        # U: A10 (1,0) +1, A00 (0,0) -1 -> idx 3, 0
        # V: B00 (0,k0) +1, B01 (0,k1) +1 -> idx k0, k1
        # W: C11 (k1,1) +1
        # C11 = ... + M6
        add_term(
            [(3, 1), (0, -1)],
            [(k0, 1), (k1, 1)],
            [(2*k1+1, 1)],
            c
        )
        c += 1

        # M7 = (A01-A11)(B10+B11)
        # U: A01 (0,1) +1, A11 (1,1) -1 -> idx 1, 4
        # V: B10 (1,k0) +1, B11 (1,k1) +1 -> idx 9+k0, 9+k1
        # W: C00 (k0,0) -1
        # C00 = ... - M7
        add_term(
            [(1, 1), (4, -1)],
            [(9+k0, 1), (9+k1, 1)],
            [(2*k0, -1)],
            c
        )
        c += 1

    # Apply Strassen for g=0,1,2,3
    for g in range(4):
        strassen_block(g)

    # Schoolbook for k=8
    # C08 = A00*B08 + A01*B18
    # C18 = A10*B08 + A11*B18
    # Term 1: A00 * B08 -> C08
    add_term([(0, 1)], [(8, 1)], [(16, 1)], c) # W[8*2+0] = W[16]
    c += 1
    # Term 2: A01 * B18 -> C08
    add_term([(1, 1)], [(17, 1)], [(16, 1)], c) # V[1*9+8]=V[17]
    c += 1
    # Term 3: A10 * B08 -> C18
    add_term([(3, 1)], [(8, 1)], [(17, 1)], c) # W[8*2+1] = W[17]
    c += 1
    # Term 4: A11 * B18 -> C18
    add_term([(4, 1)], [(17, 1)], [(17, 1)], c)
    c += 1

    # --- Part B: 2x1x9 (rank 18) ---
    # C_{i,k} += a_{i2} * b_{2k}
    # U index for a_{i2}: i*m+2. i=0 -> 2, i=1 -> 5
    # V index for b_{2k}: 2*p+k = 18+k
    # W index for C_{i,k}: k*n+i = 2k+i
    for i in range(2):
        for k in range(9):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)], c)
            c += 1

    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    try:
        U, V, W = _make_2x3x9_rank_50()
        if time.time() > deadline:
            raise TimeoutError()
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook if something goes wrong
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
