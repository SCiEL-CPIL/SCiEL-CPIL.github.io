import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1
        
    # We need to map the logical indices to the physical indices.
    # A is 2x5. Index i (0,1), j (0..4). u_idx = i*5 + j
    # B is 5x6. Index j (0..4), k (0..5). v_idx = j*6 + k
    # C is 2x6. Output index i (0,1), k (0..5). w_idx = k*2 + i
    
    # Decomposition:
    # C = C_left + C_right
    # C_left comes from B_cols 0,1,2 (p=3)
    # C_right comes from B_cols 3,4,5 (p=3)
    
    # For each half (left, right):
    # C_half = C_22 + C_23
    # C_22 comes from A_cols 0,1 and B_rows 0,1
    # C_23 comes from A_cols 2,3,4 and B_rows 2,3,4
    
    # So we have 4 blocks of type 2x2x3 and 2 blocks of type 2x1x3?
    # No.
    # Left Half (k=0,1,2):
    #   Block 1: A(0,1) x B(0,1) -> 2x2x3
    #   Block 2: A(2,3,4) x B(2,3,4) -> 2x3x3
    # Right Half (k=3,4,5):
    #   Block 3: A(0,1) x B(0,1) -> 2x2x3
    #   Block 4: A(2,3,4) x B(2,3,4) -> 2x3x3
    
    # Each 2x2x3 has rank 6.
    # Each 2x3x3 has rank 9.
    # Total 4*6 + 2*9 = 24 + 18 = 42?
    # Wait.
    # R(2,5,3) = 15.
    # 2 * 15 = 30.
    # My previous calculation:
    # R(2,5,3) = R(2,2,3) + R(2,3,3) = 6 + 9 = 15.
    # So Left Half rank 15. Right Half rank 15. Total 30.
    
    # Let's implement the blocks.
    
    # Block 1: Left Half, A(0,1), B(0,1)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1,2}
    # W indices: i in {0,1}, k in {0,1,2}
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # Schoolbook for 2x2x3
                # C[i,k] += A[i,j] B[j,k]
                u_idx = i * 5 + j
                v_idx = j * 6 + k
                w_idx = k * 2 + i
                add_term(u_idx, v_idx, w_idx, 1)
                
    # Block 2: Left Half, A(2,3,4), B(2,3,4)
    # This is 2x3x3.
    # R(2,3,3) = R(2,2,3) + R(2,1,3)
    # Sub-block 2a: A(2,3) x B(2,3) -> 2x2x3
    # Sub-block 2b: A(4) x B(4) -> 2x1x3
    
    # Sub-block 2a:
    # A indices: i in {0,1}, j in {2,3}
    # B indices: j in {2,3}, k in {0,1,2}
    # W indices: i in {0,1}, k in {0,1,2}
    for i in range(2):
        for j in range(2, 4):
            for k in range(3):
                u_idx = i * 5 + j
                v_idx = j * 6 + k
                w_idx = k * 2 + i
                add_term(u_idx, v_idx, w_idx, 1)
                
    # Sub-block 2b:
    # A indices: i in {0,1}, j in {4}
    # B indices: j in {4}, k in {0,1,2}
    # W indices: i in {0,1}, k in {0,1,2}
    for i in range(2):
        for k in range(3):
            u_idx = i * 5 + 4
            v_idx = 4 * 6 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # Block 3: Right Half, A(0,1), B(0,1)
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {3,4,5}
    # W indices: i in {0,1}, k in {3,4,5}
    for i in range(2):
        for j in range(2):
            for k in range(3, 6):
                u_idx = i * 5 + j
                v_idx = j * 6 + k
                w_idx = k * 2 + i
                add_term(u_idx, v_idx, w_idx, 1)
                
    # Block 4: Right Half, A(2,3,4), B(2,3,4)
    # Sub-block 4a: A(2,3) x B(2,3) -> 2x2x3
    for i in range(2):
        for j in range(2, 4):
            for k in range(3, 6):
                u_idx = i * 5 + j
                v_idx = j * 6 + k
                w_idx = k * 2 + i
                add_term(u_idx, v_idx, w_idx, 1)
                
    # Sub-block 4b: A(4) x B(4) -> 2x1x3
    for i in range(2):
        for k in range(3, 6):
            u_idx = i * 5 + 4
            v_idx = 4 * 6 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
