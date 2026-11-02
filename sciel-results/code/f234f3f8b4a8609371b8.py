import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = 2, 4, 8
    rank = 56
    
    # Initialize matrices with zeros
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # Term t: u_t = sum x_coeffs[i,j] * A[i,j]
    #          v_t = sum y_coeffs[j,k] * B[j,k]
    #          C contribution = u_t * v_t * z_coeffs[i,k]
    strassen_terms = [
        # t=0
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        # t=1
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        # t=2
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        # t=3
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        # t=4
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        # t=5
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        # t=6
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    col_idx = 0
    # Loop over the 4 blocks of B (width 2) and 2 blocks of A (width 2)
    # A is 2x4, split into 2 blocks of 2x2? No, A is 2x4.
    # We split the multiplication into 8 independent 2x2x2 problems.
    # A columns: 0,1 (block 0), 2,3 (block 1)
    # B rows: 0,1 (block 0), 2,3 (block 1)
    # But B is 4x8.
    # We process 2x2 sub-blocks of A and 2x4 sub-blocks of B?
    # Actually, the previous logic:
    # for a in range(2): # A block column: 2a, 2a+1
    #   for b in range(4): # B column block: 2b, 2b+1
    #     for t in range(7):
    #       This assumes we are multiplying A[:, 2a:2a+2] by B[:, 2b:2b+2]?
    #       No, B is 4x8. The "inner" dimension of A is 4.
    #       The "inner" dimension of B is 4.
    #       So we sum over j=0..3.
    #       The previous code iterated `a` over 0..1 (A column blocks) and `b` over 0..3 (B column blocks).
    #       Inside, it used `j_prime` 0..1 for the Strassen 2x2.
    #       So it effectively split the inner dimension j=0..3 into two pairs: (0,1) and (2,3).
    #       For each pair, it ran Strassen.
    #       Wait, if we split inner dimension into 2 blocks of size 2, we have 2 blocks of A (2x2) and 2 blocks of B (2x8)?
    #       No, B is 4x8.
    #       If we split inner dim (size 4) into two chunks of size 2:
    #       Chunk 0: A[:, 0:2] (2x2) and B[0:2, :] (2x8).
    #       Chunk 1: A[:, 2:4] (2x2) and B[2:4, :] (2x8).
    #       Multiplication is A_chunk * B_chunk.
    #       B_chunk is 2x8. We can split B_chunk into 4 blocks of 2x2?
    #       Yes, 2x8 is 4 blocks of width 2.
    #       So for each A_chunk (2 of them), we have 4 B_sub_blocks.
    #       Total 8 sub-problems of 2x2 * 2x2.
    #       Each uses 7 terms. Total 56.
    
    for a_block in range(2):  # A column block: cols 2*a_block, 2*a_block+1
        for b_col_block in range(4):  # B column block: cols 2*b_col_block, 2*b_col_block+1
            for t in range(7):
                term = strassen_terms[t]
                
                # X part: A[i', 2*a_block + j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    # A index: row i_prime, col 2*a_block + j_prime
                    u_idx = i_prime * m + (2*a_block + j_prime)
                    U[u_idx][col_idx] += coeff
                
                # Y part: B[2*a_block + j', 2*b_col_block + k']
                # Note: B rows are 0..3.
                # The inner index of the block corresponds to the row of B.
                # In the 2x2 sub-problem, the "row" of B is local index j_prime (0 or 1).
                # Global row is 2*a_block + j_prime.
                for (j_prime, k_prime), coeff in term['y'].items():
                    b_row = 2*a_block + j_prime
                    b_col = 2*b_col_block + k_prime
                    v_idx = b_row * p + b_col
                    V[v_idx][col_idx] += coeff
                
                # Z part: C[i', 2*b_col_block + k'] -> W[2*b_col_block + k', i']
                # Output C is 2x8.
                # Local col k_prime (0 or 1) corresponds to global col 2*b_col_block + k_prime.
                # Local row i_prime (0 or 1) is global row i_prime.
                for (i_prime, k_prime), coeff in term['z'].items():
                    c_col = 2*b_col_block + k_prime
                    c_row = i_prime
                    w_idx = c_col * n + c_row
                    W[w_idx][col_idx] += coeff
                
                col_idx += 1
                
                if time.time() > deadline:
                    # Safety break, though unlikely to trigger
                    pass

    return {"U": U, "V": V, "W": W}
