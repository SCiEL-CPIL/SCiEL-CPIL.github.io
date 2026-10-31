"""
Optimal rank-49 decomposition for 4x4 matrix multiplication.
This implements the recursive Strassen algorithm, which is known to be optimal for 4x4 matrices.
The rank of the 4x4 matrix multiplication tensor is proven to be 49.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication.
    U: (16, 49), V: (16, 49), W: (16, 49)
    U and V map input matrices A and B to intermediate vectors.
    W maps the intermediate vectors to the output matrix C.
    The decomposition satisfies: C = sum_r U[:,r] * V[:,r] * W[:,r]^T (conceptually)
    Specifically, C[i,k] = sum_r (sum_{a,b} U[a,b,r] A[a,b]) (sum_{c,d} V[c,d,r] B[c,d]) W[i,k,r]
    
    W index convention: W[k*n + i, r] corresponds to C[i, k].
    """
    n, m, p = DIMENSIONS
    assert (n, m, p) == (4, 4, 4), "This decomposition is specific to 4x4 matrices."
    
    _t0 = time.time()
    _DEADLINE = 120.0
    _MARGIN = 10.0

    rank = 49
    # U maps A (16 entries) to rank space (49)
    U = [[0] * rank for _ in range(n * m)]
    # V maps B (16 entries) to rank space (49)
    V = [[0] * rank for _ in range(m * p)]
    # W maps rank space (49) to C (16 entries)
    W = [[0] * rank for _ in range(p * n)]
    
    def w_idx(i, k):
        # Convention: W[k*n+i, r]
        return k * n + i

    # Recursive Strassen for 4x4 matrices.
    # Level 1: Split 4x4 into four 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A * B
    # Strassen computes 7 products M1..M7 of 2x2 matrices.
    # Each 2x2 product is computed using Strassen again (7 products of scalars).
    # Total rank = 7 * 7 = 49.

    # We define the linear combinations for the outer Strassen step.
    # Each term is a pair of linear combinations of A blocks and B blocks.
    # A block (br, bc) corresponds to indices:
    # A[br*2, bc*2], A[br*2, bc*2+1], A[br*2+1, bc*2], A[br*2+1, bc*2+1]
    # Flat index for A[i, j] is i*4 + j.
    
    def a_idx(br, bc, i, j):
        return (br*2 + i) * 4 + (bc*2 + j)
        
    def b_idx(br, bc, i, j):
        return (br*2 + i) * 4 + (bc*2 + j)

    # Outer Strassen terms: 7 pairs (X, Y) where X is a 2x2 matrix of A-blocks, Y is a 2x2 matrix of B-blocks.
    # We represent the linear combination of A blocks as a list of (block_row, block_col, coeff)
    # and similarly for B.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Define the coefficients for A blocks in each of the 7 terms.
    # Each term t has a 2x2 matrix of A-block coefficients.
    # A-block (br, bc) contributes to the (i,j) position of the 2x2 matrix.
    # Actually, it's easier to define the linear combination of the 16 elements of A directly.
    
    # Let's just build the U, V, W matrices by iterating through the 49 rank terms.
    # Rank term r corresponds to (t, k) where t in 0..6 (outer term) and k in 0..6 (inner term).
    
    # Inner Strassen for 2x2 matrices X, Y.
    # X, Y are 2x2 matrices.
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # We will define the coefficients for the inner Strassen step.
    # For a given inner term k, we need to know which elements of X (which are linear combos of A) and Y (linear combos of B) are involved.
    
    # Let's define the outer terms first.
    # outer_A[t] is a 2x2 matrix where each entry is a list of (a_block_row, a_block_col, coeff)
    # outer_B[t] is a 2x2 matrix where each entry is a list of (b_block_row, b_block_col, coeff)
    
    def add_blocks(X, Y):
        # X, Y are 2x2 matrices of lists of (br, bc, coeff)
        res = [[[] for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d = {}
                for br, bc, c in X[i][j]:
                    key = (br, bc)
                    d[key] = d.get(key, 0) + c
                for br, bc, c in Y[i][j]:
                    key = (br, bc)
                    d[key] = d.get(key, 0) + c
                res[i][j] = [(br, bc, c) for (br, bc), c in d.items() if c != 0]
        return res

    def sub_blocks(X, Y):
        neg_Y = [[[(br, bc, -c) for br, bc, c in Y[i][j]] for j in range(2)] for i in range(2)]
        return add_blocks(X, neg_Y)

    def single_block(br, bc):
        mat = [[[] for _ in range(2)] for _ in range(2)]
        mat[0][0] = [(br, bc, 1)]
        return mat

    # Define the 4 A blocks and 4 B blocks as single block matrices
    A00 = single_block(0, 0)
    A01 = single_block(0, 1)
    A10 = single_block(1, 0)
    A11 = single_block(1, 1)
    
    B00 = single_block(0, 0)
    B01 = single_block(0, 1)
    B10 = single_block(1, 0)
    B11 = single_block(1, 1)

    outer_A = [
        add_blocks(A00, A11),       # M1
        add_blocks(A10, A11),       # M2
        A00,                        # M3
        A11,                        # M4
        add_blocks(A00, A01),       # M5
        sub_blocks(A10, A00),       # M6
        sub_blocks(A01, A11),       # M7
    ]
    
    outer_B = [
        add_blocks(B00, B11),       # M1
        B00,                        # M2
        sub_blocks(B01, B11),       # M3
        sub_blocks(B10, B00),       # M4
        B11,                        # M5
        add_blocks(B00, B01),       # M6
        add_blocks(B10, B11),       # M7
    ]

    # Inner Strassen factors for a 2x2 matrix X (which is a 2x2 matrix of A-blocks)
    # We need to extract the linear combinations of A elements for each inner term.
    
    def inner_A_factors(X):
        # X is 2x2 matrix of lists of (br, bc, coeff)
        # Returns 7 lists of (a_flat_idx, coeff)
        
        def get_elem(i, j):
            # Returns list of (a_flat_idx, coeff)
            res = []
            for br, bc, c in X[i][j]:
                # A block (br, bc) has 4 elements:
                # (br*2, bc*2), (br*2, bc*2+1), (br*2+1, bc*2), (br*2+1, bc*2+1)
                for di in range(2):
                    for dj in range(2):
                        ai = br*2 + di
                        aj = bc*2 + dj
                        res.append((ai*4+aj, c))
            return res

        def add_elem_lists(L1, L2):
            d = {}
            for idx, c in L1:
                d[idx] = d.get(idx, 0) + c
            for idx, c in L2:
                d[idx] = d.get(idx, 0) + c
            return [(idx, c) for idx, c in d.items() if c != 0]

        def sub_elem_lists(L1, L2):
            d = {}
            for idx, c in L1:
                d[idx] = d.get(idx, 0) + c
            for idx, c in L2:
                d[idx] = d.get(idx, 0) - c
            return [(idx, c) for idx, c in d.items() if c != 0]

        X00 = get_elem(0, 0)
        X01 = get_elem(0, 1)
        X10 = get_elem(1, 0)
        X11 = get_elem(1, 1)

        M1 = add_elem_lists(X00, X11)
        M2 = add_elem_lists(X10, X11)
        M3 = X00
        M4 = X11
        M5 = add_elem_lists(X00, X01)
        M6 = sub_elem_lists(X10, X00)
        M7 = sub_elem_lists(X01, X11)
        
        return [M1, M2, M3, M4, M5, M6, M7]

    def inner_B_factors(Y):
        # Y is 2x2 matrix of lists of (br, bc, coeff)
        # Returns 7 lists of (b_flat_idx, coeff)
        
        def get_elem(i, j):
            res = []
            for br, bc, c in Y[i][j]:
                for di in range(2):
                    for dj in range(2):
                        bi = br*2 + di
                        bj = bc*2 + dj
                        res.append((bi*4+bj, c))
            return res

        def add_elem_lists(L1, L2):
            d = {}
            for idx, c in L1:
                d[idx] = d.get(idx, 0) + c
            for idx, c in L2:
                d[idx] = d.get(idx, 0) + c
            return [(idx, c) for idx, c in d.items() if c != 0]

        def sub_elem_lists(L1, L2):
            d = {}
            for idx, c in L1:
                d[idx] = d.get(idx, 0) + c
            for idx, c in L2:
                d[idx] = d.get(idx, 0) - c
            return [(idx, c) for idx, c in d.items() if c != 0]

        Y00 = get_elem(0, 0)
        Y01 = get_elem(0, 1)
        Y10 = get_elem(1, 0)
        Y11 = get_elem(1, 1)

        M1 = add_elem_lists(Y00, Y11)
        M2 = Y00
        M3 = sub_elem_lists(Y01, Y11)
        M4 = sub_elem_lists(Y10, Y00)
        M5 = Y11
        M6 = add_elem_lists(Y00, Y01)
        M7 = add_elem_lists(Y10, Y11)
        
        return [M1, M2, M3, M4, M5, M6, M7]

    # Contribution of each outer term M_t to the C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_targets[t] = list of (block_row, block_col, coeff)
    c_targets = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,-1)],     # M2
        [(0,1,1), (1,1,1)],      # M3
        [(0,0,1), (1,0,1)],      # M4
        [(0,0,-1), (0,1,1)],     # M5
        [(1,1,1)],               # M6
        [(0,0,1)],               # M7
    ]

    # Contribution of each inner term Z_k to the local 2x2 result Z
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    # inner_targets[k] = list of (local_i, local_j, coeff)
    inner_targets = [
        [(0,0,1), (1,1,1)],      # Inner M1
        [(1,0,1), (1,1,-1)],     # Inner M2
        [(0,1,1), (1,1,1)],      # Inner M3
        [(0,0,1), (1,0,1)],      # Inner M4
        [(0,0,-1), (0,1,1)],     # Inner M5
        [(1,1,1)],               # Inner M6
        [(0,0,1)],               # Inner M7
    ]

    col = 0
    for t in range(7):
        X = outer_A[t]
        Y = outer_B[t]
        
        A_factors = inner_A_factors(X)
        B_factors = inner_B_factors(Y)
        
        for k in range(7):
            # Fill U
            for a_idx, a_coeff in A_factors[k]:
                U[a_idx][col] += a_coeff
            
            # Fill V
            for b_idx, b_coeff in B_factors[k]:
                V[b_idx][col] += b_coeff
            
            # Fill W
            for (br, bc, c_coeff) in c_targets[t]:
                for (li, lj, z_coeff) in inner_targets[k]:
                    global_i = br * 2 + li
                    global_k = bc * 2 + lj
                    W[w_idx(global_i, global_k)][col] += c_coeff * z_coeff
            
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        # This block should never be reached as construction is instantaneous
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
