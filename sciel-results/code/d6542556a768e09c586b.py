import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We accumulate U, V, W as lists of lists (sparse representation initially, or dense)
    # Since rank is ~105, dense is fine.
    # But we don't know exact rank until we generate.
    # We will use a list of tuples (U_col, V_col, W_col) and then convert.
    # Or just build dense matrices with a growing rank.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w):
        """
        Add a rank-1 term to the decomposition.
        u_idx: linear index in A (n*m)
        v_idx: linear index in B (m*p)
        w_idx: linear index in C (p*n) -- Note: Task says W[k*n+i, r], so C is p*n?
        Wait, "W[k*n+i,r]" implies C is stored as p*n?
        Standard matrix multiplication: A (n x m) * B (m x p) = C (n x p).
        But the task says: "output factor follows the official trace convention W[k*n+i,r]".
        And shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        This implies C is of size p x n?
        Usually C[i, j] where i in 0..n-1, j in 0..p-1.
        If W index is k*n+i, and W is p*n, then k in 0..p-1, i in 0..n-1.
        So W[k, i] corresponds to C[i, k]? Or C[k, i]?
        "W[k*n+i,r]" -> row index of W is k*n+i.
        If W is (p*n, rank), then the first dimension is p*n.
        This usually means the tensor is $T_{i,j,k}$ with $i \in A, j \in B, k \in C$.
        If C is $n \times p$, then $k$ should be flattened.
        If W is $p \times n$, then $k$ is the column of C?
        Let's check the seed program.
        Seed: `W[k * n + i][column] = 1`.
        Loops: `for i in range(n): for j in range(m): for k in range(p):`
        U[i*m+j], V[j*p+k], W[k*n+i].
        This corresponds to:
        Sum over j: U[i*m+j, r] V[j*p+k, r] -> C[i, k]?
        If W[k*n+i, r] is the coefficient for C[i, k], then W is indexed by (k, i).
        So C[i, k] = Sum_r W[k*n+i, r] ... ?
        Usually Tensor Decomposition: $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        $T_{i,j,k} = \delta_{i,i} \delta_{j,j} \delta_{k,k}$? No.
        The multiplication tensor $T$ has $T_{i,j,k} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        Here $i \in 0..n-1, j \in 0..m-1, k \in 0..p-1$.
        The seed uses `U[i*m+j]`, `V[j*p+k]`, `W[k*n+i]`.
        This matches the standard definition where the third index of the tensor is flattened as $k \times n + i$.
        So C is effectively stored as $p \times n$ in the W matrix dimensions?
        Or just that the linear index for the output coordinate $(i, k)$ is $k \times n + i$.
        So when I generate a term for $C[i_{out}, k_{out}]$, the W index is $k_{out} \times n + i_{out}$.
        """
        U_cols.append([(u_idx, val_u)])
        V_cols.append([(v_idx, val_v)])
        W_cols.append([(w_idx, val_w)])
        
    def add_block_schoolbook(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, A_shape, B_shape, C_shape):
        """
        Adds schoolbook terms for a block product.
        A is A_shape (r_a, c_a), located at global (A_r0, A_c0).
        B is B_shape (r_b, c_b), located at global (B_r0, B_c0).
        C is C_shape (r_c, c_c), located at global (C_r0, C_c0).
        Assumes A_shape[1] == B_shape[0].
        """
        r_a, c_a = A_shape
        r_b, c_b = B_shape
        r_c, c_c = C_shape
        assert c_a == r_b
        for i_local in range(r_a):
            for k_local in range(c_b):
                for j_local in range(c_a):
                    u_idx = (A_r0 + i_local) * m + (A_c0 + j_local)
                    v_idx = (B_r0 + j_local) * p + (B_c0 + k_local)
                    w_idx = (C_r0 + i_local) * p + (C_c0 + k_local) 
                    # Wait, W index is k*n+i.
                    # C is n x p. i is row (0..n-1), k is col (0..p-1).
                    # w_idx = k * n + i.
                    # Here i_local is row in C block, k_local is col in C block.
                    # Global row i = C_r0 + i_local.
                    # Global col k = C_c0 + k_local.
                    w_idx = (C_c0 + k_local) * n + (C_r0 + i_local)
                    
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    def add_block_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0):
        """
        Adds Strassen terms for a 2x2 block product.
        """
        # Local indices 0,1
        # Global helpers
        def g_a(r, c): return (A_r0 + r) * m + (A_c0 + c)
        def g_b(r, c): return (B_r0 + r) * p + (B_c0 + c)
        def g_c(r, c): return (C_c0 + c) * n + (C_r0 + r) # W index: k*n+i
        
        # M1 = (A11+A22)(B11+B22)
        # A11: g_a(0,0), A22: g_a(1,1)
        # B11: g_b(0,0), B22: g_b(1,1)
        # Terms for U: A11(1), A22(1)
        # Terms for V: B11(1), B22(1)
        # This is a sum of 4 rank-1 terms in the product? No, Strassen combines them.
        # We need to add 7 columns.
        
        # Column 0: M1 part 1? No, M1 is one multiplication.
        # M1 = (A11+A22) * (B11+B22).
        # This is a product of two sums.
        # (A11+A22) * (B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # This requires 4 multiplications if done naively.
        # Strassen uses 7 multiplications total for the 4 outputs.
        # The "multiplication" is the rank-1 term.
        # So for M1, we need to form the vectors u = A11+A22 and v = B11+B22.
        # u has non-zeros at g_a(0,0) and g_a(1,1).
        # v has non-zeros at g_b(0,0) and g_b(1,1).
        
        # We add one column to U, V, W for each Strassen product M_k.
        # M_k contributes to C with coefficients.
        
        # M1
        u_1 = {g_a(0,0): 1, g_a(1,1): 1}
        v_1 = {g_b(0,0): 1, g_b(1,1): 1}
        # Contribution to C:
        # C11 += M1, C22 += M1
        w_1 = {g_c(0,0): 1, g_c(1,1): 1}
        add_term_dict(u_1, v_1, w_1)
        
        # M2 = (A21+A22) B11
        u_2 = {g_a(1,0): 1, g_a(1,1): 1}
        v_2 = {g_b(0,0): 1}
        w_2 = {g_c(1,0): 1, g_c(1,1): -1} # C21 += M2, C22 -= M2
        add_term_dict(u_2, v_2, w_2)
        
        # M3 = A11 (B21-B22)
        u_3 = {g_a(0,0): 1}
        v_3 = {g_b(1,0): 1, g_b(1,1): -1}
        w_3 = {g_c(0,1): 1, g_c(1,1): 1} # C12 += M3, C22 += M3
        add_term_dict(u_3, v_3, w_3)
        
        # M4 = A22 (B12-B11)
        u_4 = {g_a(1,1): 1}
        v_4 = {g_b(0,1): 1, g_b(0,0): -1}
        w_4 = {g_c(0,0): 1, g_c(1,0): 1} # C11 += M4, C21 += M4
        add_term_dict(u_4, v_4, w_4)
        
        # M5 = (A11+A12) B22
        u_5 = {g_a(0,0): 1, g_a(0,1): 1}
        v_5 = {g_b(1,1): 1}
        w_5 = {g_c(0,0): -1, g_c(0,1): 1} # C11 -= M5, C12 += M5
        add_term_dict(u_5, v_5, w_5)
        
        # M6 = (A21-A11) (B11+B12)
        u_6 = {g_a(1,0): 1, g_a(0,0): -1}
        v_6 = {g_b(0,0): 1, g_b(0,1): 1}
        w_6 = {g_c(1,1): 1} # C22 += M6
        add_term_dict(u_6, v_6, w_6)
        
        # M7 = (A12-A22) (B21+B22)
        u_7 = {g_a(0,1): 1, g_a(1,1): -1}
        v_7 = {g_b(1,0): 1, g_b(1,1): 1}
        w_7 = {g_c(0,0): 1} # C11 += M7
        add_term_dict(u_7, v_7, w_7)

    def add_term_dict(u_dict, v_dict, w_dict):
        # Convert dicts to sparse lists for storage
        U_cols.append(list(u_dict.items()))
        V_cols.append(list(v_dict.items()))
        W_cols.append(list(w_dict.items()))

    # 1. A22 B22 -> C22 (2x2, Strassen)
    add_block_strassen(0, 0, 0, 0, 0, 0)
    
    # 2. A23 B32 -> C22 (2x3, 3x2, Schoolbook)
    # A23: rows 0-1, cols 2-4. B32: rows 2-4, cols 0-1.
    add_block_schoolbook(0, 2, 2, 0, 0, 0, (2,3), (3,2), (2,2))
    
    # 3. A22 B23 -> C23 (2x2, 2x3, Schoolbook)
    # A22: rows 0-1, cols 0-1. B23: rows 0-1, cols 2-4.
    add_block_schoolbook(0, 0, 0, 2, 0, 2, (2,2), (2,3), (2,3))
    
    # 4. A32 B22 -> C32 (3x2, 2x2, Schoolbook)
    # A32: rows 2-4, cols 0-1. B22: rows 0-1, cols 0-1.
    add_block_schoolbook(2, 0, 0, 0, 2, 0, (3,2), (2,2), (3,2))
    
    # 5. A23 B33 -> C23 (2x3, 3x3, Schoolbook)
    # A23: rows 0-1, cols 2-4. B33: rows 2-4, cols 2-4.
    add_block_schoolbook(0, 2, 2, 2, 0, 2, (2,3), (3,3), (2,3))
    
    # 6. A32 B23 -> C32 (3x2, 2x3, Schoolbook)
    # A32: rows 2-4, cols 0-1. B23: rows 0-1, cols 2-4.
    add_block_schoolbook(2, 0, 0, 2, 2, 2, (3,2), (2,3), (3,3))
    
    # 7. A33 B33 -> C33 (3x3, 3x3, Hybrid 2+1)
    # Local 3x3 blocks:
    # A33 local (0,0) is global (2,2)
    # We split local 3x3 into 2x2, 2x1, 1x2, 1x1.
    # Local 2x2: rows 0-1, cols 0-1. Global: rows 2-3, cols 2-3.
    # Local 2x1: rows 0-1, col 2. Global: rows 2-3, col 4.
    # Local 1x2: row 2, cols 0-1. Global: row 4, cols 2-3.
    # Local 1x1: row 2, col 2. Global: row 4, col 4.
    
    # a. A22' B22' -> C22' (2x2, Strassen)
    # A22' global: (2,2) to (3,3).
    add_block_strassen(2, 2, 2, 2, 2, 2)
    
    # b. A21' B12' -> C22' (2x1, 1x2, Schoolbook)
    # A21' global: rows 2-3, col 4.
    # B12' global: row 4, cols 2-3.
    # C22' global: rows 2-3, cols 2-3.
    add_block_schoolbook(2, 4, 4, 2, 2, 2, (2,1), (1,2), (2,2))
    
    # c. A22' B21' -> C21' (2x2, 2x1, Schoolbook)
    # A22' global: rows 2-3, cols 2-3.
    # B21' global: rows 2-3, col 4.
    # C21' global: rows 2-3, col 4.
    add_block_schoolbook(2, 2, 2, 4, 2, 4, (2,2), (2,1), (2,1))
    
    # d. A12' B22' -> C12' (1x2, 2x2, Schoolbook)
    # A12' global: row 4, cols 2-3.
    # B22' global: rows 2-3, cols 2-3.
    # C12' global: row 4, cols 2-3.
    add_block_schoolbook(4, 2, 2, 2, 4, 2, (1,2), (2,2), (1,2))
    
    # e. A12' B21' -> C11' (1x2, 2x1, Schoolbook)
    # A12' global: row 4, cols 2-3.
    # B21' global: rows 2-3, col 4.
    # C11' global: row 4, col 4.
    add_block_schoolbook(4, 2, 2, 4, 4, 4, (1,2), (2,1), (1,1))
    
    # f. A11' B12' -> C11' (1x1, 1x2, Schoolbook)
    # A11' global: row 4, col 4.
    # B12' global: row 4, cols 2-3.
    # C11' global: row 4, col 4.
    add_block_schoolbook(4, 4, 4, 2, 4, 4, (1,1), (1,2), (1,1))
    
    # g. A11' B11' -> C11' (1x1, 1x1, Schoolbook)
    # A11' global: row 4, col 4.
    # B11' global: row 4, col 4.
    # C11' global: row 4, col 4.
    add_block_schoolbook(4, 4, 4, 4, 4, 4, (1,1), (1,1), (1,1))

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U_cols[r]:
            U[idx][r] = val
        for idx, val in V_cols[r]:
            V[idx][r] = val
        for idx, val in W_cols[r]:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
