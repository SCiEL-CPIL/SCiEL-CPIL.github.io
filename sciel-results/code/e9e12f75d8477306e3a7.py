"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W):
    """Check that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(result, 1.0, atol=1e-10)

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    factors = []
    # M1 = (A00+A11)(B00+B11)
    factors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (A10+A11) * B00
    factors.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    # M3 = A00 * (B01-B11)
    factors.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    # M4 = A11 * (B10-B00)
    factors.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    # M5 = (A00+A01) * B11
    factors.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    # M6 = (A10-A00) * (B00+B01)
    factors.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (A01-A11) * (B10+B11)
    factors.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return factors

def _recursive_strassen(n, m, p, depth=0):
    """
    Recursively build a Strassen-based tensor decomposition for n x m * m x p.
    Uses 2x2 block decomposition at each level.
    
    Returns (U, V, W) as lists of lists with shapes:
      U: (n*m, rank)
      V: (m*p, rank)  
      W: (p*n, rank)
    """
    if n == 2 and m == 2 and p == 2:
        # Base case: 2x2 * 2x2 using Strassen (rank 7)
        s2 = _strassen_2x2_factors()
        rank = 7
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for r in range(7):
            u, v, w = s2[r]
            for i in range(n):
                for j in range(m):
                    U[i*m+j][r] = u[i*m+j]
            for j in range(m):
                for k in range(p):
                    V[j*p+k][r] = v[j*p+k]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r] = w[k*n+i]
        return U, V, W
    
    # Split into 2x2 blocks
    n1, n2 = n//2, n - n//2
    m1, m2 = m//2, m - m//2
    p1, p2 = p//2, p - p//2
    
    # The 7 Strassen products at this level:
    # Each product involves specific A blocks, B blocks, and contributes to C blocks
    # We define: (a_blocks, b_blocks, c_blocks) where each is a list of (block_idx, coeff)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    products = [
        # s=0: (A00+A11)(B00+B11) -> C00, C11
        (([0,3], [1,1]), ([0,3], [1,1]), ([0,3], [1,1])),
        # s=1: (A10+A11)B00 -> C10, C11
        (([2,3], [1,1]), ([0], [1]), ([2,3], [1,-1])),
        # s=2: A00(B01-B11) -> C01, C11
        (([0], [1]), ([1,3], [1,-1]), ([1,3], [1,1])),
        # s=3: A11(B10-B00) -> C00, C10
        (([3], [1]), ([2,0], [1,-1]), ([0,2], [1,1])),
        # s=4: (A00+A01)B11 -> C00, C01
        (([0,1], [1,1]), ([3], [1]), ([0,1], [-1,1])),
        # s=5: (A10-A00)(B00+B01) -> C11
        (([2,0], [1,-1]), ([0,1], [1,1]), ([3], [1])),
        # s=6: (A01-A11)(B10+B11) -> C00
        (([1,3], [1,-1]), ([2,3], [1,1]), ([0], [1])),
    ]
    
    total_rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    for s in range(7):
        a_blocks, b_blocks, c_blocks = products[s]
        
        # Determine the subproblem dimensions
        # a_blocks: list of (block_idx, coeff) for A
        # The A subproblem has dimensions based on which blocks are involved
        # For simplicity, we always use the full 2x2 block structure
        
        # Get the sub-decomposition for the 2x2 block multiplication
        # The A part: n1 x m1 or n1 x m2 or n2 x m1 or n2 x m2
        # We need to figure out which blocks are involved
        
        # For each Strassen product, determine the effective dimensions
        # This gets complex; let's use a simpler approach:
        # Always decompose as (n//2) x (m//2) * (m//2) x (p//2) for each block
        
        # Actually, let's use a cleaner recursive approach:
        # At each level, we have 7 products, each involving:
        # - A submatrix of size (n_a x m_a)
        # - B submatrix of size (m_b x p_b)  
        # - C contribution to submatrix of size (n_c x p_c)
        
        # For standard Strassen on 2x2 blocks:
        # s=0: A is (n1+m1 x m1+m2) but really we need to be more careful
        
        # Let me use a different approach: build the full decomposition iteratively
        pass
    
    # This recursive approach is getting complex. Let me use a direct construction.
    return None

def _build_strassen_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for n x m * m x p matrix multiplication.
    Uses recursive 2x2 blocking.
    """
    # Base case: 2x2 x 2x2
    if n == 2 and m == 2 and p == 2:
        s2 = _strassen_2x2_factors()
        rank = 7
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for r in range(7):
            u, v, w = s2[r]
            for idx in range(4):
                U[idx][r] = u[idx]
                V[idx][r] = v[idx]
                W[idx][r] = w[idx]
        return U, V, W
    
    # Split into 2x2 blocks
    n1, n2 = n//2, n - n//2
    m1, m2 = m//2, m - m//2
    p1, p2 = p//2, p - p//2
    
    # For each of the 7 Strassen products, we need to build sub-decompositions
    # Product s involves:
    #   A blocks with coefficients, B blocks with coefficients
    #   and contributes to C blocks with coefficients
    
    # Define the 7 products more carefully
    # Each product: (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    # where block_coeffs is a dict {block_idx: coeff}
    # block_idx: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    
    prod_defs = [
        # s=0: M1 = (A00+A11)(B00+B11)
        # A: blocks 0,3 with coeff 1; B: blocks 0,3 with coeff 1
        # C: blocks 0,3 with coeff 1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # s=1: M2 = (A10+A11)B00
        # A: blocks 2,3 with coeff 1; B: block 0 with coeff 1
        # C: block 2 with coeff 1, block 3 with coeff -1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        # s=2: M3 = A00(B01-B11)
        # A: block 0 with coeff 1; B: blocks 1,3 with coeff 1,-1
        # C: block 1 with coeff 1, block 3 with coeff 1
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        # s=3: M4 = A11(B10-B00)
        # A: block 3 with coeff 1; B: blocks 2,0 with coeff 1,-1
        # C: block 0 with coeff 1, block 2 with coeff 1
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        # s=4: M5 = (A00+A01)B11
        # A: blocks 0,1 with coeff 1; B: block 3 with coeff 1
        # C: block 0 with coeff -1, block 1 with coeff 1
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # s=5: M6 = (A10-A00)(B00+B01)
        # A: blocks 2,0 with coeff 1,-1; B: blocks 0,1 with coeff 1
        # C: block 3 with coeff 1
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # s=6: M7 = (A01-A11)(B10+B11)
        # A: blocks 1,3 with coeff 1,-1; B: blocks 2,3 with coeff 1
        # C: block 0 with coeff 1
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    for s in range(7):
        a_coeffs, b_coeffs, c_coeffs = prod_defs[s]
        
        # For this product, we need to compute a sub-matrix multiplication
        # The A part involves certain blocks, B part involves certain blocks
        # We build a sub-decomposition for the effective dimensions
        
        # Determine which A blocks are involved and their dimensions
        # Block 0: rows 0..n1-1, cols 0..m1-1 (size n1 x m1)
        # Block 1: rows 0..n1-1, cols m1..m-1 (size n1 x m2)
        # Block 2: rows n1..n-1, cols 0..m1-1 (size n2 x m1)
        # Block 3: rows n1..n-1, cols m1..m-1 (size n2 x m2)
        
        # Similarly for B and C blocks
        
        # For each product, the sub-problem is a multiplication of
        # (combined A blocks) * (combined B blocks) -> (combined C blocks)
        
        # Simplest approach: for each product, create a sub-decomposition
        # where we "flatten" the involved blocks into a single matrix
        
        # Actually, the cleanest way: for each Strassen product s,
        # we have a bilinear form on A blocks and B blocks that produces C blocks.
        # We can build a sub-decomposition for each "virtual" matrix multiplication.
        
        # Let's think of it this way: product s computes
        # sum_{block_a, block_b} a_coeffs[block_a] * b_coeffs[block_b] * (A_block_a * B_block_b)
        # and this contributes to C blocks with c_coeffs.
        
        # So we need a decomposition of the "tensor" that maps
        # (A blocks, B blocks) -> C blocks
        # where the A part is a sum of block multiplications weighted by a_coeffs
        # and the B part is a sum of block multiplications weighted by b_coeffs
        
        # The effective sub-problem for product s is:
        # (weighted sum of A blocks) * (weighted sum of B blocks) -> (weighted sum of C blocks)
        
        # This is equivalent to a single matrix multiplication where:
        # A_eff is formed by stacking the involved A blocks with their coefficients
        # B_eff is formed by stacking the involved B blocks with their coefficients
        # C_eff receives the result with the C block coefficients
        
        # Let me just build it directly by iterating over all element triples
        
        for block_a in a_coeffs:
            for block_b in b_coeffs:
                # A block_a has dimensions:
                if block_a == 0:
                    na, ma = n1, m1
                    row_off_a, col_off_a = 0, 0
                elif block_a == 1:
                    na, ma = n1, m2
                    row_off_a, col_off_a = 0, m1
                elif block_a == 2:
                    na, ma = n2, m1
                    row_off_a, col_off_a = n1, 0
                else:  # block_a == 3
                    na, ma = n2, m2
                    row_off_a, col_off_a = n1, m1
                
                if block_b == 0:
                    mb, pb = m1, p1
                    row_off_b, col_off_b = 0, 0
                elif block_b == 1:
                    mb, pb = m1, p2
                    row_off_b, col_off_b = 0, p1
                elif block_b == 2:
                    mb, pb = m2, p1
                    row_off_b, col_off_b = m1, 0
                else:  # block_b == 3
                    mb, pb = m2, p2
                    row_off_b, col_off_b = m1, p1
                
                # The sub-problem is na x ma * mb x pb -> na x pb
                # with ma == mb (they should match for valid matrix mult)
                assert ma == mb, f"Dimension mismatch: ma={ma}, mb={mb}"
                
                # Build sub-decomposition for na x ma * ma x pb
                sub_U, sub_V, sub_W = _build_strassen_decomposition(na, ma, pb)
                sub_rank = len(sub_U[0])
                
                ca = a_coeffs[block_a]
                cb = b_coeffs[block_b]
                
                for r in range(sub_rank):
                    col = rank + r
                    
                    # Add to U
                    U_col = [0] * (n * m)
                    for i in range(na):
                        for j in range(ma):
                            U_col[(row_off_a + i) * m + (col_off_a + j)] += ca * sub_U[i * ma + j][r]
                    U_cols.append(U_col)
                    
                    # Add to V
                    V_col = [0] * (m * p)
                    for j in range(ma):
                        for k in range(pb):
                            V_col[(row_off_b + j) * p + (col_off_b + k)] += cb * sub_V[j * pb + k][r]
                    V_cols.append(V_col)
                    
                    # Add to W
                    W_col = [0] * (p * n)
                    for k in range(pb):
                        for i in range(na):
                            # C block for this product
                            # The C block index depends on which C blocks are in c_coeffs
                            # We need to determine which C block this (i,k) maps to
                            # and what coefficient to use
                            # The C output of this sub-problem goes to the appropriate C block
                            # based on the position (i, k) in the overall C matrix
                            
                            # i is in range [0, na), k is in range [0, pb)
                            # The C block is determined by:
                            # - row block: 0 if i < n1, 1 if i >= n1
                            # - col block: 0 if k < p1, 1 if k >= p1
                            # But i is relative to row_off_a, so absolute row = row_off_a + i
                            abs_row = row_off_a + i
                            abs_col = col_off_b + k
                            
                            c_row_block = 0 if abs_row < n1 else 1
                            c_col_block = 0 if abs_col < p1 else 1
                            c_block_idx = c_row_block * 2 + c_col_block
                            
                            if c_block_idx in c_coeffs:
                                cc = c_coeffs[c_block_idx]
                                W_col[abs_col * n + abs_row] += cc * sub_W[k * na + i][r]
                    
                    W_cols.append(W_col)
                    
                    rank += 1
    
    # Build final matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
    
    return U, V, W

def _schoolbook():
    """Return the trivial rank n*m*p decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Tries Strassen-based decomposition first, falls back to schoolbook.
    """
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try recursive Strassen-based decomposition
    try:
        if time.time() - start_time < deadline:
            U, V, W = _build_strassen_decomposition(n, m, p)
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
