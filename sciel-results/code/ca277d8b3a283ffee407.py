"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114
"""

import numpy as np
import time

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses Strassen-based block decomposition with rank 114.
    """
    start_time = time.time()
    
    n, m, p = 4, 4, 9
    num_A = n * m  # 16
    num_B = m * p  # 36
    num_C = p * n  # 36
    
    terms = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """Add a rank-1 term."""
        u = [0] * num_A
        v = [0] * num_B
        w = [0] * num_C
        u[u_idx] = u_val
        v[v_idx] = v_val
        w[w_idx] = w_val
        terms.append((u, v, w))
    
    # Strassen 2x2*2x2 decomposition (rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # Indices: a=0, b=1, c=2, d=3, e=0, f=1, g=2, h=3, c11=0, c12=1, c21=2, c22=3
    
    strassen_2x2 = [
        # (A_indices, A_coeffs, B_indices, B_coeffs, C_indices, C_coeffs)
        # m1 = (a+d)(e+h) -> c11(+1), c22(+1)
        ([0,3], [1,1], [0,3], [1,1], [0,3], [1,1]),
        # m2 = (c+d)e -> c21(+1), c22(-1)
        ([2,3], [1,1], [0], [1], [2,3], [1,-1]),
        # m3 = a(f-h) -> c12(+1), c22(+1)
        ([0], [1], [1,3], [1,-1], [1,3], [1,1]),
        # m4 = d(g-e) -> c11(+1), c21(+1)
        ([3], [1], [2,0], [1,-1], [0,2], [1,1]),
        # m5 = (a+b)h -> c11(-1), c12(+1)
        ([0,1], [1,1], [3], [1], [0,1], [-1,1]),
        # m6 = (c-a)(e+f) -> c22(+1)
        ([2,0], [1,-1], [0,1], [1,1], [3], [1]),
        # m7 = (b-d)(g+h) -> c11(+1)
        ([1,3], [1,-1], [2,3], [1,1], [0], [1]),
    ]
    
    def add_strassen_2x2(a_base, b_base, c_base, scale=1):
        """Add 7 Strassen terms for a 2x2*2x2 block.
        
        a_base: (row_offset, col_offset) in A
        b_base: (row_offset, col_offset) in B
        c_base: (row_offset, col_offset) in C
        """
        ai, aj = a_base
        bi, bj = b_base
        ci, ck = c_base
        
        for a_idxs, a_cs, b_idxs, b_cs, c_idxs, c_cs in strassen_2x2:
            u = [0] * num_A
            v = [0] * num_B
            w = [0] * num_C
            
            for idx, cs in zip(a_idxs, a_cs):
                r, c = divmod(idx, 2)
                gi = ai + r
                gj = aj + c
                u[gi * m + gj] = cs * scale
            
            for idx, cs in zip(b_idxs, b_cs):
                r, c = divmod(idx, 2)
                gj = bi + r
                gk = bj + c
                v[gj * p + gk] = cs * scale
            
            for idx, cs in zip(c_idxs, c_cs):
                r, c = divmod(idx, 2)
                gi = ci + r
                gk = ck + c
                w[gk * n + gi] = cs * scale
            
            terms.append((u, v, w))
    
    # Strassen 4x4*4x4 using 2x2 blocks (rank 49)
    # A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]]
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_4x4(a_base, b_base, c_base, scale=1):
        """Add 49 Strassen terms for a 4x4*4x4 block.
        
        a_base: (row_offset, col_offset) in A (top-left corner)
        b_base: (row_offset, col_offset) in B (top-left corner)
        c_base: (row_offset, col_offset) in C (top-left corner)
        """
        ai, aj = a_base
        bi, bj = b_base
        ci, ck = c_base
        
        # Block positions (each 2x2):
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2)
        # B11: (0,0), B12: (0,2), B21: (2,0), B22: (2,2)
        # C11: (0,0), C12: (0,2), C21: (2,0), C22: (2,2)
        
        def block_pos(base_r, base_c, br, bc):
            return (base_r + br * 2, base_c + bc * 2)
        
        # M1 = (A11+A22)(B11+B22)
        # Decompose (X+Y)(Z+W) where X=A11, Y=A22, Z=B11, W=B22
        # = XZ + XW + YZ + YW
        # Each is 2x2*2x2, use Strassen (7 terms each)
        # But we can do better: use Strassen on the summed matrices
        
        # Actually, for the tensor decomposition, we need each term to be a single
        # bilinear product. Let's use the approach where we apply Strassen to
        # the 2x2 meta-matrix formed by the block sums.
        
        # For M1: A' = A11+A22, B' = B11+B22
        # A'[0,0] = A(ai,aj) + A(ai+2,aj+2)
        # A'[0,1] = A(ai,aj+1) + A(ai+2,aj+2+1) = A(ai,aj+1) + A(ai+2,aj+3)
        # A'[1,0] = A(ai+1,aj) + A(ai+3,aj+2)
        # A'[1,1] = A(ai+1,aj+1) + A(ai+3,aj+3)
        
        # B'[0,0] = B(bi,bj) + B(bi+2,bj+2)
        # etc.
        
        # For each Strassen term in the meta-matrix, we add a rank-1 term
        # where the u coefficients are sums from the A blocks and v from B blocks.
        
        def add_meta_strassen(a_blocks, b_blocks, c_blocks, c_scales):
            """
            a_blocks: list of (br, bc) - 2x2 blocks in A to sum
            b_blocks: list of (br, bc) - 2x2 blocks in B to sum
            c_blocks: list of (br, bc) - 2x2 blocks in C
            c_scales: list of scales for each C block (same length as c_blocks)
            """
            # Build the meta-matrix Strassen terms
            # For each of the 7 Strassen terms, we have:
            #   u: coefficients for A' entries (4 entries)
            #   v: coefficients for B' entries (4 entries)
            #   w: coefficients for C' entries (4 entries)
            
            for a_idxs, a_cs, b_idxs, b_cs, c_idxs, c_cs in strassen_2x2:
                u = [0] * num_A
                v = [0] * num_B
                w = [0] * num_C
                
                # For each A' entry with coefficient, add contributions from all A blocks
                for idx, meta_cs in zip(a_idxs, a_cs):
                    li, lj = divmod(idx, 2)
                    for (br, bc) in a_blocks:
                        gi = ai + br * 2 + li
                        gj = aj + bc * 2 + lj
                        u[gi * m + gj] += meta_cs * scale
                
                # For each B' entry with coefficient, add contributions from all B blocks
                for idx, meta_cs in zip(b_idxs, b_cs):
                    lj, lk = divmod(idx, 2)
                    for (br, bc) in b_blocks:
                        gj = bi + br * 2 + lj
                        gk = bj + bc * 2 + lk
                        v[gj * p + gk] += meta_cs * scale
                
                # For each C' entry with coefficient, add contributions to all C blocks
                for idx, meta_cs in zip(c_idxs, c_cs):
                    li, lk = divmod(idx, 2)
                    for (br, bc), cs in zip(c_blocks, c_scales):
                        gi = ci + br * 2 + li
                        gk = ck + bc * 2 + lk
                        w[gk * n + gi] += meta_cs * cs * scale
                
                terms.append((u, v, w))
        
        # M1 = (A11+A22)(B11+B22), contributes to C11(+1), C22(+1)
        add_meta_strassen(
            [(0,0), (1,1)],  # A11, A22
            [(0,0), (1,1)],  # B11, B22
            [(0,0), (1,1)],  # C11, C22
            [1, 1]
        )
        
        # M2 = (A21+A22)B11, contributes to C21(+1), C22(-1)
        add_meta_strassen(
            [(1,0), (1,1)],  # A21, A22
            [(0,0)],         # B11
            [(1,0), (1,1)],  # C21, C22
            [1, -1]
        )
        
        # M3 = A11(B12-B22), contributes to C12(+1), C22(+1)
        add_meta_strassen(
            [(0,0)],         # A11
            [(0,1), (1,1)],  # B12, B22
            [(0,1), (1,1)],  # C12, C22
            [1, 1]
        )
        
        # M4 = A22(B21-B11), contributes to C11(+1), C21(+1)
        add_meta_strassen(
            [(1,1)],         # A22
            [(1,0), (0,0)],  # B21, B11
            [(0,0), (1,0)],  # C11, C21
            [1, 1]
        )
        
        # M5 = (A11+A12)B22, contributes to C11(-1), C12(+1)
        add_meta_strassen(
            [(0,0), (0,1)],  # A11, A12
            [(1,1)],         # B22
            [(0,0), (0,1)],  # C11, C12
            [-1, 1]
        )
        
        # M6 = (A21-A11)(B11+B12), contributes to C22(+1)
        add_meta_strassen(
            [(1,0), (0,0)],  # A21, A11
            [(0,0), (0,1)],  # B11, B12
            [(1,1)],         # C22
            [1]
        )
        
        # M7 = (A12-A22)(B21+B22), contributes to C11(+1)
        add_meta_strassen(
            [(0,1), (1,1)],  # A12, A22
            [(1,0), (1,1)],  # B21, B22
            [(0,0)],         # C11
            [1]
        )
    
    # Part 1: A * B[0:4] (4x4 * 4x4) -> C[0:4]
    add_strassen_4x4((0, 0), (0, 0), (0, 0))
    
    # Part 2: A * B[4:8] (4x4 * 4x4) -> C[4:8]
    add_strassen_4x4((0, 0), (0, 4), (0, 4))
    
    # Part 3: A * B[8:9] (4x4 * 4x1) -> C[8:9], schoolbook 16 terms
    for i in range(4):
        for j in range(4):
            u = [0] * num_A
            v = [0] * num_B
            w = [0] * num_C
            u[i * m + j] = 1
            v[j * p + 8] = 1
            w[8 * n + i] = 1
            terms.append((u, v, w))
    
    rank = len(terms)
    
    # Build U, V, W as (dim, rank) matrices
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    
    for k, (u, v, w) in enumerate(terms):
        U[:, k] = u
        V[:, k] = v
        W[:, k] = w
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
