import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank-73 decomposition.
    # Part 1: Rank-49 Strassen decomposition for the 4x4x4 sub-problem (first 4 cols of B).
    # Part 2: Rank-24 Schoolbook decomposition for the 4x4x2 sub-problem (last 2 cols of B).
    # Total rank = 49 + 24 = 73.
    
    rank_total = 73
    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)
    
    col_idx = 0
    
    # --- Part 1: 4x4x4 Strassen (Rank 49) ---
    # We map the first 4 columns of B (indices 0-3 in k) to a 4x4x4 problem.
    # The 4x4x4 Strassen decomposition is recursive.
    # Base case 2x2x2 has rank 7.
    # 4x4x4 has 4 block multiplications of 2x2x2, each replaced by 7, total 49.
    
    def add_strassen_2x2x2(base_i, base_j, base_k, scale=1):
        """
        Adds the 7 rank-1 terms for a 2x2x2 block.
        Indices in the 4x4x6 grid are offset by base_i, base_j, base_k.
        scale is the coefficient (1, -1, 1/2, etc.) but we use integers if possible.
        Strassen terms for 2x2x2:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A00 + A11)B10
        M3 = A00(B01 - B11)
        M4 = A11(B10 - B00)
        M5 = (A10 - A00)(B00 + B10)
        M6 = (A01 + A10)(B11 + B01)
        M7 = (A01 - A11)(B00 + B10) -- Wait, standard Strassen:
        
        C00 = M1 + M4 - M5 - M7
        C01 = M3 + M7
        C10 = M2 + M5
        C11 = M1 - M2 + M3 + M6
        
        Let's define the 7 terms (u, v, w) for the 2x2x2 block.
        i, j, k in {0, 1}.
        """
        nonlocal col_idx
        
        # The 7 rank-1 tensors for 2x2x2 (Strassen)
        # Each term is (U_val, V_val, W_val) for the local 2x2 indices.
        # U is 4x4 (flattened), V is 4x4 (flattened), W is 4x4 (flattened).
        
        # Term 1: (A00 + A11)(B00 + B11) -> contributes to C00, C11
        # Local indices:
        # U: (0,0), (1,1) with +1
        # V: (0,0), (1,1) with +1
        # W: (0,0), (1,1) with +1? No.
        # The rank-1 tensor is u_i v_j w_k.
        # For M1, the product is (A00+A11)(B00+B11).
        # This is not a single rank-1 term in the scalar sense, but in the tensor decomposition of the multiplication algorithm, 
        # we treat the "multiplication" M1 as a scalar.
        # The rank-1 terms of the 2x2x2 multiplication tensor T[i,j,k] = sum M_t U_t[i,j] V_t[j,k] W_t[k,i] ?
        # No, the standard tensor decomposition is T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i].
        
        # For 2x2x2, the 7 terms are:
        # 1. U: [[1,0],[1,0]]? No.
        
        # Let's use the explicit 7 triples for 2x2x2 Strassen.
        # Reference: 
        # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
        
        # u1 = (1, 0, 0, 1) [A00, A01, A10, A11] -> [[1,0],[1,0]]? 
        # Let's use the list of 7 rank-1 tensors for 2x2x2.
        
        terms_2x2x2 = [
            # (u, v, w) where u,v,w are 4-vectors corresponding to 2x2 matrices flattened.
            # u: A coeffs, v: B coeffs, w: C coeffs.
            # 1
            ([[1,0],[1,0]], [[1,0],[1,0]], [[1,0],[0,1]]), # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1? 
            # Wait, M1 = (A00+A11)(B00+B11).
            # C00 gets M1. C11 gets M1.
            # So w should be [[1,0],[0,1]].
            # u = [[1,0],[1,0]] is wrong. u should be [[1,0],[0,1]]? No.
            # u represents the linear form on A.
            # A00+A11 corresponds to u = [[1,0],[0,1]].
            # B00+B11 corresponds to v = [[1,0],[0,1]].
            # So term 1: u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]].
            
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
            
            # 2. M2 = (A00+A11)B10
            # u = [[1,0],[0,1]], v = [[0,0],[1,0]], w = [[0,0],[1,0]] (C10 gets M2)
            ([[1,0],[0,1]], [[0,0],[1,0]], [[0,0],[1,0]]),
            
            # 3. M3 = A00(B01-B11)
            # u = [[1,0],[0,0]], v = [[0,1],[0,-1]], w = [[0,1],[0,0]] (C01 gets M3)
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
            
            # 4. M4 = A11(B10-B00)
            # u = [[0,0],[0,1]], v = [[0,-1],[1,0]], w = [[1,0],[0,0]] (C00 gets M4)
            ([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[0,0]]),
            
            # 5. M5 = (A10-A00)(B00+B10)
            # u = [[-1,0],[1,0]], v = [[1,0],[1,0]], w = [[0,0],[1,0]] (C10 gets M5)
            ([[-1,0],[1,0]], [[1,0],[1,0]], [[0,0],[1,0]]),
            
            # 6. M6 = (A01+A10)(B11+B01)
            # u = [[0,1],[1,0]], v = [[0,1],[0,1]], w = [[0,0],[0,1]] (C11 gets M6)
            ([[0,1],[1,0]], [[0,1],[0,1]], [[0,0],[0,1]]),
            
            # 7. M7 = (A01-A11)(B00+B10)
            # u = [[0,1],[0,-1]], v = [[1,0],[1,0]], w = [[0,1],[0,0]] (C01 gets M7)
            ([[0,1],[0,-1]], [[1,0],[1,0]], [[0,1],[0,0]]),
        ]
        
        for u_mat, v_mat, w_mat in terms_2x2x2:
            # Flatten to vectors
            u_vec = np.array(u_mat).flatten()
            v_vec = np.array(v_mat).flatten()
            w_vec = np.array(w_mat).flatten()
            
            # Map to global indices
            # i is row of A/C, j is col of A/row of B, k is col of B/C
            # In our 2x2x2 block:
            # Local i, j, k in 0..1
            # Global i = base_i + i
            # Global j = base_j + j
            # Global k = base_k + k
            
            for li in range(2):
                for lj in range(2):
                    u_val = u_vec[li*2 + lj]
                    if u_val != 0:
                        gi = base_i + li
                        gj = base_j + lj
                        # U index: i * m + j. Here m=4 for the sub-problem?
                        # In the 4x4x4 sub-problem, A is 4x4.
                        # So U index is gi * 4 + gj.
                        U[gi * 4 + gj, col_idx] = u_val
                        
            for lj in range(2):
                for lk in range(2):
                    v_val = v_vec[lj*2 + lk]
                    if v_val != 0:
                        gj = base_j + lj
                        gk = base_k + lk
                        # V index: j * p + k. Here p=4 for sub-problem.
                        V[gj * 4 + gk, col_idx] = v_val
                        
            for lk in range(2):
                for li in range(2):
                    w_val = w_vec[lk*2 + li]
                    if w_val != 0:
                        gk = base_k + lk
                        gi = base_i + li
                        # W index: k * n + i. Here n=4.
                        W[gk * 4 + gi, col_idx] = w_val
                        
            col_idx += 1

    # Recursive Strassen for 4x4x4
    # We have 4 blocks of 2x2x2.
    # Top-Left: i=0,1 j=0,1 k=0,1
    add_strassen_2x2x2(0, 0, 0)
    # Top-Right: i=0,1 j=0,1 k=2,3
    add_strassen_2x2x2(0, 0, 2)
    # Bottom-Left: i=2,3 j=0,1 k=0,1
    add_strassen_2x2x2(2, 0, 0)
    # Bottom-Right: i=2,3 j=0,1 k=2,3
    add_strassen_2x2x2(2, 0, 2)
    
    # Wait, this is just 4 * 7 = 28 terms.
    # But Strassen 4x4x4 requires 7 multiplications of 2x2 matrices, each of rank 7.
    # My `add_strassen_2x2x2` adds 7 rank-1 terms for a 2x2x2 scalar multiplication.
    # But for 4x4x4, the "scalar" is a 2x2 matrix.
    # So I need to combine the 4 2x2x2 blocks into the Strassen 7 terms.
    
    # Correct Approach for 4x4x4 Strassen:
    # The 7 terms are linear combinations of the 4 blocks.
    # This requires a different implementation where the U, V, W matrices span the 4x4 space.
    
    # Let's restart the 4x4x4 part with a proper recursive Strassen implementation.
    
    col_idx = 0
    
    def get_strassen_2x2x2_terms():
        """Returns the 7 (u,v,w) for 2x2x2."""
        return [
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
            ([[1,0],[0,1]], [[0,0],[1,0]], [[0,0],[1,0]]),
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
            ([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[0,0]]),
            ([[-1,0],[1,0]], [[1,0],[1,0]], [[0,0],[1,0]]),
            ([[0,1],[1,0]], [[0,1],[0,1]], [[0,0],[0,1]]),
            ([[0,1],[0,-1]], [[1,0],[1,0]], [[0,1],[0,0]]),
        ]

    # We will build the 49 terms for 4x4x4 by expanding the 7 Strassen terms for 4x4x4,
    # where each term is a product of two 2x2 matrices, expanded into 7 rank-1 terms of 2x2x2.
    
    # Strassen 4x4x4 terms (linear forms on 4x4 matrices):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A00+A11)B10
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01+A10)(B11+B01)
    # M7 = (A01-A11)(B00+B10)
    
    # We need to represent these as rank-1 tensors in the 4x4x4 space.
    # Each M_k is a matrix product of two linear combinations of 2x2 blocks.
    # Let U_k be the 4x4 matrix representing the linear form on A.
    # Let V_k be the 4x4 matrix representing the linear form on B.
    # Then M_k = U_k * V_k (matrix product).
    # We then decompose M_k into 7 rank-1 terms using the 2x2x2 Strassen decomposition?
    # No, M_k is a 4x4 matrix. We need to express the identity map 4x4 * 4x4 -> 4x4 using 49 rank-1 terms.
    
    # Actually, the rank-49 decomposition of 4x4x4 is exactly the result of applying Strassen twice.
    # I will hardcode the 49 terms for 4x4x4.
    
    # List of 49 (i,j,k) triples for 4x4x4.
    # I will generate them using the recursive structure.
    
    def build_strassen_4x4_4x4():
        terms = []
        
        # The 7 linear forms for A and B in the 4x4x4 Strassen step
        # Each form is a 4x4 matrix (representing the block structure)
        
        # M1: (A00+A11) and (B00+B11)
        # A form: [[1,0],[0,1]] in block space -> [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]?
        # No, A00 is top-left 2x2, A11 is bottom-right 2x2.
        # A00+A11 corresponds to the identity matrix I4?
        # A = [[A00, A01], [A10, A11]].
        # A00+A11 is the matrix [[1,0],[0,1]] in the "block scalar" sense.
        # As a 4x4 matrix, it is diag(1,1,1,1) = I4.
        
        # Let's use the explicit 49 triples from a known source or generate them.
        # Generation:
        # For each of the 7 Strassen steps, we have a pair of 4x4 matrices U4, V4.
        # The product U4 V4 is added to C.
        # We decompose the identity mapping of 4x4 matrix multiplication into 49 rank-1 terms?
        # No, we decompose the specific U4 V4 product.
        # Since U4 V4 is just a matrix product, and we are adding it to C,
        # we can just add the rank-1 terms of the 4x4x4 tensor that correspond to this specific multiplication.
        
        # This is getting too complex to generate on the fly without error.
        # I will use the schoolbook rank-96 for the whole thing?
        # No, I must improve.
        
        # I will use the rank-73 solution:
        # 1. Rank-49 for 4x4x4.
        # 2. Rank-24 for 4x4x2.
        
        # I will hardcode the 49 triples for 4x4x4.
        # (This is a standard list)
        
        # To save space and ensure correctness, I will use the recursive Strassen generator 
        # that directly outputs the (i,j,k) triples for the 4x4x4 problem.
        
        # Recursive Strassen for 4x4x4:
        # Step 1: 7 terms for 4x4x4, each term is a product of two 2x2 matrices (blocks).
        # Step 2: Each 2x2x2 product is expanded into 7 rank-1 terms of 2x2x2.
        
        # The 7 terms for 4x4x4 (block level):
        # 1. (A00+A11)(B00+B11) -> C00+=1, C11+=1
        # 2. (A00+A11)B10 -> C10+=1
        # 3. A00(B01-B11) -> C01+=1
        # 4. A11(B10-B00) -> C00+=1
        # 5. (A10-A00)(B00+B10) -> C10+=1
        # 6. (A01+A10)(B11+B01) -> C11+=1
        # 7. (A01-A11)(B00+B10) -> C01+=1
        
        # For each of these 7, we expand the 2x2x2 product.
        
        block_terms = [
            # (U_block, V_block, W_block) where block indices are 0..3
            # U_block is 2x2 matrix of coefficients for A's 4 blocks
            # V_block is 2x2 matrix of coefficients for B's 4 blocks
            # W_block is 2x2 matrix of coefficients for C's 4 blocks
            
            # 1
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
            # 2
            ([[1,0],[0,1]], [[0,0],[1,0]], [[0,0],[1,0]]),
            # 3
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
            # 4
            ([[0,0],[0,1]], [[0,-1],[1,0]], [[1,0],[0,0]]),
            # 5
            ([[-1,0],[1,0]], [[1,0],[1,0]], [[0,0],[1,0]]),
            # 6
            ([[0,1],[1,0]], [[0,1],[0,1]], [[0,0],[0,1]]),
            # 7
            ([[0,1],[0,-1]], [[1,0],[1,0]], [[0,1],[0,0]]),
        ]
        
        terms_2x2x2 = get_strassen_2x2x2_terms()
        
        for ub, vb, wb in block_terms:
            # Expand each block term into 7 scalar rank-1 terms
            for ui, vi, wi in terms_2x2x2:
                # Combine block and scalar indices
                # U_global[i,j] = sum_{bi,bj} ub[bi,bj] * ui[li,lj] where i = 2bi+li, j = 2bj+lj
                # This is a Kronecker product of the block matrix and the scalar matrix.
                
                U_mat = np.kron(ub, ui)
                V_mat = np.kron(vb, vi)
                W_mat = np.kron(wb, wi)
                
                # Add this rank-1 term to the list
                terms.append((U_mat, V_mat, W_mat))
                
        return terms

    strassen_49_terms = build_strassen_4x4_4x4()
    
    for U_mat, V_mat, W_mat in strassen_49_terms:
        U[:, col_idx] = U_mat.flatten()
        V[:, col_idx] = V_mat.flatten()
        W[:, col_idx] = W_mat.flatten()
        col_idx += 1
        
    # --- Part 2: 4x4x2 Schoolbook (Rank 24) ---
    # k ranges from 4 to 5.
    for i in range(4):
        for j in range(4):
            for k in range(2):
                k_global = k + 4
                U[i * 4 + j, col_idx] = 1
                V[j * 4 + k_global, col_idx] = 1
                W[k_global * 4 + i, col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
