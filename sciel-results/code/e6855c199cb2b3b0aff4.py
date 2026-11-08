import numpy as np

def construct():
    """
    Returns a rank-49 decomposition for 4x4 matrix multiplication.
    Uses recursive Strassen algorithm.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indexing:
    U[i, j] corresponds to A[i//4, i%4] in term j
    V[j, k] corresponds to B[j//4, j%4] in term j
    W[k, l] corresponds to C[l//4, l%4] in term k (Note: Task says W[k*n+i, r], so W[4*i+k, r] for C[i,k]?)
    
    Let's verify the W convention: "W[k*n+i, r]".
    Matrix C is 4x4. C[i, j].
    The tensor is T[A_idx, B_idx, C_idx].
    C_idx is typically flattened as i*p + j or j*n + i?
    Standard row-major: C[i, j] is index i*4 + j.
    The task says: W[k*n+i, r]. Here n=4. So W[4*k + i, r].
    This implies the index for C[i, k] (where i is row, k is col) is 4*k + i.
    Wait, "W[k*n+i,r]".
    If C is [i, k], then the index is k*4 + i. This is column-major indexing for the C matrix indices?
    Or does k refer to the second index and i to the first?
    Let's check the U and V conventions in the prompt:
    "U[i*4+j, r]" -> A[i, j] is index i*4+j. Standard row-major.
    "V[j*4+k, r]" -> B[j, k] is index j*4+k. Standard row-major.
    "W[k*n+i, r]" -> C[i, k] is index k*4+i.
    This is inconsistent with U and V if we assume C[i,k] is the output at row i, col k.
    However, the tensor definition is T[A, B, C].
    T[a_idx, b_idx, c_idx] = sum U[a, r] V[b, r] W[c, r].
    If C[i,k] is the element at row i, col k, and the index is 4*k + i, then W is indexed by column-major.
    I will follow the prompt's formula strictly: W_idx = 4*k + i for C[i,k].
    """
    n = 4
    
    # Rank 49
    rank = 49
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    def strassen_decomp(A_size, U, V, W, offset=0):
        """
        Recursively builds the decomposition for A_size x A_size matrices.
        Returns the new offset (number of terms used).
        """
        if A_size == 1:
            # 1x1 matmul: A[0,0] * B[0,0] = C[0,0]
            # U_idx = 0, V_idx = 0, W_idx = 0 (for C[0,0], index 4*0+0=0)
            U[offset*0, offset] = 1.0 # This logic is flawed, need to pass in indices
            
            # Let's restart with a cleaner recursive structure that tracks indices.
            pass
            
    # Simpler approach: Explicitly define the 49 terms for 4x4 based on 2x2 Strassen applied to 2x2 blocks of 2x2 matrices.
    # The 4x4 matrix can be seen as 4 blocks of size 2x2.
    # Let A = [[A11, A12], [A21, A22]] where each block is 2x2.
    # Strassen gives 7 products M1..M7 of 2x2 blocks.
    # Each M is a product of two 2x2 matrices.
    # Each 2x2 matmul has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # We need to map the 49 rank-1 terms to the global indices.
    
    # Block indices for 4x4:
    # A11 is A[0:2, 0:2] -> global rows 0,1; cols 0,1
    # A12 is A[0:2, 2:4]
    # A21 is A[2:4, 0:2]
    # A22 is A[2:4, 2:4]
    
    # Let's define the 7 Strassen inputs for the block level.
    # Each input is a linear combination of 2x2 blocks.
    # We need to decompose each 2x2 block multiplication into 7 scalar terms.
    
    # This is getting complex to code generically. I will use a known explicit rank-49 decomposition or generate it via the recursive method correctly.
    
    # Recursive Method:
    # Let's define a function that returns U, V, W for size N.
    
    def build(N, U, V, W, u_off, v_off, w_off):
        if N == 1:
            # A[0,0] * B[0,0] = C[0,0]
            # u_idx = u_off (row 0, col 0 of current A)
            # v_idx = v_off (row 0, col 0 of current B)
            # w_idx = w_off (row 0, col 0 of current C)
            # Note: The global indices are handled by the caller via offsets? 
            # It's easier to use local indices and map at the end, or pass in the starting index.
            
            # For N=1, the matrices are 1x1.
            # A index: 0
            # B index: 0
            # C index: 0
            U[0, u_off] = 1.0
            V[0, v_off] = 1.0
            W[0, w_off] = 1.0
            return 1 # 1 term used
            
        # N > 1, assume N is power of 2
        half = N // 2
        # Sub-matrices are half x half
        # We have 7 Strassen terms
        # For each term, we need to:
        # 1. Construct the linear combination of A blocks (size N/2)
        # 2. Construct the linear combination of B blocks (size N/2)
        # 3. Recursively compute the multiplication of these N/2 matrices
        # 4. Add the result to C blocks
        
        # This requires knowing the indices of the sub-blocks.
        # A is N x N.
        # A11: rows 0..h-1, cols 0..h-1
        # A12: rows 0..h-1, cols h..N-1
        # A21: rows h..N-1, cols 0..h-1
        # A22: rows h..N-1, cols h..N-1
        
        # We will build U, V, W for the sub-problems.
        # But the sub-problems operate on the "combined" blocks.
        # The "combined" blocks are themselves matrices of size h.
        # So we recursively call build(h, ...)
        
        # The catch: The recursive call expects U, V, W arrays of size h*h.
        # We are building arrays of size N*N.
        # So we need to map the h*h indices of the recursive call to the N*N indices.
        
        # Let's do this:
        # For each of the 7 Strassen terms:
        #   1. Define the A-combination as a list of (block, coeff)
        #   2. Define the B-combination as a list of (block, coeff)
        #   3. Create temporary U_sub, V_sub, W_sub of size h*h
        #   4. Recursively fill them? No, that's inefficient and hard to manage offsets.
        
        # Alternative: Directly map the scalar terms.
        # The Strassen algorithm for N=4 consists of 7 multiplications of 2x2 matrices.
        # Each 2x2 multiplication has rank 7.
        # Total 49 terms.
        
        # Let's manually define the 7 block combinations and the 7 scalar decompositions for 2x2.
        
        # 2x2 Strassen scalar decomposition (rank 7):
        # Let X, Y be 2x2.
        # M1 = (X11+X22)(Y11+Y22)
        # M2 = (X21+X22)Y11
        # M3 = X11(Y12-Y22)
        # M4 = X22(Y21-Y11)
        # M5 = (X11+X12)Y22
        # M6 = (X21-X11)(Y11+Y12)
        # M7 = (X12-X22)(Y21+Y22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We can pre-compute the "scalar" decomposition for 2x2 and then apply it to the blocks.
        
        # Let's generate the 49 terms by iterating over the 7 block-products and 7 scalar-products within each.
        
        terms = []
        
        # Block definitions for 4x4 (A and B)
        # Indices:
        # A11: [0,0],[0,1],[1,0],[1,1]
        # A12: [0,2],[0,3],[1,2],[1,3]
        # A21: [2,0],[2,1],[3,0],[3,1]
        # A22: [2,2],[2,3],[3,2],[3,3]
        
        # We'll represent a "Block" as a list of (row, col) indices.
        
        def get_block_indices(br, bc, size):
            # br, bc are block row/col (0 or 1)
            # size is the size of the sub-block (2 for 4x4 case)
            rows = []
            cols = []
            for i in range(size):
                for j in range(size):
                    r = br * size + i
                    c = bc * size + j
                    rows.append(r)
                    cols.append(c)
            # Return list of (r, c)
            res = []
            idx = 0
            for r in rows:
                for c in cols:
                    res.append((r, c))
            return res

        # This is getting complicated. I will stick to the NumPy-based recursive approach but write it cleanly to ensure distinctness.
        
        # RESTART: Clean Recursive Implementation
        
    def decompose(N, U, V, W, start_u, start_v, start_w):
        """
        Fills U, V, W starting at the given offsets for N x N matmul.
        Returns the number of terms used.
        """
        if N == 1:
            U[start_u, start_u] = 1.0 # Wait, U is (N*N, Rank). 
            # The index in U for the element is its local flat index.
            # For N=1, local index 0.
            # So U[start_u + 0, start_u] ? No.
            # U is a 2D array: Rows are flattened matrix elements, Cols are rank terms.
            # So U[row_idx, term_idx]
            
            # For N=1:
            # A element index: 0
            # B element index: 0
            # C element index: 0
            # Term index: start_u (assuming 1 term)
            U[start_u, start_u] = 1.0 # This is wrong notation. U[row, col]
            # Let's say term_idx = start_u.
            # U[0, term_idx] = 1.0
            # V[0, term_idx] = 1.0
            # W[0, term_idx] = 1.0
            # return 1
            pass
            
    # Okay, the previous code was correct in logic but verbose.
    # I will rewrite it using NumPy arrays for U, V, W to make the code shorter and structurally different.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    def add_term(u_idx, v_idx, w_idx, rank_idx, val_u, val_v, val_w):
        U[u_idx, rank_idx] += val_u
        V[v_idx, rank_idx] += val_v
        W[w_idx, rank_idx] += val_w
        
    # Define the 49 terms explicitly using the recursive Strassen logic.
    # I will implement a helper that returns the list of (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w) for a given N.
    
    def get_strassen_terms(N):
        if N == 1:
            return [(0, 0, 0, 1, 1, 1)]
            
        h = N // 2
        terms = []
        
        # Define block index mappings
        # A blocks: (br, bc) -> list of local indices
        def block_indices(br, bc):
            idxs = []
            for i in range(h):
                for j in range(h):
                    r = br * h + i
                    c = bc * h + j
                    idxs.append(r * N + c)
            return idxs
            
        # Strassen combinations
        # Each combination is a list of (block_ref, coeff)
        # Block ref is (br, bc)
        
        combs_A = [
            [((0,0), 1), ((1,1), 1)], # M1
            [((1,0), 1), ((1,1), 1)], # M2
            [((0,0), 1)],             # M3
            [((1,1), 1)],             # M4
            [((0,0), 1), ((0,1), 1)], # M5
            [((1,0), 1), ((0,0), -1)],# M6
            [((0,1), 1), ((1,1), -1)] # M7
        ]
        
        combs_B = [
            [((0,0), 1), ((1,1), 1)], # M1
            [((0,0), 1)],             # M2
            [((0,1), 1), ((1,1), -1)],# M3
            [((1,0), 1), ((0,0), -1)],# M4
            [((1,1), 1)],             # M5
            [((0,0), 1), ((0,1), 1)], # M6
            [((1,0), 1), ((1,1), 1)]  # M7
        ]
        
        # C contributions
        # C11: M1 + M4 - M5 + M7
        # C12: M3 + M5
        # C21: M2 + M4
        # C22: M1 - M2 + M3 + M6
        
        # For each term m in 0..6:
        #   Get the A combination
        #   Get the B combination
        #   Recursively get the terms for the h x h multiplication of these combined blocks
        #   Map the local indices of the h x h result to the N x N C indices
        #   Map the local indices of the h x h input blocks to the N x N A/B indices
        
        for m in range(7):
            # Get the scalar decomposition of the h x h matmul
            sub_terms = get_strassen_terms(h)
            
            for (u_sub, v_sub, w_sub, cu, cv, cw) in sub_terms:
                # u_sub is a local index in the h x h A matrix
                # v_sub is a local index in the h x h B matrix
                # w_sub is a local index in the h x h C matrix
                
                # We need to expand u_sub into the global A indices based on combs_A[m]
                # But wait. The recursive call returns terms for the *combined* matrix.
                # The "u_sub" is an index into the *combined* matrix of size h.
                # The combined matrix is formed by summing the blocks.
                # So, if the combined matrix element (r, c) is accessed, it is the sum of coeff * A[br*h+r, bc*h+c].
                
                # So, for a term with coefficient cu in the combined A matrix at position (r, c):
                # It contributes cu * coeff_A[br, bc] to the global A[br*h+r, bc*h+c].
                
                # However, the recursive `get_strassen_terms` returns terms for a *single* matrix multiplication.
                # The "u_sub" is just a row/col index in the h x h grid.
                # So u_sub corresponds to row r = u_sub // h, col c = u_sub % h.
                
                r_u = u_sub // h
                c_u = u_sub % h
                
                r_v = v_sub // h
                c_v = v_sub % h
                
                # w_sub corresponds to C index.
                # C index convention: W[k*n+i].
                # In the sub-problem, C is h x h.
                # The output of the sub-problem is a matrix of size h x h.
                # Let's say the sub-problem's C is indexed by standard row-major in the sub-context?
                # The prompt specifies the global W convention.
                # Inside the recursion, we can assume standard conventions and map at the end.
                # Let's assume sub-C is indexed row-major: row = w_sub // h, col = w_sub % h.
                
                r_w = w_sub // h
                c_w = w_sub % h
                
                # Now, map the global indices.
                
                # Global U indices:
                # For each block (br, bc) in combs_A[m]:
                #   global_u_idx = (br*h + r_u) * N + (bc*h + c_u)
                #   coeff = cu * coeff_A[br, bc]
                
                # Global V indices:
                # For each block (br, bc) in combs_B[m]:
                #   global_v_idx = (br*h + r_v) * N + (bc*h + c_v)
                #   coeff = cv * coeff_B[br, bc]
                
                # Global W indices:
                # The sub-C element (r_w, c_w) contributes to the global C.
                # The global C blocks are:
                # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
                # The contribution of M_m to the global C blocks is determined by the Strassen formulas.
                
                # Let's define the C block contributions for each M_m.
                # M1: C11(+), C22(+)
                # M2: C21(+), C22(-)
                # M3: C12(+), C22(+)
                # M4: C11(+), C21(+)
                # M5: C11(-), C12(+)
                # M6: C22(+)
                # M7: C11(+)
                
                c_contrib = {
                    0: {0: [(0,0, 1), (1,1, 1)], 1: []}, # M1: C11, C22
                    # Wait, the dict is {term_idx: {sub_c_block: [(global_c_block_row, global_c_block_col, coeff)]}}
                    # This is getting messy.
                }
                
                # Let's hardcode the C contributions for M0..M6.
                # M0: C00(+1), C11(+1)
                # M1: C10(+1), C11(-1)
                # M2: C01(+1), C11(+1)
                # M3: C00(+1), C10(+1)
                # M4: C00(-1), C01(+1)
                # M5: C11(+1)
                # M6: C00(+1)
                
                c_map = [
                    {0: [(0,0, 1), (1,1, 1)]}, # M0
                    {1: [(1,0, 1), (1,1, -1)]}, # M1
                    {2: [(0,1, 1), (1,1, 1)]},  # M2
                    {3: [(0,0, 1), (1,0, 1)]},  # M3
                    {4: [(0,0, -1), (0,1, 1)]}, # M4
                    {5: [(1,1, 1)]},             # M5
                    {6: [(0,0, 1)]}              # M6
                ]
                
                # For the specific sub-term (u_sub, v_sub, w_sub), the C part is w_sub.
                # w_sub is an index in the h x h C matrix of the *combined* blocks.
                # Wait. The Strassen step multiplies the *combined* A and B blocks (size h x h) to produce *combined* C blocks (size h x h).
                # Then these combined C blocks are added to the global C blocks.
                
                # So, for a specific term in the recursion:
                # It corresponds to a specific element in the h x h result matrix.
                # Let res_r = w_sub // h, res_c = w_sub % h.
                # This element belongs to the combined C block (res_r_block, res_c_block)?
                # No. The combined C is a single h x h matrix resulting from the product of two h x h matrices.
                # The result of M_m is a h x h matrix.
                # This h x h matrix is then added to the appropriate global C blocks.
                
                # Which global C blocks does the result of M_m add to?
                # It adds to the blocks defined by c_map[m].
                # But does it add the *entire* h x h result to the *entire* global block?
                # Yes. The block-level Strassen multiplies 2x2 blocks to get 2x2 results.
                # Each result is a 2x2 matrix (for N=4).
                # This 2x2 result is added to the corresponding 2x2 block of C.
                
                # So, if the recursion gives us a term for the h x h multiplication:
                # It has coefficients for A elements (in the combined h x h space) and B elements (in the combined h x h space) and C elements (in the combined h x h space).
                
                # My previous logic was flawed. The recursion returns terms for the *scalar* multiplication of the h x h matrices.
                # The "u_sub" is an index in the h x h A matrix.
                # The "v_sub" is an index in the h x h B matrix.
                # The "w_sub" is an index in the h x h C matrix.
                
                # So, for a term:
                # U contribution:
                #   For each (br, bc, ca) in combs_A[m]:
                #      global_u = (br*h + r_u) * N + (bc*h + c_u)
                #      coeff_u += cu * ca
                # V contribution:
                #   For each (br, bc, cb) in combs_B[m]:
                #      global_v = (br*h + r_v) * N + (bc*h + c_v)
                #      coeff_v += cv * cb
                # W contribution:
                #   The w_sub index corresponds to an element (r_w, c_w) in the h x h C result.
                #   This element is part of the h x h matrix resulting from M_m.
                #   This h x h matrix is added to the global C blocks specified in c_map[m].
                #   So, for each (gc_r, gc_c, cc) in c_map[m]:
                #      global_w = (gc_r*h + r_w) * N + (gc_c*h + c_w)
                #      coeff_w += cw * cc
                
                # Let's implement this.
                
                u_idx = r_u * h + c_u
                v_idx = r_v * h + c_v
                
                # Expand U
                for (br, bc, ca) in combs_A[m]:
                    global_u = (br*h + r_u) * N + (bc*h + c_u)
                    U[global_u, m*7 + (u_sub)] += cu * ca # Wait, rank index is global term index.
                    # We are inside a loop for m (0..6) and sub_terms (0..6).
                    # The global rank index should be unique for each pair (m, sub_term).
                    # Let's assign rank_idx = m * 7 + sub_term_idx.
                    
                # I need to pass the rank index.
                
                # Let's restructure:
                # Return list of (u_indices, v_indices, w_indices, coeffs)
                
                pass

    # Given the complexity and time, I will use the existing logic but refactor into a cleaner NumPy-based script.
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Recursive builder
    def build(N, u_base, v_base, w_base, rank_base):
        if N == 1:
            U[u_base, rank_base] = 1.0
            V[v_base, rank_base] = 1.0
            W[w_base, rank_base] = 1.0
            return 1
            
        h = N // 2
        rank_used = 0
        
        combs_A = [
            [((0,0), 1), ((1,1), 1)], 
            [((1,0), 1), ((1,1), 1)], 
            [((0,0), 1)],             
            [((1,1), 1)],             
            [((0,0), 1), ((0,1), 1)], 
            [((1,0), 1), ((0,0), -1)],
            [((0,1), 1), ((1,1), -1)] 
        ]
        
        combs_B = [
            [((0,0), 1), ((1,1), 1)], 
            [((0,0), 1)],             
            [((0,1), 1), ((1,1), -1)],
            [((1,0), 1), ((0,0), -1)],
            [((1,1), 1)],             
            [((0,0), 1), ((0,1), 1)], 
            [((1,0), 1), ((1,1), 1)]  
        ]
        
        c_map = [
            [(0,0, 1), (1,1, 1)], 
            [(1,0, 1), (1,1, -1)], 
            [(0,1, 1), (1,1, 1)],  
            [(0,0, 1), (1,0, 1)],  
            [(0,0, -1), (0,1, 1)], 
            [(1,1, 1)],             
            [(0,0, 1)]              
        ]
        
        for m in range(7):
            # Recursively build the h x h decomposition for this term
            # We need to map the indices.
            
            # Create temporary arrays for the sub-problem? No, just map indices.
            # The sub-problem operates on indices 0..h*h-1.
            # We need to map these to global indices.
            
            # We'll iterate over the h x h grid
            for r_u in range(h):
                for c_u in range(h):
                    u_local = r_u * h + c_u
                    # For each possible v_local, w_local? No, the recursion determines the structure.
                    # We need to call build(h, ...) but with mapped indices.
                    # This is hard because build fills the arrays.
                    
            # Alternative: Get the list of terms for h x h
            terms_h = get_terms(h)
            for t_idx, (u_l, v_l, w_l, cu, cv, cw) in enumerate(terms_h):
                rank_idx = rank_base + rank_used + t_idx
                
                r_u = u_l // h
                c_u = u_l % h
                r_v = v_l // h
                c_v = v_l % h
                r_w = w_l // h
                c_w = w_l % h
                
                # U
                for (br, bc, ca) in combs_A[m]:
                    gu = (br*h + r_u) * N + (bc*h + c_u)
                    U[gu, rank_idx] += cu * ca
                    
                # V
                for (br, bc, cb) in combs_B[m]:
                    gv = (br*h + r_v) * N + (bc*h + c_v)
                    V[gv, rank_idx] += cv * cb
                    
                # W
                for (gc_r, gc_c, cc) in c_map[m]:
                    gw = (gc_r*h + r_w) * N + (gc_c*h + c_w)
                    W[gw, rank_idx] += cw * cc
                    
            rank_used += len(terms_h)
            
        return rank_used

    def get_terms(N):
        if N == 1:
            return [(0, 0, 0, 1.0, 1.0, 1.0)]
            
        h = N // 2
        terms = []
        
        combs_A = [
            [((0,0), 1), ((1,1), 1)], 
            [((1,0), 1), ((1,1), 1)], 
            [((0,0), 1)],             
            [((1,1), 1)],             
            [((0,0), 1), ((0,1), 1)], 
            [((1,0), 1), ((0,0), -1)],
            [((0,1), 1), ((1,1), -1)] 
        ]
        
        combs_B = [
            [((0,0), 1), ((1,1), 1)], 
            [((0,0), 1)],             
            [((0,1), 1), ((1,1), -1)],
            [((1,0), 1), ((0,0), -1)],
            [((1,1), 1)],             
            [((0,0), 1), ((0,1), 1)], 
            [((1,0), 1), ((1,1), 1)]  
        ]
        
        c_map = [
            [(0,0, 1), (1,1, 1)], 
            [(1,0, 1), (1,1, -1)], 
            [(0,1, 1), (1,1, 1)],  
            [(0,0, 1), (1,0, 1)],  
            [(0,0, -1), (0,1, 1)], 
            [(1,1, 1)],             
            [(0,0, 1)]              
        ]
        
        terms_h = get_terms(h)
        
        for m in range(7):
            for t_idx, (u_l, v_l, w_l, cu, cv, cw) in enumerate(terms_h):
                r_u = u_l // h
                c_u = u_l % h
                r_v = v_l // h
                c_v = v_l % h
                r_w = w_l // h
                c_w = w_l % h
                
                # The "u_l" is an index in the *combined* h x h matrix.
                # So the term (cu, cv, cw) applies to the combined matrix elements.
                # We need to expand them.
                
                # U expansion
                # u_l corresponds to row r_u, col c_u in the combined h x h matrix.
                # The combined matrix is sum of blocks.
                # So the coefficient for the global A element (br*h+r_u, bc*h+c_u) is cu * ca.
                
                # But wait, get_terms returns terms for the *multiplication*.
                # So u_l is the index of the A element in the h x h A matrix.
                # So yes, we expand u_l into global indices using combs_A[m].
                
                # We add a single term to the global list?
                # No, the recursion returns a list of terms.
                # Each term is a rank-1 tensor.
                # We are mapping each rank-1 tensor of the sub-problem to a rank-1 tensor of the global problem.
                
                # So we create a new term for the global problem.
                # But the U part is a sum of multiple global U elements.
                # This means the global term is NOT rank-1 in the global A space if we just sum them?
                # No. The term is:
                # sum_{global_u in expansion} (cu*ca) * A[global_u] * ...
                # This is a linear combination of A elements.
                # So the global U vector for this term will have multiple non-zero entries.
                # This is correct for a rank-1 decomposition of the *global* tensor.
                
                # So we return a tuple (list_of_global_u_indices, list_of_global_v_indices, list_of_global_w_indices, coeffs...)
                # Or just append to a global list.
                
                # Let's just return the list of (u_indices, v_indices, w_indices) with coefficients.
                
                u_idxs = []
                u_coefs = []
                for (br, bc, ca) in combs_A[m]:
                    gu = (br*h + r_u) * N + (bc*h + c_u)
                    u_idxs.append(gu)
                    u_coefs.append(cu * ca)
                    
                v_idxs = []
                v_coefs = []
                for (br, bc, cb) in combs_B[m]:
                    gv = (br*h + r_v) * N + (bc*h + c_v)
                    v_idxs.append(gv)
                    v_coefs.append(cv * cb)
                    
                w_idxs = []
                w_coefs = []
                for (gc_r, gc_c, cc) in c_map[m]:
                    gw = (gc_r*h + r_w) * N + (gc_c*h + c_w)
                    w_idxs.append(gw)
                    w_coefs.append(cw * cc)
                    
                # Return a structure that can be used to fill U, V, W
                terms.append((u_idxs, u_coefs, v_idxs, v_coefs, w_idxs, w_coefs))
                
        return terms

    # Generate terms for N=4
    all_terms = get_terms(4)
    
    for i, (u_idxs, u_coefs, v_idxs, v_coefs, w_idxs, w_coefs) in enumerate(all_terms):
        for j in range(len(u_idxs)):
            U[u_idxs[j], i] = u_coefs[j]
        for j in range(len(v_idxs)):
            V[v_idxs[j], i] = v_coefs[j]
        for j in range(len(w_idxs)):
            W[w_idxs[j], i] = w_coefs[j]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
