"""Rewritten program: rank-minimal exact tensor decomposition for 4x4x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [4, 4, 8]

# Strassen rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
# Each term is (U_pattern, V_pattern, W_pattern) over the 4 elements of each factor.
# U indices: (i, j) flattened as i*2 + j  (i=0,1 row; j=0,1 col)
# V indices: (j, k) flattened as j*2 + k
# W indices: (k, i) flattened as k*2 + i  (trace convention)
#
# Strassen formulas for C = A*B where A,B are 2x2:
#   C00 = M1 - M4 + M6 + M7
#   C01 = M3 + M4
#   C10 = M1 + M2
#   C11 = M3 - M2 + M5 + M6
# where:
#   M1 = (A00+A11)*(B00+B11)
#   M2 = (A10+A11)*B00
#   M3 = A00*(B01-B11)
#   M4 = A11*(B10-B00)
#   M5 = (A00+A01)*B11
#   M6 = (A10-A00)*(B00+B01)
#   M7 = (A01-A11)*(B10+B11)

def _strassen_222():
    """Return list of 7 terms, each (u, v, w) as length-4 vectors (0/1/-1 entries)."""
    # Index mapping:
    # A elements: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B elements: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C elements (trace convention C[k*n+i]): for 2x2, C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
    #   i.e., C index = k*2 + i, where k is column index, i is row index.
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    u = [1, 0, 0, 1]  # A00 + A11
    v = [1, 0, 0, 1]  # B00 + B11
    w = [1, 1, 1, 1]  # contributes to all C elements with coefficient 1
    terms.append((u, v, w))
    
    # M2 = (A10+A11)*B00
    u = [0, 0, 1, 1]  # A10 + A11
    v = [1, 0, 0, 0]  # B00
    w = [0, 0, 1, 1]  # contributes to C10 and C11 (k=1: C[1,0]=1, C[1,1]=3) -> indices 1,3
    terms.append((u, v, w))
    
    # M3 = A00*(B01-B11)
    u = [1, 0, 0, 0]  # A00
    v = [0, 1, 0, -1] # B01 - B11
    w = [0, 1, 1, 0]  # contributes to C01 and C11 (C[0,1]=2, C[1,1]=3) -> wait, let me recheck
    
    # Actually let me be more careful with the W convention.
    # W[k*n+i] where n=2 for the 2x2 case. So W index = k*2 + i.
    # C[i,j] = sum_k A[i,k]*B[k,j]. In trace convention, output index is (k,i)?? 
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This means W has shape (p*n, rank) and the entry W[k*n+i, r] corresponds to the
    # (k,i) component. For standard matrix multiplication C = A*B, C[i,j] = sum_k A[i,k]*B[k,j].
    # So the "k" in the output is the summation index, and "i" is the row of A.
    # The output matrix C has entries C[i,j], but the trace convention labels them as (k,i)?
    # No, that doesn't make sense for a standard matrix multiplication.
    
    # Let me re-read: "W[k*n+i,r]" with W shape (p*n, rank).
    # For our problem: n=4 (rows of A), m=4 (cols of A / rows of B), p=8 (cols of B).
    # Output C is 4x8, so C[i,k] for i in 0..3, k in 0..7.
    # W has shape (p*n, rank) = (8*4, rank) = (32, rank).
    # W[k*n+i, r] means the index is k*4 + i, where k is the column index of C (0..7) and i is the row index (0..3).
    # So W index = k*n + i = k*4 + i. This is just the row-major flattening of C^T or column-major of C.
    # Actually, it's flattening C with column index as the "outer" index: C[i,k] -> index k*4+i.
    
    # For the 2x2 sub-problem within a 4x4x4 block:
    # A is 4x4, B is 4x4, C is 4x4.
    # In the 2x2 block decomposition, each sub-block is 2x2.
    # For a given block multiplication (block of A, block of B, block of C):
    #   A_block[i_local, j_local], B_block[j_local, k_local], C_block[i_local, k_local]
    #   i_local in 0..1, j_local in 0..1, k_local in 0..1
    #   C_block index in the trace convention: k_local * 2 + i_local (for the 2x2 sub-problem)
    
    # So for Strassen on 2x2x2:
    # C[i_local, k_local] = sum_{j_local} A[i_local, j_local] * B[j_local, k_local]
    # W index = k_local * 2 + i_local
    
    # Let me redo the Strassen terms with this convention.
    
    # M1 = (A00+A11)*(B00+B11)
    #   A indices: A[0,0]=0, A[1,1]=3
    #   B indices: B[0,0]=0, B[1,1]=3
    #   C contribution: C[0,0] += 1, C[1,0] += 1, C[0,1] += 1, C[1,1] += 1
    #   W indices: C[0,0]->0*2+0=0, C[1,0]->0*2+1=1, C[0,1]->1*2+0=2, C[1,1]->1*2+1=3
    terms[0] = ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1])
    
    # M2 = (A10+A11)*B00
    #   A: A[1,0]=2, A[1,1]=3
    #   B: B[0,0]=0
    #   C: C[1,0] += 1, C[1,1] += 1  (i_local=1, k_local=0 and k_local=1)
    #   W indices: C[1,0]->0*2+1=1, C[1,1]->1*2+1=3
    terms[1] = ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1])
    
    # M3 = A00*(B01-B11)
    #   A: A[0,0]=0
    #   B: B[0,1]=1, B[1,1]=3
    #   C: C[0,1] += 1, C[1,1] += 1  (i_local=0,k_local=1; i_local=1,k_local=1)
    #   W indices: C[0,1]->1*2+0=2, C[1,1]->1*2+1=3
    terms[2] = ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1])
    
    # M4 = A11*(B10-B00)
    #   A: A[1,1]=3
    #   B: B[1,0]=2, B[0,0]=0
    #   C: C[1,0] += 1, C[0,0] -= 1  (i_local=1,k_local=0; i_local=0,k_local=0)
    #   W indices: C[1,0]->1, C[0,0]->0
    terms[3] = ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0])  # W[0]=1 for C[0,0], W[1]=1 for C[1,0]
    # Wait: C[0,0] gets -1*1 = -1 from -B00*A11, and C[1,0] gets +1*1 = +1 from B10*A11.
    # So W[0] = -1 (for C[0,0]), W[1] = 1 (for C[1,0])
    terms[3] = ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0])
    
    # M5 = (A00+A01)*B11
    #   A: A[0,0]=0, A[0,1]=1
    #   B: B[1,1]=3
    #   C: C[0,1] += 1, C[1,1] += 1  (i_local=0,k_local=1; i_local=1,k_local=1)
    #   W indices: C[0,1]->2, C[1,1]->3
    terms[4] = ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1])
    
    # M6 = (A10-A00)*(B00+B01)
    #   A: A[1,0]=2, A[0,0]=0
    #   B: B[0,0]=0, B[0,1]=1
    #   C: C[1,0] += 1, C[1,1] += 1, C[0,0] -= 1, C[0,1] -= 1
    #   W indices: C[0,0]->0: -1, C[1,0]->1: +1, C[0,1]->2: -1, C[1,1]->3: +1
    terms[5] = ([1, 0, 1, 0], [1, 1, 0, 0], [-1, 1, -1, 1])
    
    # M7 = (A01-A11)*(B10+B11)
    #   A: A[0,1]=1, A[1,1]=3
    #   B: B[1,0]=2, B[1,1]=3
    #   C: C[0,0] += 1, C[1,0] += 1, C[0,1] -= 1, C[1,1] -= 1
    #   W indices: C[0,0]->0: +1, C[1,0]->1: +1, C[0,1]->2: -1, C[1,1]->3: -1
    terms[6] = ([0, 1, 0, 1], [0, 0, 1, 1], [1, 1, -1, -1])
    
    return terms

def _verify_strassen():
    """Verify that the Strassen decomposition is correct."""
    strassen = _strassen_222()
    # Build the tensor: T[i,j,k] = 1 if this is a valid multiplication term
    # T[i*4+j*2+k... no, let's just check numerically.
    # For random A, B, check that sum_s u_s(A) * v_s(B) * w_s gives C = A*B
    
    import random
    random.seed(42)
    for trial in range(10):
        A = np.array([[random.randint(-5,5) for _ in range(2)] for _ in range(2)], dtype=float)
        B = np.array([[random.randint(-5,5) for _ in range(2)] for _ in range(2)], dtype=float)
        C_true = A @ B
        C_strassen = np.zeros((2, 2))
        for u, v, w in strassen:
            u_vec = np.array(u, dtype=float)
            v_vec = np.array(v, dtype=float)
            w_vec = np.array(w, dtype=float)
            a_val = sum(u_vec[i*2+j] * A[i,j] for i in range(2) for j in range(2))
            b_val = sum(v_vec[j*2+k] * B[j,k] for j in range(2) for k in range(2))
            for i in range(2):
                for k in range(2):
                    C_strassen[i,k] += a_val * b_val * w_vec[k*2+i]
        assert np.allclose(C_strassen, C_true), f"Strassen verification failed: {C_strassen} vs {C_true}"
    return True

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strategy: Decompose 4x4x8 into two independent 4x4x4 multiplications
    # (one for B's first 4 columns, one for B's last 4 columns).
    # Each 4x4x4 uses a rank-49 Strassen-based decomposition.
    # Total rank: 98.
    
    # First, build the rank-49 decomposition for 4x4x4.
    # Use 2x2 block Strassen: 4x4 = 2x2 blocks of 2x2.
    # The 4x4x4 tensor decomposes into 7 "block-level" Strassen terms,
    # each of which is a Kronecker-like product of block selection and 2x2 Strassen.
    
    # Block indices: block row (0,1), block col (0,1) for A; block row (0,1), block col (0,1) for B.
    # For each of the 7 Strassen formulas at the block level, we need to handle
    # the 2x2 sub-multiplications within each block.
    
    # Actually, a cleaner approach: The 4x4x4 matrix multiplication tensor can be
    # viewed as 4 "diagonal" block multiplications plus cross terms, but Strassen
    # combines them. Let me use a different approach:
    # 
    # The rank of 4x4x4 is known to be at most 49 (by Strassen's method applied to
    # 2x2 blocks). Let me construct it properly.
    #
    # Partition A (4x4) into 4 blocks of 2x2: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # Similarly for B (4x4) and C (4x4).
    #
    # The product C = A*B can be computed by treating the blocks as 2x2 matrices of 2x2 blocks,
    # and applying Strassen. This gives 7 "super-terms", each involving products of 2x2 block
    # sums/differences. Each such super-term requires 2x2x2 multiplications (rank 7 each)
    # for each pair of 2x2 blocks involved.
    #
    # But this is complex. Let me use a simpler approach: direct rank-49 construction.
    #
    # Alternative: Use the known result that R(4,4,4) <= 49 via the "tensor rank of matrix
    # multiplication" bounds. Let me just use schoolbook for 4x4x4 (rank 64) for each half,
    # giving total rank 128, same as the original. That's not an improvement.
    #
    # Let me try a different decomposition. Actually, for 4x4x8, we can think of it as
    # multiplying a 4x4 matrix by a 4x8 matrix. We can split the 4x8 into two 4x4 matrices.
    # Each 4x4x4 multiplication has schoolbook rank 64. Total: 128. Same as before.
    #
    # To do better, I need a sub-schoolbook decomposition for 4x4x4.
    # The best known upper bound for R(4,4,4) is 49 (via Strassen's 2x2 block method).
    # Let me construct this.
    
    # Construction of rank-49 decomposition for 4x4x4:
    # Apply Strassen at the 2x2 block level.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2.
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2.
    #
    # Strassen formulas (at block level):
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B01-B11)
    #   M4 = A11(B10-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B01)
    #   M7 = (A01-A11)(B10+B11)
    #
    #   C00 = M1 - M4 + M6 + M7
    #   C01 = M3 + M4
    #   C10 = M1 + M2
    #   C11 = M3 - M2 + M5 + M6
    #
    # Each M is a product of two 2x2 matrices, which has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # Now I need to construct the actual U, V, W vectors for the 4x4x4 case.
    # For each block-level Strassen term (M1..M7), and for each 2x2 Strassen term (s1..s7),
    # I create a combined term.
    
    strassen_222 = _strassen_222()  # 7 terms for 2x2x2
    
    # Block-level Strassen coefficients:
    # For each block-level term b (0..6), define:
    #   U_blocks[b]: dict from block index (0,1,2,3) to coefficient
    #   V_blocks[b]: dict from block index to coefficient
    #   W_blocks[b]: dict from (output_block, sign) to coefficient
    #
    # Block indexing:
    #   A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    #   B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    #   C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    block_strassen = [
        # (U_blocks, V_blocks)  -- each is dict: block_idx -> coeff
        ({0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}),          # M2
        ({0: 1}, {1: 1, 3: -1}),         # M3
        ({3: 1}, {2: 1, 0: -1}),         # M4
        ({0: 1, 1: 1}, {3: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),   # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),   # M7
    ]
    
    # C block contributions:
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4
    # C10 = M1 + M2
    # C11 = M3 - M2 + M5 + M6
    
    C_contribs = [
        {0: 1, 3: -1, 5: 1, 6: 1},   # C00: M1 - M4 + M6 + M7
        {2: 1, 3: 1},                 # C01: M3 + M4
        {0: 1, 1: 1},                 # C10: M1 + M2
        {2: 1, 1: -1, 4: 1, 5: 1},   # C11: M3 - M2 + M5 + M6
    ]
    
    # Now construct the 4x4x4 decomposition.
    # U has shape (4*4, 49) = (16, 49)
    # V has shape (4*4, 49) = (16, 49)  
    # W has shape (4*4, 49) = (16, 49)
    
    rank_444 = 49
    U_444 = np.zeros((16, rank_444))
    V_444 = np.zeros((16, rank_444))
    W_444 = np.zeros((16, rank_444))
    
    col = 0
    for b in range(7):  # block-level Strassen term
        u_blocks, v_blocks = block_strassen[b]
        for s in range(7):  # 2x2 Strassen term within each block multiplication
            u_local, v_local, w_local = strassen_222[s]
            
            # For each block in U_blocks:
            for ab, coeff_a in u_blocks.items():
                # ab is the A block index: 0=A00, 1=A01, 2=A10, 3=A11
                # A block ab is a 2x2 matrix at position (ar, ac) where ar=ab//2, ac=ab%2
                ar, ac = ab // 2, ab % 2
                for il in range(2):  # local row in the 2x2 block
                    for jl in range(2):  # local col in the 2x2 block
                        i = ar * 2 + il
                        j = ac * 2 + jl
                        U_444[i * 4 + j, col] += coeff_a * u_local[j * 2 + il]
            
            for bb, coeff_b in v_blocks.items():
                # bb is the B block index: 0=B00, 1=B01, 2=B10, 3=B11
                br, bc = bb // 2, bb % 2
                for jl in range(2):  # local row (= col of A)
                    for kl in range(2):  # local col
                        j = br * 2 + jl
                        k = bc * 2 + kl
                        V_444[j * 4 + k, col] += coeff_b * v_local[jl * 2 + kl]
            
            # W contributions:
            for cb, coeff_c in C_contribs[0].items():
                pass  # This is per-output-block, need to handle differently
            
            # Actually, the W part: for each output block cb that receives contribution from term b,
            # with coefficient coeff_c, we add coeff_c * w_local to the corresponding W entries.
            for cb, coeff_c in C_contribs[cb].items() if False else []:
                pass
            
            # Let me restructure: for each output block cb:
            if b in C_contribs[0]:
                cb_coeff = C_contribs[0][b]
                # C00 block: rows 0-1, cols 0-1 of C
                for il in range(2):
                    for kl in range(2):
                        i = il
                        k = kl
                        W_444[k * 4 + i, col] += cb_coeff * w_local[kl * 2 + il]
            
            if b in C_contribs[1]:
                cb_coeff = C_contribs[1][b]
                # C01 block: rows 0-1, cols 2-3 of C
                for il in range(2):
                    for kl in range(2):
                        i = il
                        k = 2 + kl
                        W_444[k * 4 + i, col] += cb_coeff * w_local[kl * 2 + il]
            
            if b in C_contribs[2]:
                cb_coeff = C_contribs[2][b]
                # C10 block: rows 2-3, cols 0-1 of C
                for il in range(2):
                    for kl in range(2):
                        i = 2 + il
                        k = kl
                        W_444[k * 4 + i, col] += cb_coeff * w_local[kl * 2 + il]
            
            if b in C_contribs[3]:
                cb_coeff = C_contribs[3][b]
                # C11 block: rows 2-3, cols 2-3 of C
                for il in range(2):
                    for kl in range(2):
                        i = 2 + il
                        k = 2 + kl
                        W_444[k * 4 + i, col] += cb_coeff * w_local[kl * 2 + il]
            
            col += 1
    
    # Verify the 4x4x4 decomposition
    def verify_444():
        import random
        random.seed(123)
        for trial in range(5):
            A = np.array([[random.randint(-3,3) for _ in range(4)] for _ in range(4)], dtype=float)
            B = np.array([[random.randint(-3,3) for _ in range(4)] for _ in range(4)], dtype=float)
            C_true = A @ B
            C_dec = np.zeros((4, 4))
            for c in range(rank_444):
                u_col = U_444[:, c]
                v_col = V_444[:, c]
                w_col = W_444[:, c]
                a_val = sum(u_col[i*4+j] * A[i,j] for i in range(4) for j in range(4))
                b_val = sum(v_col[j*4+k] * B[j,k] for j in range(4) for k in range(4))
                for i in range(4):
                    for k in range(4):
                        C_dec[i,k] += a_val * b_val * w_col[k*4+i]
            assert np.allclose(C_dec, C_true, atol=1e-9), f"4x4x4 verification failed"
    
    verify_444()
    
    # Now build the 4x4x8 decomposition by using the 4x4x4 decomposition twice:
    # Once for B[:, 0:4] -> C[:, 0:4], once for B[:, 4:8] -> C[:, 4:8]
    # These are independent, so we just concatenate the columns.
    
    rank_total = 2 * rank_444  # 98
    U_total = np.zeros((16, rank_total))
    V_total = np.zeros((16, rank_total))
    W_total = np.zeros((32, rank_total))
    
    # First half: B columns 0-3, C columns 0-3
    U_total[:, 0:49] = U_444
    V_total[:, 0:49] = V_444
    W_total[0:16, 0:49] = W_444  # C columns 0-3, so W indices 0..15
    
    # Second half: B columns 4-7, C columns 4-7
    U_total[:, 49:98] = U_444
    V_total[:, 49:98] = V_444
    # For the second half, B's columns are shifted by 4.
    # V_444 has indices j*4+k where j in 0..3, k in 0..3.
    # For the second half, k should be 4..7, so we need to remap.
    V_second = np.zeros_like(V_444)
    for j in range(4):
        for k in range(4):
            V_second[j*4 + (k+4), 49:98] = V_444[j*4 + k, 0:49]
    # Actually, V_total[:, 49:98] should have B entries at positions (j, 4+k)
    # V index = j*4 + k_full where k_full = 4+k
    for r in range(16):
        for c in range(49):
            # Find which (j, k) this corresponds to in V_444
            j = r // 4
            k = r % 4
            if 0 <= j < 4 and 0 <= k < 4:
                V_total[j*4 + (k+4), 49+c] = V_444[j*4 + k, c]
    
    # W for second half: C columns 4-7
    # W_444 has indices k*4+i where k in 0..3.
    # For second half, k should be 4..7.
    for i in range(4):
        for k in range(4):
            W_total[(k+4)*4 + i, 49:98] = W_444[k*4 + i, 0:49]
    
    # Verify the full 4x4x8 decomposition
    def verify_448():
        import random
        random.seed(456)
        for trial in range(3):
            A = np.array([[random.randint(-3,3) for _ in range(4)] for _ in range(4)], dtype=float)
            B = np.array([[random.randint(-3,3) for _ in range(8)] for _ in range(4)], dtype=float)
            C_true = A @ B
            C_dec = np.zeros((4, 8))
            for c in range(rank_total):
                u_col = U_total[:, c]
                v_col = V_total[:, c]
                w_col = W_total[:, c]
                a_val = sum(u_col[i*4+j] * A[i,j] for i in range(4) for j in range(4))
                b_val = sum(v_col[j*8+k] * B[j,k] for j in range(4) for k in range(8))
                for i in range(4):
                    for k in range(8):
                        C_dec[i,k] += a_val * b_val * w_col[k*4+i]
            assert np.allclose(C_dec, C_true, atol=1e-9), f"4x4x8 verification failed"
    
    verify_448()
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U_total]
    V_list = [[float(x) for x in row] for row in V_total]
    W_list = [[float(x) for x in row] for row in W_total]
    
    return {"U": U_list, "V": V_list, "W": W_list}
