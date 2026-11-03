"""Combined program: hybrid Strassen decomposition with verification."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    def get_schoolbook():
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

    def get_2x2_strassen_block(i0, j0, k0, U, V, W, offset):
        """
        Add a 2x2 Strassen block for sub-matrices starting at (i0, j0, k0).
        U rows: i0..i0+1, m=2 -> indices i0*2+j for j in 0..1
        V rows: j0..j0+1, p=2 -> indices j0*2+k for k in 0..1
        W rows: k0..k0+1, n=2 -> indices k0*2+i for i in 0..1
        """
        # Strassen 2x2x2: 7 multiplications
        # U factors (for A submatrix, 2x2):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A11-A01)(B01+B11)
        #
        # In tensor form: U[row, col], V[row, col], W[row, col]
        # Each Strassen term c has U, V, W vectors.
        #
        # For 2x2x2 Strassen, the 7 terms have these (U, V, W) patterns:
        # U is over A indices (i,j), V over B indices (j,k), W over C indices (k,i)
        
        # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C01 -= M4, C10 -= M2, C11 += M1
        # Actually let me use the standard Strassen formulation more carefully.
        # 
        # Standard Strassen for C = A*B where A,B,C are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A11-A01)(B01+B11)
        #
        # C00 = M1 - M4 + M5 + M7
        # C01 = M2 + M4
        # C10 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        #
        # Tensor decomposition: each term t contributes to U, V, W.
        # U[i,j] for term t: coefficient of A[i,j] in the left factor
        # V[j,k] for term t: coefficient of B[j,k] in the right factor
        # W[k,i] for term t: coefficient of C[i,k] in the output (note W index is k*n+i)
        
        # For each term, U is a 4-vector (over i,j for 2x2), V is a 4-vector (over j,k), W is a 4-vector (over k,i)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1 -> [1,0,0,1] (i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1)
        # V: B00=1, B11=1 -> [1,0,0,1]
        # W: C00 += 1, C11 += 1 -> W[k*n+i]: C00 is k=0,i=0 -> W[0]=1; C11 is k=1,i=1 -> W[3]=1
        # So W = [1, 0, 0, 1]
        
        # Term 2: M2 = (A10+A11)B00
        # U: A10=1, A11=1 -> [0,0,1,1]
        # V: B00=1 -> [1,0,0,0]
        # W: C01 += 1, C10 -= 1 -> C01 is k=1,i=0 -> W[1*n+0]=W[2]=1; C10 is k=0,i=1 -> W[0*n+1]=W[1]=-1
        # W = [0, -1, 1, 0]
        
        # Term 3: M3 = A00(B10-B11)
        # U: A00=1 -> [1,0,0,0]
        # V: B10=1, B11=-1 -> [0,0,1,-1]
        # W: C10 += 1, C11 += 1 -> C10: k=0,i=1 -> W[1]=1; C11: k=1,i=1 -> W[3]=1
        # W = [0, 1, 0, 1]
        
        # Term 4: M4 = A11(B01-B10)
        # U: A11=1 -> [0,0,0,1]
        # V: B01=1, B10=-1 -> [0,1,-1,0]
        # W: C00 -= 1, C01 += 1 -> C00: k=0,i=0 -> W[0]=-1; C01: k=1,i=0 -> W[2]=1
        # W = [-1, 0, 1, 0]
        
        # Term 5: M5 = (A00+A10)B11
        # U: A00=1, A10=1 -> [1,0,1,0]
        # V: B11=1 -> [0,0,0,1]
        # W: C00 += 1, C10 += 1 -> C00: W[0]=1; C10: W[1]=1
        # W = [1, 1, 0, 0]
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        # U: A10=1, A00=-1 -> [-1,0,1,0]
        # V: B00=1, B10=1 -> [1,0,1,0]
        # W: C11 += 1 -> C11: k=1,i=1 -> W[3]=1
        # W = [0, 0, 0, 1]
        
        # Term 7: M7 = (A11-A01)(B01+B11)
        # U: A11=1, A01=-1 -> [0,-1,0,1]
        # V: B01=1, B11=1 -> [0,1,0,1]
        # W: C00 += 1 -> C00: k=0,i=0 -> W[0]=1
        # W = [1, 0, 0, 0]
        
        terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,-1,1,0]),
            ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
            ([0,0,0,1], [0,1,-1,0], [-1,0,1,0]),
            ([1,0,1,0], [0,0,0,1], [1,1,0,0]),
            ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),
            ([0,-1,0,1], [0,1,0,1], [1,0,0,0]),
        ]
        
        for u_t, v_t, w_t in terms:
            # Map local indices to global
            # U: local (i,j) -> global (i0+i, j0+j) -> row (i0+i)*m + (j0+j) where m=2 locally... 
            # Wait, the global m is 4, not 2. Let me reconsider.
            # 
            # Actually, the block decomposition works on sub-matrices.
            # The 2x2 block at (i0, j0, k0) means:
            # A submatrix: rows i0..i0+1, cols j0..j0+1 of A (which is 4x4)
            # B submatrix: rows j0..j0+1, cols k0..k0+1 of B (which is 4x6)
            # C submatrix: rows k0..k0+1, cols i0..i0+1 of C (which is 6x4)
            #
            # U global row index: (i0+i)*4 + (j0+j) for i,j in 0..1
            # V global row index: (j0+j)*6 + (k0+k) for j,k in 0..1
            # W global row index: (k0+k)*4 + (i0+i) for k,i in 0..1
            
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    if u_t[local_idx] != 0:
                        global_row = (i0 + li) * m + (j0 + lj)
                        U[global_row][offset] += u_t[local_idx]
            
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    if v_t[local_idx] != 0:
                        global_row = (j0 + lj) * p + (k0 + lk)
                        V[global_row][offset] += v_t[local_idx]
            
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    if w_t[local_idx] != 0:
                        global_row = (k0 + lk) * n + (i0 + li)
                        W[global_row][offset] += w_t[local_idx]
        
        return offset + 7

    def get_4x4_strassen_block(i0, j0, k0, U, V, W, offset):
        """
        Add a 4x4 Strassen block. This uses the rank-49 decomposition for 4x4x4.
        The 4x4x4 Strassen decomposition can be built from 2x2x2 Strassen blocks
        recursively, but the known optimal is rank 49.
        
        For simplicity and correctness, I'll use the tensor product of 2x2 Strassen
        which gives rank 7^2 = 49 for 4x4x4.
        
        The 4x4 matrix A is partitioned into four 2x2 blocks:
        A = [[A00, A01], [A10, A11]]
        Similarly for B and C.
        
        The 4x4x4 Strassen via tensor product of 2x2x2 Strassen:
        Each of the 7 terms of the outer Strassen combines with each of the 7 terms
        of the inner Strassen, giving 49 terms total.
        
        Actually, the standard way: C = A*B where A,B,C are 4x4.
        Partition into 2x2 blocks. Then use Strassen on the 2x2 block matrix.
        Each block multiplication (2x2 * 2x2) uses 7 scalar multiplications.
        So total = 7 (block-level) * 7 (within-block) = 49.
        
        Let me implement this by composing two levels of Strassen.
        """
        # Level 1: Strassen on 2x2 block matrix
        # Level 2: Strassen on each 2x2 block multiplication
        
        # The 7 block-level terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A11-A01)(B01+B11)
        
        # Each term involves products of sums/differences of 2x2 blocks.
        # Each block product (2x2 * 2x2) uses 7 scalar multiplications.
        
        # For term M1: (A00+A11)(B00+B11)
        # This is a product of two 2x2 matrices: X = A00+A11, Y = B00+B11
        # X is 2x2, Y is 2x2, result is 2x2.
        # Use Strassen for X*Y: 7 terms.
        
        # I'll build this by iterating over the 7 block-level terms and for each,
        # decomposing the 2x2*2x2 product using Strassen.
        
        # Block-level Strassen terms (each is a product of two linear combinations of 2x2 blocks):
        # For each term, we have:
        #   Left: sum of coefficients * A_blocks
        #   Right: sum of coefficients * B_blocks
        #   Output: coefficient * C_block
        
        # Let me define the 7 block-level terms more explicitly.
        # A blocks: A00, A01, A10, A11 (each 2x2)
        # B blocks: B00, B01, B10, B11 (each 2x2)
        # C blocks: C00, C01, C10, C11 (each 2x2)
        
        # M1 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        # M2 = (A10+A11)B00 -> contributes to C01, C10
        # M3 = A00(B10-B11) -> contributes to C10, C11
        # M4 = A11(B01-B10) -> contributes to C00, C01
        # M5 = (A00+A10)B11 -> contributes to C00, C10
        # M6 = (A10-A00)(B00+B10) -> contributes to C11
        # M7 = (A11-A01)(B01+B11) -> contributes to C00
        
        # For each block-level term, the left factor is a linear combination of A blocks,
        # and the right factor is a linear combination of B blocks.
        # The product of two 2x2 matrices is done via Strassen (7 terms).
        
        # Let me define the left and right factors for each block-level term:
        # Each factor is a 2x2 matrix that is a linear combination of the 2x2 A or B blocks.
        
        # For M1: Left = A00 + A11, Right = B00 + B11
        #   Left[i,j] = A00[i,j] + A11[i,j]
        #   Right[i,j] = B00[i,j] + B11[i,j]
        
        # The Strassen decomposition of Left * Right gives 7 terms, each of which
        # is a product of linear combinations of entries of Left and Right.
        # But Left and Right are themselves linear combinations of A and B blocks.
        
        # This is getting complex. Let me use a different approach:
        # Directly construct the 49-term decomposition for 4x4x4.
        
        # Alternative: Use the known fact that 4x4x4 has rank 49 via the
        # tensor product of two 2x2x2 Strassen decompositions.
        
        # I'll implement it by nesting: for each of the 7 outer Strassen terms,
        # decompose the 2x2*2x2 block product using 7 inner Strassen terms.
        
        # Outer term definitions (left_factor, right_factor, output_block, output_sign):
        # left_factor: dict mapping block index to coefficient
        # right_factor: dict mapping block index to coefficient
        # output_block: which C block this contributes to
        # output_sign: +1 or -1
        
        outer_terms = [
            # M1 = (A00+A11)(B00+B11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, [(0,0, 1), (0,1, -1), (1,0, -1), (1,1, 1)]),
            # M2 = (A10+A11)B00
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, [(0,1, 1), (1,0, -1)]),
            # M3 = A00(B10-B11)
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, [(1,0, 1), (1,1, 1)]),
            # M4 = A11(B01-B10)
            ({(1,1): 1}, {(0,1): 1, (1,0): -1}, [(0,0, -1), (0,1, 1)]),
            # M5 = (A00+A10)B11
            ({(0,0): 1, (1,0): 1}, {(1,1): 1}, [(0,0, 1), (1,0, 1)]),
            # M6 = (A10-A00)(B00+B10)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, [(1,1, 1)]),
            # M7 = (A11-A01)(B01+B11)
            ({(1,1): 1, (0,1): -1}, {(0,1): 1, (1,1): 1}, [(0,0, 1)]),
        ]
        
        # Inner Strassen terms for 2x2*2x2:
        # Each inner term has U (4-vector over 2x2 left), V (4-vector over 2x2 right), W (4-vector over 2x2 output)
        inner_terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,-1,1,0]),
            ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
            ([0,0,0,1], [0,1,-1,0], [-1,0,1,0]),
            ([1,0,1,0], [0,0,0,1], [1,1,0,0]),
            ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),
            ([0,-1,0,1], [0,1,0,1], [1,0,0,0]),
        ]
        
        # For each outer term, for each inner term, we get one of the 49 terms.
        # The outer term specifies which A blocks and B blocks are involved,
        # and the inner term specifies how the 2x2*2x2 product is decomposed.
        
        for outer_left, outer_right, outer_outputs in outer_terms:
            # outer_left: dict from (block_row, block_col) to coefficient
            # outer_right: dict from (block_row, block_col) to coefficient
            # outer_outputs: list of (block_row, block_col, sign)
            
            for u_inner, v_inner, w_inner in inner_terms:
                # u_inner: 4-vector over (li, lj) for left 2x2
                # v_inner: 4-vector over (lj, lk) for right 2x2
                # w_inner: 4-vector over (lk, li) for output 2x2
                
                # For this combined term, the U vector is over all 4x4 A entries:
                # A[i,j] where i = block_row*2 + li, j = block_col*2 + lj
                
                # The left factor in the outer term is: sum over (br, bc) of coeff * A[br,bc]
                # where A[br,bc] is the 2x2 block at (br, bc).
                # So A[i,j] = sum over (br, bc) where i=br*2+li, j=bc*2+lj of coeff[br,bc] * A_block[br,bc][li,lj]
                
                # For the inner Strassen term, u_inner[li*2+lj] gives the coefficient of A_left[li,lj]
                # in the left factor of the inner term.
                
                # So the combined U coefficient for A[i,j] where i=br*2+li, j=bc*2+lj is:
                # outer_left.get((br, bc), 0) * u_inner[li*2+lj]
                
                # Similarly for V and W.
                
                # Let me construct the full U, V, W vectors for this 49th term.
                # U has 16 entries (4x4 A matrix), indexed by i*4+j
                # V has 24 entries (4x6 B matrix), indexed by j*6+k
                # W has 24 entries (6x4 C matrix), indexed by k*4+i
                
                u_full = [0] * 16
                v_full = [0] * 24
                w_full = [0] * 24
                
                # Build U
                for (br, bc), coeff in outer_left.items():
                    for li in range(2):
                        for lj in range(2):
                            u_inner_val = u_inner[li * 2 + lj]
                            if coeff != 0 and u_inner_val != 0:
                                global_i = i0 + br * 2 + li
                                global_j = j0 + bc * 2 + lj
                                u_full[global_i * 4 + global_j] += coeff * u_inner_val
                
                # Build V
                for (br, bc), coeff in outer_right.items():
                    # br is the row block of B, bc is the col block of B
                    # B is 4x6, so br in 0..1, bc in 0..2 for 2x2 blocks
                    # Wait, B is 4x6. The 2x2 block at (br, bc) has rows br*2..br*2+1, cols bc*2..bc*2+1
                    # But B has 6 columns, so bc can be 0, 1, 2 (for 2x2 blocks covering cols 0-1, 2-3, 4-5)
                    # However, in the 4x4x4 context, B is 4x4. For 4x4x6, we need to be careful.
                    
                    # Actually, for the 4x4x4 block, B is 4x4. The 2x2 blocks are at:
                    # (0,0), (0,1), (1,0), (1,1) covering rows 0-1/2-3 and cols 0-1/2-3
                    # But B is 4x6, so this only works for the first 4 columns of B.
                    
                    # For the 4x4x6 problem, I'm using 4x4 Strassen for the first 4 columns
                    # and 2x2 Strassen for the last 2 columns. So the 4x4 block only involves
                    # the first 4 columns of B.
                    
                    for lj in range(2):
                        for lk in range(2):
                            v_inner_val = v_inner[lj * 2 + lk]
                            if coeff != 0 and v_inner_val != 0:
                                global_j = j0 + br * 2 + lj
                                global_k = k0 + bc * 2 + lk
                                v_full[global_j * 6 + global_k] += coeff * v_inner_val
                
                # Build W
                for (ob, oc), sign in outer_outputs:
                    # ob, oc is the C block index
                    for lk in range(2):
                        for li in range(2):
                            w_inner_val = w_inner[lk * 2 + li]
                            if sign != 0 and w_inner_val != 0:
                                global_k = k0 + ob * 2 + lk
                                global_i = i0 + oc * 2 + li
                                w_full[global_k * 4 + global_i] += sign * w_inner_val
                
                # Add to U, V, W
                for idx in range(16):
                    if u_full[idx] != 0:
                        U[idx][offset] += u_full[idx]
                for idx in range(24):
                    if v_full[idx] != 0:
                        V[idx][offset] += v_full[idx]
                for idx in range(24):
                    if w_full[idx] != 0:
                        W[idx][offset] += w_full[idx]
                
                offset += 1
        
        return offset

    def verify(U, V, W):
        """Verify that the decomposition exactly reconstructs matrix multiplication."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Build matrices
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
        # the (i,k) entry of A*B where A is 4x4, B is 4x6
        # Actually, the tensor decomposition satisfies:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # And the decomposition gives:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[j,k]... no.
        
        # The correct check: for all i, j, k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if this is the
        # "multiplication tensor" T[i,j,k] = 1 for all i,j,k.
        
        # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
        # (since (A*B)[i,k] = sum_j A[i,j]*B[j,k], the tensor is all ones).
        
        # So we check: for all i, j, k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    # Try the hybrid decomposition
    try:
        rank_total = 49 + 28  # 4x4 Strassen + two 2x2 Strassen blocks
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        # 4x4 Strassen for first 4 columns
        offset = get_4x4_strassen_block(0, 0, 0, U, V, W, 0)
        
        # 2x2 Strassen for last 2 columns
        # B has columns 4,5. So k0=4, and we need 2x2 blocks for i=0,1 and i=2,3
        offset = get_2x2_strassen_block(0, 0, 4, U, V, W, offset)
        offset = get_2x2_strassen_block(2, 0, 4, U, V, W, offset)
        
        # Verify
        if verify(U, V, W):
            if time.time() < deadline:
                return {"U": U, "V": V, "W": W}
        
        # If verification fails, try a different approach or fall back
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
