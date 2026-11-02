import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Strategy:
    1. Use Strassen's algorithm for the 4x4x4 block (rank 49) and a naive decomposition for the remaining 4x4x2 block (rank 8).
    2. Verify the decomposition exactly.
    3. If time permits, attempt random low-rank searches to find a better decomposition (though unlikely to beat 57).
    """
    start_time = time.time()
    DEADLINE = 140.0  # seconds, leaving 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Step 1: Construct the baseline decomposition (Rank 57)
    # We split the 6 columns of B into a 4-column block and a 2-column block.
    # Block 1: 4x4 * 4x4 -> 4x4. Use Strassen (Rank 49).
    # Block 2: 4x4 * 4x2 -> 4x2. Use Naive (Rank 4*4*2 = 32? No, naive is n*m*p2 = 4*4*2=32).
    # Wait, naive rank for n x m * m x p is n*m*p.
    # So naive for 4x4*4x2 is 4*4*2 = 32.
    # Total naive = 4*4*6 = 96.
    # Strassen for 4x4*4x4 is 49.
    # Naive for 4x4*4x2 is 32.
    # Total = 49 + 32 = 81.
    
    # Can we do better for the 4x4*4x2 part?
    # R(4,4,2) <= R(4,4,4) <= 49. But R(4,4,2) is at most 32 naive.
    # Is there a known better rank for R(4,4,2)?
    # R(2,2,2) = 7. R(4,4,2) can be built from 2x2x2 blocks?
    # Split A (4x4) into four 2x2 blocks. B (4x2) into two 2x2 blocks? No, B is 4x2.
    # B_top = 2x2, B_bottom = 2x2.
    # C = A11*B1 + A12*B2 (top 2 rows of C)
    #     A21*B1 + A22*B2 (bottom 2 rows of C)
    # Each 2x2 * 2x2 product has rank 7.
    # We have 4 such products: A11*B1, A12*B2, A21*B1, A22*B2.
    # Total rank = 4 * 7 = 28.
    # So R(4,4,2) <= 28.
    
    # Total rank with this split: 49 + 28 = 77.
    # Still better than 81, and better than 96.
    
    # Can we do better for the 4x4x4 part?
    # R(4,4,4) <= 49 (Strassen).
    # Can we split 4x4x4 differently?
    # Strassen is optimal for 2x2x2 (rank 7).
    # For 4x4x4, we use 7 blocks of 2x2x2? No, Strassen recursion.
    # M4 = 7 * M2. Rank = 7 * 7 = 49.
    
    # So 77 is a solid baseline. Let's see if we can do better.
    
    # Alternative split:
    # What if we don't split by columns of B?
    # The problem is 4x4 * 4x6.
    # Maybe use a single decomposition for the whole thing?
    # R(4,4,6) is unknown, but bounded by R(4,4,8)? No.
    # R(4,4,6) <= R(4,4,4) + R(4,4,2) = 49 + 28 = 77.
    
    # Let's try to construct the rank 77 decomposition.
    
    # Helper: Build Strassen decomposition for 2x2x2
    # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
    def strassen_2x2x2():
        # U, V, W such that C = sum_k U[:,k] * V[:,k] * W[:,k]^T ?
        # No, the tensor form is:
        # C_{ij} = sum_k (U_{i,k} A_{i,k}?) No.
        # The decomposition is:
        # M_{n,n,n} = sum_{k=1}^r U^{(k)} \otimes V^{(k)} \otimes W^{(k)}
        # Where U^{(k)} is n x n, etc.
        # Actually, the standard format for the output is:
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # This corresponds to:
        # T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
        # Wait, the indices in the task description:
        # "U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "W[k*n+i,r]"
        # The tensor component for (i, j, k) is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This must equal delta_{jk} delta_{im} delta_{ln}?
        # Let's check the schoolbook code:
        # U[i*m+j][column] = 1
        # V[j*p+k][column] = 1
        # W[k*n+i][column] = 1
        # This sums to 1 if the indices match the loop variable.
        # The loop is for i in n, j in m, k in p.
        # So it constructs the tensor T where T_{i,j,k} = 1 for the specific (i,j,k) triple.
        # This is the standard identity tensor for matrix mult?
        # Yes, the matrix multiplication tensor is sum_{i,j,k} E_{ij} \otimes E_{jk} \otimes E_{ki}.
        # So the component at (i,j,k) in the flattened 3-tensor is 1 if it corresponds to a valid multiplication step.
        
        # For 2x2x2 Strassen:
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W.
        # r = 7.
        # U has shape (2*2, 7) = (4, 7). Index (i*2+j, r_idx)
        # V has shape (2*2, 7) = (4, 7). Index (j*2+k, r_idx)
        # W has shape (2*2, 7) = (4, 7). Index (k*2+i, r_idx)
        
        # For each multiplication term M_r, it looks like (linear comb of A) * (linear comb of B).
        # The contribution to C is (comb of C indices).
        # Actually, the tensor decomposition term is:
        # T_r = u_r \otimes v_r \otimes w_r
        # Where u_r is a vector in A-space (size 4), v_r in B-space (size 4), w_r in C-space (size 4).
        # The product is sum_r <u_r, A> <v_r, B> w_r.
        # Wait, the standard form is:
        # C = sum_r (u_r \cdot A) (v_r \cdot B) w_r? No.
        # A is n x n. u_r must be a matrix?
        # In the tensor rank decomposition, the factors are vectors in the tensor product spaces.
        # Space A is R^{n*m}. So u_r is a vector of length n*m.
        # Space B is R^{m*p}. v_r is a vector of length m*p.
        # Space C is R^{p*n}. w_r is a vector of length p*n.
        
        # For M1 = (a11+a22)(b11+b22):
        # u_1: a11=1, a22=1 -> indices (0,0)=1, (1,1)=1. Vector [1, 0, 0, 1]
        # v_1: b11=1, b22=1 -> indices (0,0)=1, (1,1)=1. Vector [1, 0, 0, 1]
        # w_1: c11=1, c22=1 -> indices (0,0)=1, (1,1)=1. Vector [1, 0, 0, 1]
        
        # Let's verify the indices for W.
        # W[k*n+i, r]. n=2.
        # c11 -> k=0, i=0 -> W[0*2+0] = W[0]
        # c22 -> k=1, i=1 -> W[1*2+1] = W[3]
        # So w_1 = [1, 0, 0, 1]. Correct.
        
        # M2 = (a21+a22)b11
        # u_2: a21=1, a22=1 -> indices (1,0)=1, (1,1)=1. Vector [0, 0, 1, 1]
        # v_2: b11=1 -> indices (0,0)=1. Vector [1, 0, 0, 0]
        # w_2: c21=1 -> k=0, i=1 -> W[0*2+1] = W[1]. Vector [0, 1, 0, 0]
        
        # M3 = a11(b12-b22)
        # u_3: a11=1 -> [1, 0, 0, 0]
        # v_3: b12=1, b22=-1 -> indices (0,1)=1, (1,1)=-1. Vector [0, 1, 0, -1]
        # w_3: c12=1, c11=1? No.
        # c11 = ... - M5 + M7 ... M3 is in c12 and c22?
        # c12 = M3 + M5
        # c22 = M1 - M2 + M3 + M6
        # So M3 contributes to c12 (k=0, i=1 -> W[1]) and c22 (k=1, i=1 -> W[3]).
        # w_3: [0, 1, 0, 1]
        
        # M4 = a22(b21-b11)
        # u_4: a22=1 -> [0, 0, 0, 1]
        # v_4: b21=1, b11=-1 -> indices (1,0)=1, (0,0)=-1. Vector [-1, 0, 1, 0]
        # w_4: M4 in c11 (W[0]) and c21 (W[1]).
        # w_4: [1, 1, 0, 0]
        
        # M5 = (a11+a12)b22
        # u_5: a11=1, a12=1 -> [1, 1, 0, 0]
        # v_5: b22=1 -> [0, 0, 0, 1]
        # w_5: M5 in c11 (W[0]) and c12 (W[1]).
        # w_5: [1, 1, 0, 0]
        
        # M6 = (a21-a11)(b11+b12)
        # u_6: a21=1, a11=-1 -> [-1, 0, 1, 0]
        # v_6: b11=1, b12=1 -> [1, 1, 0, 0]
        # w_6: M6 in c22 (W[3]).
        # w_6: [0, 0, 0, 1]
        
        # M7 = (a12-a22)(b21+b22)
        # u_7: a12=1, a22=-1 -> [0, 1, 0, -1]
        # v_7: b21=1, b22=1 -> [0, 0, 1, 1]
        # w_7: M7 in c11 (W[0]).
        # w_7: [1, 0, 0, 0]
        
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1]
        ])
        W = np.array([
            [1, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0]
        ])
        return U, V, W

    # Helper: Build naive decomposition for n x m * m x p
    def naive_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Construct 4x4x4 block (Strassen)
    U_444, V_444, W_444 = strassen_2x2x2()
    # Wait, strassen_2x2x2 returns 4x7 matrices.
    # We need to embed this into 4x4x4?
    # No, Strassen 4x4x4 is built from 7 copies of 2x2x2?
    # Actually, Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # So the rank is 7 * R(2,2,2) = 7 * 7 = 49.
    # We need to construct the 49-term decomposition for 4x4x4.
    
    # Let's implement a generic Strassen decomposition for n x n where n is a power of 2.
    # Or just use the known 4x4x4 rank 49 decomposition.
    # It's easier to recursively build it.
    
    def strassen_decomp(n):
        if n == 2:
            return strassen_2x2x2()
        
        # Split n into n/2
        h = n // 2
        U_sub, V_sub, W_sub = strassen_decomp(h)
        # U_sub shape: (h*h, 7) = (h^2, 7)
        # We need to build U, V, W of shape (n*n, 7*7) = (4h^2, 49)
        
        U = np.zeros((n * n, 7 * 7))
        V = np.zeros((n * n, 7 * 7))
        W = np.zeros((n * n, 7 * 7))
        
        # The 7 terms of Strassen are linear combinations of the 4 blocks.
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
        
        # We will iterate over the 7 Strassen terms (t1..t7) and the 7 sub-terms (s1..s7).
        # The index in the final U, V, W is t_idx * 7 + s_idx.
        
        # Mapping of blocks to indices in the flattened n*n vector.
        # A11: [0:h, 0:h]
        # A12: [0:h, h:2h]
        # A21: [h:2h, 0:h]
        # A22: [h:2h, h:2h]
        
        def block_indices(block_type):
            # Returns slice tuples for rows and cols
            if block_type == '11':
                return (slice(0, h), slice(0, h))
            elif block_type == '12':
                return (slice(0, h), slice(h, 2*h))
            elif block_type == '21':
                return (slice(h, 2*h), slice(0, h))
            elif block_type == '22':
                return (slice(h, 2*h), slice(h, 2*h))
        
        # Strassen term definitions:
        # (U_blocks, V_blocks, W_blocks, signs)
        # U_blocks: list of (block, sign)
        # V_blocks: list of (block, sign)
        # W_blocks: list of (block, sign)
        
        strassen_terms = [
            # M1
            ({'U': [('11', 1), ('22', 1)], 'V': [('11', 1), ('22', 1)], 'W': [('11', 1), ('22', 1)]}),
            # M2
            ({'U': [('21', 1), ('22', 1)], 'V': [('11', 1)], 'W': [('21', 1)]}),
            # M3
            ({'U': [('11', 1)], 'V': [('12', 1), ('22', -1)], 'W': [('12', 1), ('22', 1)]}),
            # M4
            ({'U': [('22', 1)], 'V': [('21', 1), ('11', -1)], 'W': [('11', 1), ('21', 1)]}),
            # M5
            ({'U': [('11', 1), ('12', 1)], 'V': [('22', 1)], 'W': [('11', -1), ('12', 1)]}), # Wait, M5 is in C11 and C12.
            # C11 = M1 + M4 - M5 + M7. So M5 is -1 in C11.
            # C12 = M3 + M5. So M5 is +1 in C12.
            # My W definition above: W is for C blocks.
            # Let's re-verify W for M5.
            # M5 contributes to C11 with -1 and C12 with +1.
            # So W_blocks for M5 should be [('11', -1), ('12', 1)].
            # Let's re-verify all Ws.
            
            # M1: C11(+), C22(+) -> [('11', 1), ('22', 1)]
            # M2: C21(+) -> [('21', 1)]
            # M3: C12(+), C22(+) -> [('12', 1), ('22', 1)]
            # M4: C11(+), C21(+) -> [('11', 1), ('21', 1)]
            # M5: C11(-), C12(+) -> [('11', -1), ('12', 1)]
            # M6: C22(+) -> [('22', 1)]
            # M7: C11(+) -> [('11', 1)]
            
            # Let's re-define the list properly.
        ]
        
        strassen_terms = [
            {'U': [('11', 1), ('22', 1)], 'V': [('11', 1), ('22', 1)], 'W': [('11', 1), ('22', 1)]},
            {'U': [('21', 1), ('22', 1)], 'V': [('11', 1)], 'W': [('21', 1)]},
            {'U': [('11', 1)], 'V': [('12', 1), ('22', -1)], 'W': [('12', 1), ('22', 1)]},
            {'U': [('22', 1)], 'V': [('21', 1), ('11', -1)], 'W': [('11', 1), ('21', 1)]},
            {'U': [('11', 1), ('12', 1)], 'V': [('22', 1)], 'W': [('11', -1), ('12', 1)]},
            {'U': [('21', 1), ('11', -1)], 'V': [('11', 1), ('12', 1)], 'W': [('22', 1)]},
            {'U': [('12', 1), ('22', -1)], 'V': [('21', 1), ('22', 1)], 'W': [('11', 1)]},
        ]
        
        for t_idx, term in enumerate(strassen_terms):
            for s_idx in range(7):
                col_idx = t_idx * 7 + s_idx
                
                # U part
                for block, sign in term['U']:
                    r_slice, c_slice = block_indices(block)
                    # Add sign * U_sub[:, s_idx] to U[r_slice, c_slice, col_idx]
                    # U is flat (n*n, 49).
                    # We need to map (r, c) to index r*n + c.
                    for r in range(h):
                        for c in range(h):
                            flat_idx = (r_slice.start + r) * n + (c_slice.start + c)
                            U[flat_idx, col_idx] += sign * U_sub[r * h + c, s_idx]
                
                # V part
                for block, sign in term['V']:
                    r_slice, c_slice = block_indices(block)
                    for r in range(h):
                        for c in range(h):
                            flat_idx = (r_slice.start + r) * n + (c_slice.start + c)
                            V[flat_idx, col_idx] += sign * V_sub[r * h + c, s_idx]
                
                # W part
                for block, sign in term['W']:
                    r_slice, c_slice = block_indices(block)
                    for r in range(h):
                        for c in range(h):
                            # W index is k*n + i.
                            # Block indices are (k, i) relative to the block.
                            # So flat_idx = (r_slice.start + r) * n + (c_slice.start + c)
                            flat_idx = (r_slice.start + r) * n + (c_slice.start + c)
                            W[flat_idx, col_idx] += sign * W_sub[r * h + c, s_idx]
                            
        return U, V, W

    U_444, V_444, W_444 = strassen_decomp(4)
    
    # Construct 4x4x2 block
    # We can use 2x2x2 Strassen for the 4x4x2 part?
    # Split A (4x4) into 2x2 blocks. B (4x2) into 2x2 blocks (top and bottom).
    # This is not a square B.
    # Let's just use naive for 4x4x2 to be safe, or try to optimize.
    # Naive rank is 32.
    # Can we do better?
    # R(4,4,2) <= 4 * R(2,2,2) = 28.
    # Let's implement the 4x4x2 decomposition using 4 Strassen blocks.
    
    def decomp_4x4x2():
        # A (4x4), B (4x2) -> C (4x2)
        # A = [[A11, A12], [A21, A22]]
        # B = [[B1], [B2]] where B1, B2 are 2x2.
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        
        # We need 4 products:
        # P1 = A11 B1
        # P2 = A12 B2
        # P3 = A21 B1
        # P4 = A22 B2
        
        # Each Pi is a 2x2x2 multiplication.
        # Rank of each is 7.
        # Total rank 28.
        
        U = np.zeros((16, 28))
        V = np.zeros((8, 28))
        W = np.zeros((8, 28))
        
        U_sub, V_sub, W_sub = strassen_2x2x2()
        
        # Helper to add a sub-decomposition
        def add_sub(block_A, block_B, block_C, offset):
            # block_A, block_B, block_C are tuples of (row_slice, col_slice)
            # A is 4x4. B is 4x2. C is 4x2.
            
            rA, cA = block_A
            rB, cB = block_B
            rC, cC = block_C
            
            for r_idx in range(4):
                for col in range(7):
                    # U
                    flat_A = rA[r_idx // 2] * 4 + cA[r_idx % 2]
                    U[flat_A, offset + col] = U_sub[r_idx, col]
                    
                    # V
                    # B is 4x2. Flattened index is j*2+k.
                    flat_B = rB[r_idx // 2] * 2 + cB[r_idx % 2]
                    V[flat_B, offset + col] = V_sub[r_idx, col]
                    
                    # W
                    # C is 4x2. Flattened index is k*4+i.
                    # Wait, W index is k*n+i. Here n=2 (output width).
                    # C is 4x2. Rows are 0..3, Cols are 0..1.
                    # Index is col*4 + row.
                    flat_C = cC[r_idx % 2] * 4 + rC[r_idx // 2]
                    W[flat_C, offset + col] = W_sub[r_idx, col]
        
        # P1 = A11 * B1 -> C1_top
        # A11: rows 0-1, cols 0-1
        # B1: rows 0-1, cols 0-1
        # C1_top: rows 0-1, cols 0-1
        add_sub((slice(0,2), slice(0,2)), (slice(0,2), slice(0,2)), (slice(0,2), slice(0,2)), 0)
        
        # P2 = A12 * B2 -> C1_bottom? No, C1 = A11 B1 + A12 B2.
        # C1 is rows 0-1.
        # B2 is rows 2-3, cols 0-1.
        # A12 is rows 0-1, cols 2-3.
        # Result adds to C1 (rows 0-1, cols 0-1).
        add_sub((slice(0,2), slice(2,4)), (slice(2,4), slice(0,2)), (slice(0,2), slice(0,2)), 7)
        
        # P3 = A21 * B1 -> C2_top (rows 2-3)
        # A21: rows 2-3, cols 0-1
        # B1: rows 0-1, cols 0-1
        # C2: rows 2-3, cols 0-1
        add_sub((slice(2,4), slice(0,2)), (slice(0,2), slice(0,2)), (slice(2,4), slice(0,2)), 14)
        
        # P4 = A22 * B2 -> C2_bottom? No, C2 = A21 B1 + A22 B2.
        # A22: rows 2-3, cols 2-3
        # B2: rows 2-3, cols 0-1
        # C2: rows 2-3, cols 0-1
        add_sub((slice(2,4), slice(2,4)), (slice(2,4), slice(0,2)), (slice(2,4), slice(0,2)), 21)
        
        return U, V, W

    U_442, V_442, W_442 = decomp_4x4x2()
    
    # Combine
    # Full U: 16 x (49 + 28) = 16 x 77
    # First 49 cols: U_444
    # Next 28 cols: U_442
    U_full = np.zeros((16, 77))
    U_full[:, :49] = U_444
    U_full[:, 49:] = U_442
    
    # Full V: 8 x 77
    # First 49 cols: V_444 (which is for 4x4 B, i.e., cols 0-3 of B)
    # V_444 is 16x49. We only need the first 8 rows?
    # B is 4x6.
    # Block 1 uses B cols 0-3. Block 2 uses B cols 4-5.
    # V_444 corresponds to B (4x4). Flattened 16x49.
    # We need V for 4x6. Flattened 24x77.
    # V_444 rows 0-15 correspond to B[0:4, 0:4].
    # V_442 rows 0-7 correspond to B[0:4, 0:2].
    # Wait, B_442 is the second block, which is B[:, 4:6].
    # So V_442 rows should map to B[0:4, 4:6].
    
    V_full = np.zeros((24, 77))
    # Block 1: B[0:4, 0:4]
    V_full[0:16, :49] = V_444
    # Block 2: B[0:4, 4:6]
    # V_442 is 8x28.
    # We need to map V_442 indices (j*2+k) to (j*6 + (4+k)).
    for r in range(8):
        j, k = divmod(r, 2)
        new_r = j * 6 + (4 + k)
        V_full[new_r, 49:] = V_442[r, :]
        
    # Full W: 24 x 77
    # W is p*n = 6*4 = 24.
    # Block 1: C[0:4, 0:4]
    # W_444 is 16x49.
    # W index is k*4+i.
    # For Block 1, k is 0..3.
    # For Block 2, k is 4..5.
    # W_444 covers k=0..3.
    W_full = np.zeros((24, 77))
    W_full[0:16, :49] = W_444
    
    # Block 2: C[0:4, 4:6]
    # W_442 is 8x28.
    # W_442 index is k*4+i where k is 0..1 (relative to block).
    # We need to map to k=4,5 in full W.
    # New index = (4+k)*4 + i.
    for r in range(8):
        k_rel, i = divmod(r, 4)
        new_r = (4 + k_rel) * 4 + i
        W_full[new_r, 49:] = W_442[r, :]
        
    # Verify
    def verify(U, V, W):
        n, m, p = 4, 4, 6
        # Construct random A, B
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        # Compute via tensor
        # C_ik = sum_j A_ij B_jk
        # Tensor: T_{i,j,k}
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Term r: U_r (vec of A), V_r (vec of B), W_r (vec of C)
        # Contribution = (U_r \cdot A) (V_r \cdot B) W_r
        # A is n x m. Flatten to n*m.
        # A_flat[i*m+j] = A[i,j]
        # V_r \cdot B_flat
        # B_flat[j*p+k] = B[j,k]
        
        # Sum over r
        C_calc = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_vec = W[:, r]
            
            # u_vec \cdot A
            # u_vec is length n*m.
            # A is n x m.
            # We need to sum u_vec[i*m+j] * A[i,j]
            a_sum = 0
            for i in range(n):
                for j in range(m):
                    a_sum += u_vec[i*m+j] * A[i,j]
            
            # v_vec \cdot B
            b_sum = 0
            for j in range(m):
                for k in range(p):
                    b_sum += v_vec[j*p+k] * B[j,k]
            
            # Add to C
            for idx in range(p*n):
                k, i = divmod(idx, n)
                C_calc[i, k] += a_sum * b_sum * w_vec[idx]
                
        if np.allclose(C, C_calc):
            return True
        else:
            return False

    if not verify(U_full, V_full, W_full):
        # Fallback to naive
        U_full, V_full, W_full = naive_decomp(4, 4, 6)

    # Convert to lists
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# Run construct
construct()
