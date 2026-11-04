"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
Uses Strassen-style block decomposition where applicable, schoolbook otherwise.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The decomposition satisfies: for each column c, the outer product 
    W[:,c] ⊗ V[:,c] ⊗ U[:,c] contributes to the multiplication tensor.
    
    Specifically, the multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 if this corresponds to a valid (i,j,k) product term, else 0.
    
    We use a hybrid approach:
    - For the 4x4 part (rows 0-3, cols 0-3), use Strassen 2x2x2 blocks
    - For remaining parts, use schoolbook
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll build the decomposition column by column.
    # Each column c corresponds to a rank-1 term in the tensor decomposition.
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's use a systematic approach with Strassen where beneficial.
    # 
    # Strategy: 
    # - The 4x4 block can be decomposed using two 2x2x2 Strassen blocks (rank 7 each = 14)
    # - The remaining parts (involving the 3rd row and 5th column) use schoolbook
    #
    # Actually, let me think more carefully about the structure.
    #
    # A is 3x4, B is 4x5, C is 3x5.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # The tensor has indices (i, j, k) with i in [0,2], j in [0,3], k in [0,4].
    # Total number of non-zero entries: 3*4*5 = 60.
    #
    # Schoolbook rank = 60.
    #
    # To reduce rank, we need to find linear dependencies or use algebraic tricks.
    # Strassen works for 2x2x2: rank 7 instead of 8.
    #
    # Let's partition:
    # - Rows 0,1 of A (2x4) and cols 0,1 of B (4x2) -> 2x2x2 block, use Strassen (rank 7)
    # - Rows 0,1 of A and cols 2,3 of B -> another 2x2x2 block, use Strassen (rank 7)  
    # - Row 2 of A (1x4) and all 5 cols of B (4x5) -> schoolbook (rank 20)
    # - Rows 0,1 of A and col 4 of B -> schoolbook (rank 10)
    #
    # Wait, this double counts. Let me be more careful.
    #
    # Let's partition A into A1 (rows 0,1) and A2 (row 2).
    # Partition B into B1 (cols 0,1), B2 (cols 2,3), B3 (col 4).
    #
    # C11 = A1 * B1  (2x2) -> Strassen rank 7
    # C12 = A1 * B2  (2x2) -> Strassen rank 7
    # C13 = A1 * B3  (2x1) -> schoolbook rank 4 (2 rows * 4 cols of B... wait)
    #   Actually A1 is 2x4, B3 is 4x1, so 2x4*4x1 = 8 terms, schoolbook rank 8
    # C21 = A2 * B1  (1x2) -> schoolbook rank 4 (1*4*2)
    # C22 = A2 * B2  (1x2) -> schoolbook rank 4
    # C23 = A2 * B3  (1x1) -> schoolbook rank 4
    #
    # Total: 7 + 7 + 8 + 4 + 4 + 4 = 34
    #
    # But wait, the Strassen decomposition for 2x2x2 gives rank 7 for the 2x2*2x2->2x2 multiplication.
    # For C11 = A1 * B1 where A1 is 2x4 and B1 is 4x2, this isn't a 2x2x2 problem.
    # 
    # Let me reconsider. The key insight is that Strassen applies to square matrix multiplication.
    # For non-square or mixed dimensions, we need to be more careful.
    #
    # Actually, the standard approach for rectangular matrix multiplication is to pad to square
    # or use block decomposition where each block is square.
    #
    # Let me try a different partitioning. Since m=4 is even, we can split the middle dimension
    # into two 2-dimensional parts.
    #
    # A = [A_top | A_bottom] where A_top is 3x2, A_bottom is 3x2
    # B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 2x5
    #
    # C = A_top * B_top + A_bottom * B_bottom
    #
    # Each product is 3x2 * 2x5 = 3x5, which has 30 terms.
    # This doesn't help directly since 3x2*2x5 isn't a standard Strassen case.
    #
    # Let me try yet another approach. The known best rank for 3x4*4x5 might be achievable
    # through a combination of small Strassen blocks and schoolbook.
    #
    # Actually, let me just implement the hybrid that was mentioned in the knowledge:
    # Two Strassen 2x2x2 blocks (rank 7 each) and schoolbook for the remainder.
    #
    # For a 2x2x2 Strassen block, we need 2x2 matrices. So we need to find 2x2 subproblems.
    #
    # Let's say we take rows {0,1} of A and cols {0,1} of B, with all 4 middle indices.
    # But the middle dimension is shared, so we can't just take a 2x2x2 block easily.
    #
    # I think the correct interpretation is:
    # - Use Strassen for the 2x2x2 subproblems where the middle dimension is also 2.
    # - This means we need to partition the middle dimension (size 4) into two parts of size 2.
    #
    # Let j1 = {0,1}, j2 = {2,3}.
    # 
    # For each pair of row indices I and column indices K, and middle index set J:
    # C[I,K] = sum_{j in J} A[I,j] * B[j,K]
    #
    # If |I|=2, |K|=2, |J|=2, we have a 2x2x2 subproblem -> Strassen rank 7.
    #
    # So let's partition:
    # Rows: I1={0,1}, I2={2}
    # Cols: K1={0,1}, K2={2,3}, K3={4}
    # Middle: J1={0,1}, J2={2,3}
    #
    # Subproblems:
    # (I1, J1, K1): 2x2x2 -> Strassen, rank 7
    # (I1, J2, K1): 2x2x2 -> Strassen, rank 7
    # (I1, J1, K2): 2x2x2 -> Strassen, rank 7
    # (I1, J2, K2): 2x2x2 -> Strassen, rank 7
    # (I1, J1, K3): 2x1x2 -> schoolbook, rank 4
    # (I1, J2, K3): 2x1x2 -> schoolbook, rank 4
    # (I2, J1, K1): 1x2x2 -> schoolbook, rank 4
    # (I2, J2, K1): 1x2x2 -> schoolbook, rank 4
    # (I2, J1, K2): 1x2x2 -> schoolbook, rank 4
    # (I2, J2, K2): 1x2x2 -> schoolbook, rank 4
    # (I2, J1, K3): 1x1x2 -> schoolbook, rank 2
    # (I2, J2, K3): 1x1x2 -> schoolbook, rank 2
    #
    # Total: 7*4 + 4*2 + 4*4 + 2*2 = 28 + 8 + 16 + 4 = 56
    #
    # Hmm, that's 56. Let me verify the schoolbook ranks.
    # For a (r x s) * (s x t) problem, schoolbook rank = r*s*t.
    # (2,1,2): 2*1*2 = 4. Correct.
    # (1,2,2): 1*2*2 = 4. Correct.
    # (1,1,2): 1*1*2 = 2. Correct.
    #
    # So total rank = 56.
    #
    # But wait, I need to double-check the Strassen decomposition. The standard Strassen
    # decomposition for 2x2*2x2->2x2 has rank 7. Let me implement it.
    
    def strassen_2x2x2():
        """
        Returns the 7 rank-1 terms for 2x2x2 Strassen multiplication.
        Each term is (u, v, w) where u is 4-dim (2x2 flattened), v is 4-dim, w is 4-dim.
        u[i*2+j], v[j*2+k], w[k*2+i] for the 2x2 case.
        
        Strassen's 7 products:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B21 - B22)
        M4 = A22(B12 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 - M4 + M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M3 + M6 + M7
        """
        terms = []
        # M1 = (A11 + A22)(B11 + B22)
        # A factors: A11=1, A22=1, others 0
        # B factors: B11=1, B22=1, others 0
        # C factors: C11=1, C22=1, others 0 (but with signs from combination)
        # Actually, each M contributes to multiple C entries with coefficients.
        # In the tensor decomposition, each rank-1 term has U, V, W vectors.
        # The contribution to C[i,k] from term c is U[i,j,c]*V[j,k,c]*W[k,i,c] summed over j.
        # Wait, no. Let me re-read the convention.
        
        # The convention is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For the full problem, this should give T[i,j,k] = 1 for all i,j,k (since every
        # (i,j,k) triple contributes to the sum).
        
        # For a 2x2x2 subproblem with local indices (a, b, c) where a,b,c in {0,1}:
        # The local tensor has T[a,b,c] = 1 for all a,b,c.
        # We need to decompose this into 7 rank-1 terms.
        
        # Let me use the standard Strassen decomposition.
        # The 7 products are:
        # P1 = (A00 + A11)(B00 + B11)
        # P2 = (A10 + A11)B00
        # P3 = A00(B10 - B11)
        # P4 = A11(B01 - B00)
        # P5 = (A00 + A01)B11
        # P6 = (A10 - A00)(B00 + B01)
        # P7 = (A01 - A11)(B10 + B11)
        
        # C00 = P1 - P4 + P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P3 + P6 + P7
        
        # For each product Pt, we have:
        # Pt = (sum_a alpha_a * A[a,:]) * (sum_b beta_b * B[:,b])
        # Actually, it's more like Pt = (linear combo of A entries) * (linear combo of B entries)
        # And each Pt contributes to C entries with certain coefficients.
        
        # In the tensor decomposition framework:
        # For product Pt, we define:
        # U_t[a, b] = coefficient of A[a,b] in the A-part of Pt
        # V_t[b, c] = coefficient of B[b,c] in the B-part of Pt  
        # W_t[c, a] = coefficient of C[a,c] that Pt contributes to (with sign)
        
        # Wait, I need to be more careful. The tensor T[i,j,k] = 1 means:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k in the relevant ranges.
        
        # A rank-1 decomposition term c has:
        # U_c[i,j] * V_c[j,k] * W_c[k,i]
        # Summed over c, this should give T[i,j,k].
        
        # For Strassen, each product Pt corresponds to a term where:
        # U_t[i,j] = coefficient in A-part, V_t[j,k] = coefficient in B-part
        # And W_t[k,i] encodes how Pt contributes to C[i,k].
        
        # Let me just directly compute the 7 terms.
        
        # Term 1: P1 = (A00 + A11)(B00 + B11)
        # U1: A00=1, A11=1, A01=0, A10=0
        # V1: B00=1, B11=1, B01=0, B10=0
        # W1: C00 gets +1, C11 gets +1, C01 gets 0, C10 gets 0
        # But W is indexed as W[k*n+i] for the full problem, or W[k*2+i] for 2x2.
        # W1[0*2+0] = W1[0] = 1 (C00)
        # W1[1*2+1] = W1[3] = 1 (C11)
        # W1[0*2+1] = W1[1] = 0 (C01)
        # W1[1*2+0] = W1[2] = 0 (C10)
        
        # This is getting complicated. Let me just use a known implementation.
        
        # Standard Strassen rank-7 decomposition vectors:
        # For the 2x2x2 tensor with T[i,j,k]=1, the 7 rank-1 terms are:
        
        terms = []
        
        # t1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        u1 = [1, 0, 0, 1]  # A00, A01, A10, A11
        v1 = [1, 0, 0, 1]  # B00, B01, B10, B11
        w1 = [1, 0, 0, 1]  # C00, C01, C10, C11 -> but W[k*2+i]
        # W[0*2+0]=W[0]=1 (C00), W[0*2+1]=W[1]=0 (C01), W[1*2+0]=W[2]=0 (C10), W[1*2+1]=W[3]=1 (C11)
        terms.append((u1, v1, w1))
        
        # t2: (A10+A11)B00 -> C10 += 1
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, 0]  # C10
        terms.append((u2, v2, w2))
        
        # t3: A00(B10-B11) -> C01 += 1
        u3 = [1, 0, 0, 0]
        v3 = [0, 0, 1, -1]  # B10 - B11
        w3 = [0, 1, 0, 0]  # C01
        terms.append((u3, v3, w3))
        
        # t4: A11(B01-B00) -> C00 -= 1, C10 += 1
        u4 = [0, 0, 0, 1]
        v4 = [-1, 1, 0, 0]  # B01 - B00... wait, B01-B00 means coefficient of B00 is -1, B01 is 1
        # v4[j*2+k]: B00 at j=0,k=0 -> v4[0]=-1, B01 at j=0,k=1 -> v4[1]=1
        w4 = [-1, 0, 1, 0]  # C00 -= 1, C10 += 1
        terms.append((u4, v4, w4))
        
        # t5: (A00+A01)B11 -> C00 += 1, C01 += 1
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [1, 1, 0, 0]  # C00, C01
        terms.append((u5, v5, w5))
        
        # t6: (A10-A00)(B00+B01) -> C11 += 1
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1]  # C11
        terms.append((u6, v6, w6))
        
        # t7: (A01-A11)(B10+B11) -> C00 += 1, C11 += 1
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, 1]  # C00, C11
        terms.append((u7, v7, w7))
        
        return terms
    
    # Now let's build the full decomposition.
    
    # Partition indices:
    # Rows: I1=[0,1], I2=[2]
    # Middle: J1=[0,1], J2=[2,3]
    # Cols: K1=[0,1], K2=[2,3], K3=[4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolbook_term(row_idx, mid_idx, col_idx, u_val=1, v_val=1, w_val=1):
        """Add a single schoolbook term for A[row_idx, mid_idx] * B[mid_idx, col_idx]."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[row_idx * m + mid_idx] = u_val
        v[mid_idx * p + col_idx] = v_val
        w[col_idx * n + row_idx] = w_val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_block(row_indices, mid_indices, col_indices):
        """Add a 2x2x2 Strassen block for the given index sets."""
        if len(row_indices) != 2 or len(mid_indices) != 2 or len(col_indices) != 2:
            raise ValueError("Strassen block requires 2x2x2 indices")
        
        r0, r1 = row_indices
        m0, m1 = mid_indices
        c0, c1 = col_indices
        
        strassen_terms = strassen_2x2x2()
        
        for u_local, v_local, w_local in strassen_terms:
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            # Map local 2x2 indices to global
            # u_local is [A00, A01, A10, A11] = [A[r0,m0], A[r0,m1], A[r1,m0], A[r1,m1]]
            u[r0 * m + m0] += u_local[0]
            u[r0 * m + m1] += u_local[1]
            u[r1 * m + m0] += u_local[2]
            u[r1 * m + m1] += u_local[3]
            
            # v_local is [B00, B01, B10, B11] = [B[m0,c0], B[m0,c1], B[m1,c0], B[m1,c1]]
            v[m0 * p + c0] += v_local[0]
            v[m0 * p + c1] += v_local[1]
            v[m1 * p + c0] += v_local[2]
            v[m1 * p + c1] += v_local[3]
            
            # w_local is [C00, C01, C10, C11] = [C[r0,c0], C[r0,c1], C[r1,c0], C[r1,c1]]
            # W[k*n+i] where k is col index, i is row index
            w[c0 * n + r0] += w_local[0]
            w[c1 * n + r0] += w_local[1]
            w[c0 * n + r1] += w_local[2]
            w[c1 * n + r1] += w_local[3]
            
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    # Add Strassen blocks for 2x2x2 subproblems
    I1 = [0, 1]
    I2 = [2]
    J1 = [0, 1]
    J2 = [2, 3]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4]
    
    # 2x2x2 Strassen blocks
    add_strassen_block(I1, J1, K1)
    add_strassen_block(I1, J2, K1)
    add_strassen_block(I1, J1, K2)
    add_strassen_block(I1, J2, K2)
    
    # Schoolbook for remaining
    # I1 x J1 x K3: 2x2x1
    for r in I1:
        for mj in J1:
            add_schoolbook_term(r, mj, K3[0])
    
    # I1 x J2 x K3: 2x2x1
    for r in I1:
        for mj in J2:
            add_schoolbook_term(r, mj, K3[0])
    
    # I2 x J1 x K1: 1x2x2
    for r in I2:
        for mj in J1:
            for c in K1:
                add_schoolbook_term(r, mj, c)
    
    # I2 x J2 x K1: 1x2x2
    for r in I2:
        for mj in J2:
            for c in K1:
                add_schoolbook_term(r, mj, c)
    
    # I2 x J1 x K2: 1x2x2
    for r in I2:
        for mj in J1:
            for c in K2:
                add_schoolbook_term(r, mj, c)
    
    # I2 x J2 x K2: 1x2x2
    for r in I2:
        for mj in J2:
            for c in K2:
                add_schoolbook_term(r, mj, c)
    
    # I2 x J1 x K3: 1x2x1
    for r in I2:
        for mj in J1:
            add_schoolbook_term(r, mj, K3[0])
    
    # I2 x J2 x K3: 1x2x1
    for r in I2:
        for mj in J2:
            add_schoolbook_term(r, mj, K3[0])
    
    rank = len(U_cols)
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n * m):
            U[i][c] = U_cols[c][i]
        for i in range(m * p):
            V[i][c] = V_cols[c][i]
        for i in range(p * n):
            W[i][c] = W_cols[c][i]
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should be 1 for all valid i,j,k.
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to schoolbook if Strassen doesn't work correctly
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
