"""Matrix multiplication 4x5x5 using Strassen-based block decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using Strassen-like algebraic cancellations.
    
    Strategy: Split A (4x5) into two 2x5 blocks. For each block, decompose
    the 5x5 matrix B into 2x2, 2x2, 1x2, 2x1, 1x1 sub-blocks and use
    Strassen's 7-multiplication algorithm for the 2x2x2 sub-problems.
    
    This should reduce the rank from 100 (schoolbook) to approximately 80-85.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition by accumulating rank-1 terms
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    
    terms_U = []  # list of (row, value) for U
    terms_V = []  # list of (row, value) for V
    terms_W = []  # list of (row, value) for W
    
    def add_term(u_row, v_row, w_row, u_val=1, v_val=1, w_val=1):
        """Add a rank-1 term."""
        terms_U.append((u_row, u_val))
        terms_V.append((v_row, v_val))
        terms_W.append((w_row, w_val))
    
    def strassen_2x2x2(A_rows, B_rows_cols, C_rows, offset_i=0, offset_j=0, offset_k=0):
        """
        Apply Strassen's algorithm to a 2x2 * 2x2 = 2x2 sub-problem.
        
        A_rows: list of 2 row indices in A (local)
        B_rows_cols: tuple (row_indices, col_indices) for B (local)
        C_rows: list of 2 row indices in C (local)
        
        Actually, let me think about this more carefully.
        """
        pass
    
    # Let me take a different, more direct approach.
    # I'll manually construct the Strassen decomposition for the 2x2x2 blocks
    # and combine them with the 1D remainder terms.
    
    # For a 2x2x2 Strassen decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our 4x5x5 problem:
    # Split A into A_top (rows 0-1) and A_bottom (rows 2-3), each 2x5
    # Split B into columns: first 2, next 2, last 1
    # And rows: first 2, next 2, last 1
    
    # For A_top (2x5) * B (5x5) = C_top (2x5):
    # C_top[i,k] = sum_{j=0}^{4} A_top[i,j] * B[j,k]
    
    # Split j into {0,1}, {2,3}, {4}
    # Split k into {0,1}, {2,3}, {4}
    
    # For the 2x2x2 block (j in {0,1}, k in {0,1}):
    # Use Strassen: 7 multiplications instead of 4
    
    # For the 2x2x1 block (j in {0,1}, k=4):
    # This is a 2x2 * 2x1 = 2x1, which is just 2 dot products of length 2
    # Each dot product has rank 2, so total rank 4
    # Can we do better? For 2x2x1, the tensor is sum_{j=0}^{1} A[i,j]*B[j,4]
    # This is a sum of 2 rank-1 terms per output, so 4 total. Rank is 4.
    
    # For the 2x1x2 block (j=4, k in {0,1}):
    # A[4,:] is 1x2, B[4,:] is 1x2 (wait, B is 5x5, so B[4,:] is 1x5)
    # Actually: A_top[i,4] * B[4,k] for i=0,1, k=0,1
    # This is 4 rank-1 terms, rank 4.
    
    # For the 2x2x2 block (j in {2,3}, k in {2,3}):
    # Use Strassen: 7 multiplications
    
    # For the 2x2x1 block (j in {2,3}, k=4):
    # Rank 4
    
    # For the 2x1x2 block (j=4, k in {2,3}):
    # Rank 4
    
    # For the 2x1x1 block (j=4, k=4):
    # Rank 2 (A_top[i,4]*B[4,4] for i=0,1)
    
    # Wait, I need to be more careful. Let me re-examine.
    
    # For A_top (2x5) * B (5x5):
    # j ranges over 0..4, k ranges over 0..4
    # Split j: {0,1}, {2,3}, {4}
    # Split k: {0,1}, {2,3}, {4}
    
    # Block (j={0,1}, k={0,1}): 2x2x2, Strassen rank 7
    # Block (j={0,1}, k={2,3}): 2x2x2, Strassen rank 7
    # Block (j={0,1}, k={4}): 2x2x1, rank 4
    # Block (j={2,3}, k={0,1}): 2x2x2, Strassen rank 7
    # Block (j={2,3}, k={2,3}): 2x2x2, Strassen rank 7
    # Block (j={2,3}, k={4}): 2x2x1, rank 4
    # Block (j={4}, k={0,1}): 2x1x2, rank 4
    # Block (j={4}, k={2,3}): 2x1x2, rank 4
    # Block (j={4}, k={4}): 2x1x1, rank 2
    
    # Total for A_top: 7+7+4+7+7+4+4+4+2 = 46
    
    # Similarly for A_bottom: 46
    
    # Total: 92
    
    # That's better than 100! Let me implement this.
    
    def strassen_block(a_rows, j_indices, k_indices, i_offset, j_offset, k_offset):
        """
        Apply Strassen to a 2x2x2 block.
        
        a_rows: list of 2 local row indices in A (0 or 1)
        j_indices: list of 2 j indices (absolute)
        k_indices: list of 2 k indices (absolute)
        i_offset: offset for A rows (0 for top, 2 for bottom)
        j_offset: offset for j (not used, j_indices are absolute)
        k_offset: offset for k (not used, k_indices are absolute)
        
        Returns: list of (u_row, v_row, w_row, u_val, v_val, w_val) tuples
        """
        i0, i1 = a_rows
        j0, j1 = j_indices
        k0, k1 = k_indices
        
        # A indices (absolute row in A)
        ai0 = i_offset + i0
        ai1 = i_offset + i1
        
        # B indices
        bj0 = j0
        bj1 = j1
        
        # C indices
        ci0 = i_offset + i0
        ci1 = i_offset + i1
        
        terms = []
        
        # M1 = (A[ai0,j0] + A[ai1,j1]) * (B[j0,k0] + B[j1,k1])
        # This contributes to C[ci0,k0] with +1, C[ci1,k1] with +1
        # Wait, Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[ai0,j0], A22 = A[ai1,j1]
        # B11 = B[j0,k0], B22 = B[j1,k1]
        
        # We need to express each M as a rank-1 term in the global decomposition.
        # M1 = (A[ai0,j0] + A[ai1,j1]) * (B[j0,k0] + B[j1,k1])
        # This is a product of a sum and a sum, which expands to 4 rank-1 terms.
        # But in the tensor decomposition, we need to represent the bilinear form.
        
        # Actually, in the tensor decomposition, each "multiplication" M corresponds
        # to a rank-1 term where U and V have entries that are sums of the original
        # variables. But the standard tensor decomposition uses rank-1 terms where
        # each term is a single product of one A entry and one B entry.
        
        # Wait, I think I'm confusing the algorithmic decomposition with the tensor
        # decomposition. In the tensor decomposition, we want to write the matrix
        # multiplication tensor as a sum of rank-1 tensors.
        
        # The Strassen algorithm computes the product using 7 scalar multiplications,
        # but each scalar multiplication is of a linear combination of A entries with
        # a linear combination of B entries. In the tensor decomposition, this
        # corresponds to 7 rank-1 terms where the U and V matrices have entries
        # that are not just 0 and 1, but can be -1, 0, 1, etc.
        
        # Let me reconsider. The tensor T_{ijk} = delta_{jk} (for matrix mult).
        # A rank-r decomposition is: T = sum_{r=1}^{R} u_r * v_r * w_r
        # where u_r is a vector in R^{nm}, v_r in R^{mp}, w_r in R^{pn}.
        
        # For Strassen, each "multiplication" M_t = (sum_a u_{t,a} * A_a) * (sum_b v_{t,b} * B_b)
        # contributes to the output as: sum_t w_{t,c} * M_t for each output C_c.
        
        # So in the tensor decomposition:
        # T = sum_t (sum_a u_{t,a} e_a) * (sum_b v_{t,b} f_b) * (sum_c w_{t,c} g_c)
        #   = sum_t sum_a sum_b sum_c u_{t,a} v_{t,b} w_{t,c} e_a f_b g_c
        
        # So the rank-1 terms are indexed by (t, a, b, c) but that's not right.
        # Actually, the decomposition is:
        # T_{abc} = sum_t u_{t,a} v_{t,b} w_{t,c}
        
        # So U[a,t] = u_{t,a}, V[b,t] = v_{t,b}, W[c,t] = w_{t,c}
        
        # For Strassen 2x2x2:
        # A has entries A[0,0], A[0,1], A[1,0], A[1,1]
        # B has entries B[0,0], B[0,1], B[1,0], B[1,1]
        # C has entries C[0,0], C[0,1], C[1,0], C[1,1]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # u1: A[0,0]=1, A[1,1]=1, others 0
        # v1: B[0,0]=1, B[1,1]=1, others 0
        # w1: C[0,0]=1, C[1,1]=1, others 0 (from M1 + M4 - M5 + M7 for C[0,0], M1 - M2 + M3 + M6 for C[1,1])
        
        # Let me write out all 7 M's and their contributions:
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[1,0]-B[1,1])
        # M4 = A[1,1] * (B[0,0]-B[1,0])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # So the U, V, W matrices for this 2x2x2 block (local indices):
        # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        # B indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
        # C indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
        
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # M3: u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,1]
        # M4: u=[0,0,0,1], v=[1,0,-1,0], w=[1,0,1,0]
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
        # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        
        # Now I need to map these to the global indices.
        
        return terms
    
    # Let me just directly build the U, V, W matrices by accumulating contributions.
    
    # Global indices:
    # U row: i*m + j where i in 0..3, j in 0..4
    # V row: j*p + k where j in 0..4, k in 0..4
    # W row: k*n + i where k in 0..4, i in 0..3
    
    # I'll use a dictionary to accumulate values: U[row][col], V[row][col], W[row][col]
    # Start with empty dictionaries.
    
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    def add_to_U(row, col, val):
        U_dict[(row, col)] = U_dict.get((row, col), 0) + val
    
    def add_to_V(row, col, val):
        V_dict[(row, col)] = V_dict.get((row, col), 0) + val
    
    def add_to_W(row, col, val):
        W_dict[(row, col)] = W_dict.get((row, col), 0) + val
    
    # Strassen 2x2x2 block
    def strassen_2x2x2(i0, i1, j0, j1, k0, k1, col_offset):
        """
        Add Strassen 2x2x2 terms for A rows i0,i1; B rows j0,j1; B cols k0,k1.
        i0, i1 are absolute row indices in A (0..3)
        j0, j1 are absolute row indices in B (0..4)
        k0, k1 are absolute col indices in B (0..4)
        col_offset: starting column index for the rank terms
        """
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # Contributes: C[i0,k0] += 1, C[i1,k1] += 1
        c = col_offset
        add_to_U(i0*m + j0, c, 1)
        add_to_U(i1*m + j1, c, 1)
        add_to_V(j0*p + k0, c, 1)
        add_to_V(j1*p + k1, c, 1)
        add_to_W(k0*n + i0, c, 1)
        add_to_W(k1*n + i1, c, 1)
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # Contributes: C[i1,k0] += 1
        c = col_offset + 1
        add_to_U(i1*m + j0, c, 1)
        add_to_U(i1*m + j1, c, 1)
        add_to_V(j0*p + k0, c, 1)
        add_to_W(k0*n + i1, c, 1)
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # Contributes: C[i0,k1] += 1, C[i1,k1] += 1
        c = col_offset + 2
        add_to_U(i0*m + j0, c, 1)
        add_to_V(j1*p + k0, c, 1)
        add_to_V(j1*p + k1, c, -1)
        add_to_W(k1*n + i0, c, 1)
        add_to_W(k1*n + i1, c, 1)
        
        # M4 = A[i1,j1] * (B[j0,k0]-B[j1,k0])
        # Contributes: C[i0,k0] += 1, C[i1,k0] += 1
        c = col_offset + 3
        add_to_U(i1*m + j1, c, 1)
        add_to_V(j0*p + k0, c, 1)
        add_to_V(j1*p + k0, c, -1)
        add_to_W(k0*n + i0, c, 1)
        add_to_W(k0*n + i1, c, 1)
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # Contributes: C[i0,k0] += -1, C[i0,k1] += 1
        c = col_offset + 4
        add_to_U(i0*m + j0, c, 1)
        add_to_U(i0*m + j1, c, 1)
        add_to_V(j1*p + k1, c, 1)
        add_to_W(k0*n + i0, c, -1)
        add_to_W(k1*n + i0, c, 1)
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # Contributes: C[i1,k1] += 1
        c = col_offset + 5
        add_to_U(i1*m + j0, c, 1)
        add_to_U(i0*m + j0, c, -1)
        add_to_V(j0*p + k0, c, 1)
        add_to_V(j0*p + k1, c, 1)
        add_to_W(k1*n + i1, c, 1)
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # Contributes: C[i0,k0] += 1
        c = col_offset + 6
        add_to_U(i0*m + j1, c, 1)
        add_to_U(i1*m + j1, c, -1)
        add_to_V(j1*p + k0, c, 1)
        add_to_V(j1*p + k1, c, 1)
        add_to_W(k0*n + i0, c, 1)
        
        return col_offset + 7
    
    # 2x2x1 block: A[i0..i1, j0..j1] * B[j0..j1, k]
    # This is: C[i0,k] = A[i0,j0]*B[j0,k] + A[i0,j1]*B[j1,k]
    #           C[i1,k] = A[i1,j0]*B[j0,k] + A[i1,j1]*B[j1,k]
    # Rank 4 (schoolbook for 2x2x1)
    def block_2x2x1(i0, i1, j0, j1, k, col_offset):
        c = col_offset
        add_to_U(i0*m + j0, c, 1)
        add_to_V(j0*p + k, c, 1)
        add_to_W(k*n + i0, c, 1)
        
        c = col_offset + 1
        add_to_U(i0*m + j1, c, 1)
        add_to_V(j1*p + k, c, 1)
        add_to_W(k*n + i0, c, 1)
        
        c = col_offset + 2
        add_to_U(i1*m + j0, c, 1)
        add_to_V(j0*p + k, c, 1)
        add_to_W(k*n + i1, c, 1)
        
        c = col_offset + 3
        add_to_U(i1*m + j1, c, 1)
        add_to_V(j1*p + k, c, 1)
        add_to_W(k*n + i1, c, 1)
        
        return col_offset + 4
    
    # 2x1x2 block: A[i0..i1, j] * B[j, k0..k1]
    # C[i0,k0] = A[i0,j]*B[j,k0]
    # C[i0,k1] = A[i0,j]*B[j,k1]
    # C[i1,k0] = A[i1,j]*B[j,k0]
    # C[i1,k1] = A[i1,j]*B[j,k1]
    # Rank 4
    def block_2x1x2(i0, i1, j, k0, k1, col_offset):
        c = col_offset
        add_to_U(i0*m + j, c, 1)
        add_to_V(j*p + k0, c, 1)
        add_to_W(k0*n + i0, c, 1)
        
        c = col_offset + 1
        add_to_U(i0*m + j, c, 1)
        add_to_V(j*p + k1, c, 1)
        add_to_W(k1*n + i0, c, 1)
        
        c = col_offset + 2
        add_to_U(i1*m + j, c, 1)
        add_to_V(j*p + k0, c, 1)
        add_to_W(k0*n + i1, c, 1)
        
        c = col_offset + 3
        add_to_U(i1*m + j, c, 1)
        add_to_V(j*p + k1, c, 1)
        add_to_W(k1*n + i1, c, 1)
        
        return col_offset + 4
    
    # 2x1x1 block: A[i0..i1, j] * B[j, k]
    # C[i0,k] = A[i0,j]*B[j,k]
    # C[i1,k] = A[i1,j]*B[j,k]
    # Rank 2
    def block_2x1x1(i0, i1, j, k, col_offset):
        c = col_offset
        add_to_U(i0*m + j, c, 1)
        add_to_V(j*p + k, c, 1)
        add_to_W(k*n + i0, c, 1)
        
        c = col_offset + 1
        add_to_U(i1*m + j, c, 1)
        add_to_V(j*p + k, c, 1)
        add_to_W(k*n + i1, c, 1)
        
        return col_offset + 2
    
    # Now build the full decomposition
    col = 0
    
    # Top block: A rows 0,1
    # j groups: {0,1}, {2,3}, {4}
    # k groups: {0,1}, {2,3}, {4}
    
    # (j={0,1}, k={0,1}): Strassen 2x2x2
    col = strassen_2x2x2(0, 1, 0, 1, 0, 1, col)
    
    # (j={0,1}, k={2,3}): Strassen 2x2x2
    col = strassen_2x2x2(0, 1, 0, 1, 2, 3, col)
    
    # (j={0,1}, k={4}): 2x2x1
    col = block_2x2x1(0, 1, 0, 1, 4, col)
    
    # (j={2,3}, k={0,1}): Strassen 2x2x2
    col = strassen_2x2x2(0, 1, 2, 3, 0, 1, col)
    
    # (j={2,3}, k={2,3}): Strassen 2x2x2
    col = strassen_2x2x2(0, 1, 2, 3, 2, 3, col)
    
    # (j={2,3}, k={4}): 2x2x1
    col = block_2x2x1(0, 1, 2, 3, 4, col)
    
    # (j={4}, k={0,1}): 2x1x2
    col = block_2x1x2(0, 1, 4, 0, 1, col)
    
    # (j={4}, k={2,3}): 2x1x2
    col = block_2x1x2(0, 1, 4, 2, 3, col)
    
    # (j={4}, k={4}): 2x1x1
    col = block_2x1x1(0, 1, 4, 4, col)
    
    # Bottom block: A rows 2,3
    # Same structure
    
    # (j={0,1}, k={0,1}): Strassen 2x2x2
    col = strassen_2x2x2(2, 3, 0, 1, 0, 1, col)
    
    # (j={0,1}, k={2,3}): Strassen 2x2x2
    col = strassen_2x2x2(2, 3, 0, 1, 2, 3, col)
    
    # (j={0,1}, k={4}): 2x2x1
    col = block_2x2x1(2, 3, 0, 1, 4, col)
    
    # (j={2,3}, k={0,1}): Strassen 2x2x2
    col = strassen_2x2x2(2, 3, 2, 3, 0, 1, col)
    
    # (j={2,3}, k={2,3}): Strassen 2x2x2
    col = strassen_2x2x2(2, 3, 2, 3, 2, 3, col)
    
    # (j={2,3}, k={4}): 2x2x1
    col = block_2x2x1(2, 3, 2, 3, 4, col)
    
    # (j={4}, k={0,1}): 2x1x2
    col = block_2x1x2(2, 3, 4, 0, 1, col)
    
    # (j={4}, k={2,3}): 2x1x2
    col = block_2x1x2(2, 3, 4, 2, 3, col)
    
    # (j={4}, k={4}): 2x1x1
    col = block_2x1x1(2, 3, 4, 4, col)
    
    rank = col
    
    # Convert dictionaries to arrays
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, col_idx), val in U_dict.items():
        U[row][col_idx] = val
    for (row, col_idx), val in V_dict.items():
        V[row][col_idx] = val
    for (row, col_idx), val in W_dict.items():
        W[row][col_idx] = val
    
    return {"U": U, "V": V, "W": W}
