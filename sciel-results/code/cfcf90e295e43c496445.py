import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x5 matrix.
    
    Strategy:
    Split the 4x5 matrix B into B1 (4x4) and B2 (4x1).
    C = A * B = A * B1 + A * B2 = C1 + C2.
    
    For C1 = A * B1 (4x4 * 4x4), use recursive Strassen (rank 49).
    For C2 = A * B2 (4x4 * 4x1), use recursive Strassen-style (rank 16).
    
    Total rank = 49 + 16 = 65.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We will accumulate rank-1 terms
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Index helpers
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    # ------------------------------------------------------------------
    # Strassen 2x2x2 decomposition
    # Multiplies A (2x2) by B (2x2) to get C (2x2)
    # A is at rows ai..ai+1, cols aj..aj+1
    # B is at rows bj..bj+1, cols bk..bk+1
    # C is at rows ai..ai+1, cols bk..bk+1
    # ------------------------------------------------------------------
    def strassen_2x2x2(ai, aj, bj, bk):
        # Local elements:
        # a=A[ai,aj], b=A[ai,aj+1], c=A[ai+1,aj], d=A[ai+1,aj+1]
        # e=B[bj,bk], f=B[bj,bk+1], g=B[bj+1,bk], h=B[bj+1,bk+1]
        
        # M1 = (a+d)(e+h)
        u1 = np.zeros(n*m); u1[u_idx(ai,aj)] = 1; u1[u_idx(ai+1,aj+1)] = 1
        v1 = np.zeros(m*p); v1[v_idx(bj,bk)] = 1; v1[v_idx(bj+1,bk+1)] = 1
        w1 = np.zeros(p*n); w1[w_idx(bk,ai)] = 1; w1[w_idx(bk+1,ai+1)] = 1
        add_term(u1, v1, w1)
        
        # M2 = (c+d)e
        u2 = np.zeros(n*m); u2[u_idx(ai+1,aj)] = 1; u2[u_idx(ai+1,aj+1)] = 1
        v2 = np.zeros(m*p); v2[v_idx(bj,bk)] = 1
        w2 = np.zeros(p*n); w2[w_idx(bk,ai+1)] = 1; w2[w_idx(bk+1,ai+1)] = -1
        add_term(u2, v2, w2)
        
        # M3 = a(f-h)
        u3 = np.zeros(n*m); u3[u_idx(ai,aj)] = 1
        v3 = np.zeros(m*p); v3[v_idx(bj,bk+1)] = 1; v3[v_idx(bj+1,bk+1)] = -1
        w3 = np.zeros(p*n); w3[w_idx(bk+1,ai)] = 1; w3[w_idx(bk+1,ai+1)] = 1
        add_term(u3, v3, w3)
        
        # M4 = d(g-e)
        u4 = np.zeros(n*m); u4[u_idx(ai+1,aj+1)] = 1
        v4 = np.zeros(m*p); v4[v_idx(bj+1,bk)] = 1; v4[v_idx(bj,bk)] = -1
        w4 = np.zeros(p*n); w4[w_idx(bk,ai)] = 1; w4[w_idx(bk,ai+1)] = 1
        add_term(u4, v4, w4)
        
        # M5 = (a+b)h
        u5 = np.zeros(n*m); u5[u_idx(ai,aj)] = 1; u5[u_idx(ai,aj+1)] = 1
        v5 = np.zeros(m*p); v5[v_idx(bj+1,bk+1)] = 1
        w5 = np.zeros(p*n); w5[w_idx(bk,ai)] = -1; w5[w_idx(bk+1,ai)] = 1
        add_term(u5, v5, w5)
        
        # M6 = (c-a)(e+f)
        u6 = np.zeros(n*m); u6[u_idx(ai+1,aj)] = 1; u6[u_idx(ai,aj)] = -1
        v6 = np.zeros(m*p); v6[v_idx(bj,bk)] = 1; v6[v_idx(bj,bk+1)] = 1
        w6 = np.zeros(p*n); w6[w_idx(bk+1,ai+1)] = 1
        add_term(u6, v6, w6)
        
        # M7 = (b-d)(g+h)
        u7 = np.zeros(n*m); u7[u_idx(ai,aj+1)] = 1; u7[u_idx(ai+1,aj+1)] = -1
        v7 = np.zeros(m*p); v7[v_idx(bj+1,bk)] = 1; v7[v_idx(bj+1,bk+1)] = 1
        w7 = np.zeros(p*n); w7[w_idx(bk,ai)] = 1
        add_term(u7, v7, w7)
    
    # ------------------------------------------------------------------
    # 2x2x1 decomposition (A 2x2 * B 2x1 = C 2x1)
    # A is at rows ai..ai+1, cols aj..aj+1
    # B is at rows bj..bj+1, col bk
    # C is at rows ai..ai+1, col bk
    # Rank 4: schoolbook for 2x2x1
    # Actually, we can do better. Let's use rank 4 for 2x2x1.
    # C0 = a*e + b*g
    # C1 = c*e + d*g
    # This is 4 terms.
    # ------------------------------------------------------------------
    def naive_2x2x1(ai, aj, bj, bk):
        # a=A[ai,aj], b=A[ai,aj+1], c=A[ai+1,aj], d=A[ai+1,aj+1]
        # e=B[bj,bk], g=B[bj+1,bk]
        
        # Term 1: a*e -> C[ai,bk]
        u1 = np.zeros(n*m); u1[u_idx(ai,aj)] = 1
        v1 = np.zeros(m*p); v1[v_idx(bj,bk)] = 1
        w1 = np.zeros(p*n); w1[w_idx(bk,ai)] = 1
        add_term(u1, v1, w1)
        
        # Term 2: b*g -> C[ai,bk]
        u2 = np.zeros(n*m); u2[u_idx(ai,aj+1)] = 1
        v2 = np.zeros(m*p); v2[v_idx(bj+1,bk)] = 1
        w2 = np.zeros(p*n); w2[w_idx(bk,ai)] = 1
        add_term(u2, v2, w2)
        
        # Term 3: c*e -> C[ai+1,bk]
        u3 = np.zeros(n*m); u3[u_idx(ai+1,aj)] = 1
        v3 = np.zeros(m*p); v3[v_idx(bj,bk)] = 1
        w3 = np.zeros(p*n); w3[w_idx(bk,ai+1)] = 1
        add_term(u3, v3, w3)
        
        # Term 4: d*g -> C[ai+1,bk]
        u4 = np.zeros(n*m); u4[u_idx(ai+1,aj+1)] = 1
        v4 = np.zeros(m*p); v4[v_idx(bj+1,bk)] = 1
        w4 = np.zeros(p*n); w4[w_idx(bk,ai+1)] = 1
        add_term(u4, v4, w4)
    
    # ------------------------------------------------------------------
    # 4x4x4 decomposition using recursive Strassen
    # A (4x4) * B (4x4) = C (4x4)
    # Split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # 
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_i is a 2x2 matrix product of linear combinations.
    # We can compute each M_i using strassen_2x2x2 on the appropriate blocks.
    # But the inputs are sums/differences of blocks.
    # 
    # Instead of building the full 49-term decomposition explicitly,
    # we can use the following approach:
    # For each of the 7 Strassen products, we need to multiply two 2x2 matrices
    # that are linear combinations of the 4x4 blocks.
    # 
    # Let's define a helper that multiplies two 2x2 matrices where each element
    # is a linear combination of A and B elements.
    # 
    # Actually, it's easier to just expand. Each M_i involves 4 elements from A
    # and 4 elements from B (in linear combinations). The product is a 2x2 matrix.
    # We can use strassen_2x2x2 on the "effective" 2x2 blocks.
    # 
    # Let's define the 7 products:
    # M1: U1 = A00+A11, V1 = B00+B11
    # M2: U2 = A10+A11, V2 = B00
    # M3: U3 = A00, V3 = B01-B11
    # M4: U4 = A11, V4 = B10-B00
    # M5: U5 = A00+A01, V5 = B11
    # M6: U6 = A10-A00, V6 = B00+B01
    # M7: U7 = A01-A11, V7 = B10+B11
    #
    # For each product, U_i and V_i are 2x2 matrices of linear forms.
    # We can compute U_i * V_i using strassen_2x2x2 if we can map the linear forms
    # to specific indices.
    #
    # This is complex. Let's use a different approach:
    # We'll directly construct the 49 rank-1 terms by iterating over the
    # 7 Strassen products and, for each, the 7 sub-Strassen products.
    #
    # For product M_i = U_i * V_i, where U_i and V_i are 2x2 matrices:
    # U_i = [[u11, u12], [u21, u22]]
    # V_i = [[v11, v12], [v21, v22]]
    # Each u_ab and v_cd is a linear form in A and B.
    #
    # The product M_i is a 2x2 matrix:
    # M_i = [[m11, m12], [m21, m22]]
    # m11 = u11*v11 + u12*v21
    # m12 = u11*v12 + u12*v22
    # m21 = u21*v11 + u22*v21
    # m22 = u21*v12 + u22*v22
    #
    # Each m_ab is a product of two linear forms, so it's a rank-1 term
    # in the tensor decomposition.
    #
    # So for each M_i, we have 4 rank-1 terms (for m11, m12, m21, m22).
    # Total: 7 * 4 = 28 terms? No, that's not right.
    #
    # Wait, the standard Strassen 4x4x4 has rank 49. Let me reconsider.
    #
    # Actually, the rank of 4x4x4 using Strassen is 7^2 = 49.
    # This is because we decompose the 4x4x4 tensor into 7 "blocks" of 2x2x2 tensors.
    # Each 2x2x2 tensor has rank 7.
    # So total rank = 7 * 7 = 49.
    #
    # The 7 blocks correspond to the 7 Strassen products M_i.
    # Each M_i is a 2x2x2 tensor (not just a matrix, but a tensor that maps
    # 2x2 inputs to 2x2 output).
    #
    # So for each M_i, we need to decompose the 2x2x2 tensor into 7 rank-1 terms.
    # The inputs to this 2x2x2 tensor are the linear forms U_i and V_i.
    #
    # Let's implement this by defining the 7 rank-1 terms for each M_i.
    #
    # For M_i = U_i * V_i, where U_i and V_i are 2x2 matrices of linear forms:
    # We use the standard Strassen 2x2x2 decomposition.
    # The 7 products are:
    # P1 = (U11+U22)(V11+V22)
    # P2 = (U21+U22)V11
    # P3 = U11(V12-V22)
    # P4 = U22(V21-V11)
    # P5 = (U11+U12)V22
    # P6 = (U21-U11)(V11+V12)
    # P7 = (U12-U22)(V21+V22)
    #
    # And the output is:
    # M11 = P1 + P4 - P5 + P7
    # M12 = P3 + P5
    # M21 = P2 + P4
    # M22 = P1 - P2 + P3 + P6
    #
    # Each P_j is a product of two linear forms (which are themselves linear
    # combinations of A and B elements). So each P_j is a rank-1 term.
    #
    # Let's define the linear forms for U_i and V_i for each i.
    
    # Define linear forms as dictionaries mapping (row, col) to coefficient vectors
    # A linear form is a vector of length 16 (for A) or 20 (for B)
    
    # Let's represent linear forms as numpy arrays of length 16 (for A) or 20 (for B)
    
    # For M1: U1 = A00 + A11, V1 = B00 + B11
    # U1 is a 2x2 matrix where each element is a linear form.
    # U1[0,0] = A[0,0] + A[2,2]
    # U1[0,1] = A[0,1] + A[2,3]
    # U1[1,0] = A[1,0] + A[3,2]
    # U1[1,1] = A[1,1] + A[3,3]
    
    def make_A_form(indices):
        """Create a linear form for A from a list of (i,j) indices with +1 coeff."""
        form = np.zeros(n*m)
        for i, j in indices:
            form[u_idx(i, j)] = 1
        return form
    
    def make_B_form(indices):
        """Create a linear form for B from a list of (j,k) indices with +1 coeff."""
        form = np.zeros(m*p)
        for j, k in indices:
            form[v_idx(j, k)] = 1
        return form
    
    # Define the 7 Strassen products for 4x4x4
    # Each product is defined by U_i (2x2 matrix of A forms) and V_i (2x2 matrix of B forms)
    
    # M1: U1 = A00 + A11, V1 = B00 + B11
    U1 = [
        [make_A_form([(0,0), (2,2)]), make_A_form([(0,1), (2,3)])],
        [make_A_form([(1,0), (3,2)]), make_A_form([(1,1), (3,3)])]
    ]
    V1 = [
        [make_B_form([(0,0), (2,2)]), make_B_form([(0,1), (2,3)])],
        [make_B_form([(1,0), (3,2)]), make_B_form([(1,1), (3,3)])]
    ]
    
    # M2: U2 = A10 + A11, V2 = B00
    U2 = [
        [make_A_form([(2,0), (0,0)]), make_A_form([(2,1), (0,1)])],
        [make_A_form([(3,0), (1,0)]), make_A_form([(3,1), (1,1)])]
    ]
    V2 = [
        [make_B_form([(0,0)]), make_B_form([(0,1)])],
        [make_B_form([(1,0)]), make_B_form([(1,1)])]
    ]
    
    # M3: U3 = A00, V3 = B01 - B11
    U3 = [
        [make_A_form([(0,0)]), make_A_form([(0,1)])],
        [make_A_form([(1,0)]), make_A_form([(1,1)])]
    ]
    V3 = [
        [make_B_form([(0,1)]), make_B_form([(0,2)])],
        [make_B_form([(1,1)]), make_B_form([(1,2)])]
    ]
    # Subtract B11
    V3[0][0] -= make_B_form([(2,2)])
    V3[0][1] -= make_B_form([(2,3)])
    V3[1][0] -= make_B_form([(3,2)])
    V3[1][1] -= make_B_form([(3,3)])
    
    # M4: U4 = A11, V4 = B10 - B00
    U4 = [
        [make_A_form([(2,2)]), make_A_form([(2,3)])],
        [make_A_form([(3,2)]), make_A_form([(3,3)])]
    ]
    V4 = [
        [make_B_form([(1,0)]), make_B_form([(1,1)])],
        [make_B_form([(2,0)]), make_B_form([(2,1)])]
    ]
    # Subtract B00
    V4[0][0] -= make_B_form([(0,0)])
    V4[0][1] -= make_B_form([(0,1)])
    V4[1][0] -= make_B_form([(1,0)])
    V4[1][1] -= make_B_form([(1,1)])
    
    # M5: U5 = A00 + A01, V5 = B11
    U5 = [
        [make_A_form([(0,0), (0,2)]), make_A_form([(0,1), (0,3)])],
        [make_A_form([(1,0), (1,2)]), make_A_form([(1,1), (1,3)])]
    ]
    V5 = [
        [make_B_form([(2,2)]), make_B_form([(2,3)])],
        [make_B_form([(3,2)]), make_B_form([(3,3)])]
    ]
    
    # M6: U6 = A10 - A00, V6 = B00 + B01
    U6 = [
        [make_A_form([(2,0)]), make_A_form([(2,1)])],
        [make_A_form([(3,0)]), make_A_form([(3,1)])]
    ]
    # Subtract A00
    U6[0][0] -= make_A_form([(0,0)])
    U6[0][1] -= make_A_form([(0,1)])
    U6[1][0] -= make_A_form([(1,0)])
    U6[1][1] -= make_A_form([(1,1)])
    
    V6 = [
        [make_B_form([(0,0), (0,2)]), make_B_form([(0,1), (0,3)])],
        [make_B_form([(1,0), (1,2)]), make_B_form([(1,1), (1,3)])]
    ]
    
    # M7: U7 = A01 - A11, V7 = B10 + B11
    U7 = [
        [make_A_form([(0,2)]), make_A_form([(0,3)])],
        [make_A_form([(1,2)]), make_A_form([(1,3)])]
    ]
    # Subtract A11
    U7[0][0] -= make_A_form([(2,2)])
    U7[0][1] -= make_A_form([(2,3)])
    U7[1][0] -= make_A_form([(3,2)])
    U7[1][1] -= make_A_form([(3,3)])
    
    V7 = [
        [make_B_form([(1,0), (2,2)]), make_B_form([(1,1), (2,3)])],
        [make_B_form([(2,0), (3,2)]), make_B_form([(2,1), (3,3)])]
    ]
    
    # Now, for each M_i, we compute the 7 sub-products P_j using Strassen 2x2x2
    # on the linear forms U_i and V_i.
    
    def add_strassen_2x2_from_forms(U, V, w_offsets):
        """
        Add the 7 rank-1 terms for the Strassen 2x2x2 decomposition
        of the product of two 2x2 matrices of linear forms.
        
        U: 2x2 list of A forms (numpy arrays of length n*m)
        V: 2x2 list of B forms (numpy arrays of length m*p)
        w_offsets: 2x2 list of (row_offset, col_offset) for the C block
                   w_offsets[i][j] gives the offset (di, dj) for C[i,j]
        """
        # P1 = (U00+U11)(V00+V11)
        u1 = U[0][0] + U[1][1]
        v1 = V[0][0] + V[1][1]
        w1 = np.zeros(p*n)
        w1[w_idx(w_offsets[0][0], w_offsets[0][1])] = 1  # C00
        w1[w_idx(w_offsets[1][0]+1, w_offsets[1][1]+1)] = 1  # C11
        add_term(u1, v1, w1)
        
        # P2 = (U10+U11)V00
        u2 = U[1][0] + U[1][1]
        v2 = V[0][0]
        w2 = np.zeros(p*n)
        w2[w_idx(w_offsets[1][0], w_offsets[1][1])] = 1  # C10
        w2[w_idx(w_offsets[1][0]+1, w_offsets[1][1]+1)] = -1  # C11
        add_term(u2, v2, w2)
        
        # P3 = U00(V01-V11)
        u3 = U[0][0]
        v3 = V[0][1] - V[1][1]
        w3 = np.zeros(p*n)
        w3[w_idx(w_offsets[0][0]+1, w_offsets[0][1])] = 1  # C01
        w3[w_idx(w_offsets[1][0]+1, w_offsets[1][1]+1)] = 1  # C11
        add_term(u3, v3, w3)
        
        # P4 = U11(V10-V00)
        u4 = U[1][1]
        v4 = V[1][0] - V[0][0]
        w4 = np.zeros(p*n)
        w4[w_idx(w_offsets[0][0], w_offsets[0][1])] = 1  # C00
        w4[w_idx(w_offsets[1][0], w_offsets[1][1])] = 1  # C10
        add_term(u4, v4, w4)
        
        # P5 = (U00+U01)V11
        u5 = U[0][0] + U[0][1]
        v5 = V[1][1]
        w5 = np.zeros(p*n)
        w5[w_idx(w_offsets[0][0], w_offsets[0][1])] = -1  # C00
        w5[w_idx(w_offsets[0][0]+1, w_offsets[0][1])] = 1  # C01
        add_term(u5, v5, w5)
        
        # P6 = (U10-U00)(V00+V01)
        u6 = U[1][0] - U[0][0]
        v6 = V[0][0] + V[0][1]
        w6 = np.zeros(p*n)
        w6[w_idx(w_offsets[1][0]+1, w_offsets[1][1]+1)] = 1  # C11
        add_term(u6, v6, w6)
        
        # P7 = (U01-U11)(V10+V11)
        u7 = U[0][1] - U[1][1]
        v7 = V[1][0] + V[1][1]
        w7 = np.zeros(p*n)
        w7[w_idx(w_offsets[0][0], w_offsets[0][1])] = 1  # C00
        add_term(u7, v7, w7)
    
    # Now apply to each M_i
    # M1 contributes to C00 and C11
    # C00 is at rows 0-1, cols 0-1
    # C11 is at rows 2-3, cols 2-3
    w_offsets_M1 = [
        [(0, 0), (0, 1)],
        [(2, 2), (2, 3)]
    ]
    add_strassen_2x2_from_forms(U1, V1, w_offsets_M1)
    
    # M2 contributes to C10 and C11
    # C10 is at rows 2-3, cols 0-1
    # C11 is at rows 2-3, cols 2-3
    w_offsets_M2 = [
        [(2, 0), (2, 1)],
        [(2, 2), (2, 3)]
    ]
    add_strassen_2x2_from_forms(U2, V2, w_offsets_M2)
    
    # M3 contributes to C01 and C11
    # C01 is at rows 0-1, cols 2-3
    # C11 is at rows 2-3, cols 2-3
    w_offsets_M3 = [
        [(0, 2), (0, 3)],
        [(2, 2), (2, 3)]
    ]
    add_strassen_2x2_from_forms(U3, V3, w_offsets_M3)
    
    # M4 contributes to C00 and C10
    # C00 is at rows 0-1, cols 0-1
    # C10 is at rows 2-3, cols 0-1
    w_offsets_M4 = [
        [(0, 0), (0, 1)],
        [(2, 0), (2, 1)]
    ]
    add_strassen_2x2_from_forms(U4, V4, w_offsets_M4)
    
    # M5 contributes to C00 and C01
    # C00 is at rows 0-1, cols 0-1
    # C01 is at rows 0-1, cols 2-3
    w_offsets_M5 = [
        [(0, 0), (0, 1)],
        [(0, 2), (0, 3)]
    ]
    add_strassen_2x2_from_forms(U5, V5, w_offsets_M5)
    
    # M6 contributes to C11
    w_offsets_M6 = [
        [(2, 2), (2, 3)],
        [(2, 2), (2, 3)]
    ]
    add_strassen_2x2_from_forms(U6, V6, w_offsets_M6)
    
    # M7 contributes to C00
    w_offsets_M7 = [
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1)]
    ]
    add_strassen_2x2_from_forms(U7, V7, w_offsets_M7)
    
    # Now for C2 = A * B2 (4x4 * 4x1)
    # B2 is the last column of B, i.e., B[:, 4]
    # We use recursive Strassen-style decomposition.
    # Split A into 2x2 blocks, B2 into 2x1 blocks.
    # C2 = [[C00, C01], [C10, C11]] where each Cij is 2x1.
    
    # For each 2x2x1 block, we use naive_2x2x1 (rank 4)
    # Total: 4 blocks * 4 = 16 terms
    
    # C00: A00 * B2_0
    naive_2x2x1(0, 0, 0, 4)
    # C01: A01 * B2_0
    naive_2x2x1(0, 2, 0, 4)
    # C10: A10 * B2_1
    naive_2x2x1(2, 0, 2, 4)
    # C11: A11 * B2_1
    naive_2x2x1(2, 2, 2, 4)
    
    # Convert lists of columns to matrices
    U = np.column_stack(U_cols) if U_cols else np.zeros((n*m, 0))
    V = np.column_stack(V_cols) if V_cols else np.zeros((m*p, 0))
    W = np.column_stack(W_cols) if W_cols else np.zeros((p*n, 0))
    
    return {"U": U, "V": V, "W": W}
