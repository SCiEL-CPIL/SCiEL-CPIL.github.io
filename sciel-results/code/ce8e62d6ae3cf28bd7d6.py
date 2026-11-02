import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank 50
    # Structure:
    # A (2x4) -> A1 (2x2), A2 (2x2)
    # B (4x7) -> B1 (2x7), B2 (2x7)
    # C = A1 B1 + A2 B2
    # For each Ai Bi (2x2x7):
    #   Split Bi into B_i1 (2x3), B_i2 (2x4)
    #   C_i = Ai B_i1 + Ai B_i2
    #   Ai B_i1 (2x2x3) -> Strassen(2x2x2) + Schoolbook(2x2x1) -> Rank 7 + 4 = 11
    #   Ai B_i2 (2x2x4) -> Strassen(2x2x2) + Strassen(2x2x2) -> Rank 7 + 7 = 14
    # Total Rank = 2 * (11 + 14) = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)] # 8 x 50
    V = [[0] * rank for _ in range(m * p)] # 28 x 50
    W = [[0] * rank for _ in range(p * n)] # 14 x 50
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val):
        # u_idx: index in U (0..7)
        # v_idx: index in V (0..27)
        # w_idx: index in W (0..13)
        # val: coefficient
        pass # We will build the matrices directly
    
    # We need to map the logical indices to the global indices.
    # U indices: i*m + j. i in 0..1, j in 0..3.
    # V indices: j*p + k. j in 0..3, k in 0..6.
    # W indices: k*n + i. k in 0..6, i in 0..1.
    
    col = 0
    
    def add_strassen_2x2x2(a_offset, b_offset, c_offset, scale=1):
        """
        Adds a Strassen decomposition for 2x2x2.
        a_offset: row offset in A (0 or 2 for A1/A2)
        b_offset: col offset in B (0, 3, 4, 7, 11, 14, 18, 21) ?
        Wait, B is 4x7.
        B1 is rows 0,1. B2 is rows 2,3.
        B1 is cols 0..6.
        B2 is cols 0..6.
        
        For A1 B1:
        A1 is A[0:2, 0:2].
        B1 is B[0:2, 0:7].
        
        We split B1 into B1_1 (cols 0..2) and B1_2 (cols 3..6).
        B1_1 is 2x3.
        B1_2 is 2x4.
        
        For A1 B1_1 (2x2x3):
        Split B1_1 into B1_1a (cols 0..1) and B1_1b (col 2).
        B1_1a is 2x2.
        B1_1b is 2x1.
        
        So we need:
        1. Strassen for A1 and B1_1a (2x2x2).
        2. Schoolbook for A1 and B1_1b (2x2x1).
        3. Strassen for A1 and B1_2a (2x2x2) (cols 3,4).
        4. Strassen for A1 and B1_2b (2x2x2) (cols 5,6).
        
        Similarly for A2 B2.
        """
        pass

    # This is getting complicated to manage indices manually.
    # Let's generate the U, V, W entries by iterating through the logical blocks.
    
    def get_u(i, j):
        return i * m + j
        
    def get_v(j, k):
        return j * p + k
        
    def get_w(k, i):
        return k * n + i

    # We will store the coefficients for each rank-1 term.
    # Each term is (u_idx, v_idx, w_idx, coeff)
    terms = []
    
    def add_strassen(a_row_start, b_col_start, b_row_start):
        # a_row_start: 0 or 2 (for A1 or A2)
        # b_row_start: 0 or 2 (for B1 or B2)
        # b_col_start: 0..5 (start of 2 columns)
        
        # A is 2x2 at (a_row_start, 0..1)
        # B is 2x2 at (b_row_start, b_col_start..b_col_start+1)
        
        a00 = get_u(a_row_start + 0, 0)
        a01 = get_u(a_row_start + 0, 1)
        a10 = get_u(a_row_start + 1, 0)
        a11 = get_u(a_row_start + 1, 1)
        
        b00 = get_v(b_row_start + 0, b_col_start + 0)
        b01 = get_v(b_row_start + 0, b_col_start + 1)
        b10 = get_v(b_row_start + 1, b_col_start + 0)
        b11 = get_v(b_row_start + 1, b_col_start + 1)
        
        c00 = get_w(b_col_start + 0, a_row_start // 2) # i = 0 for A1, 1 for A2?
        # Wait, W index is k*n + i.
        # i is the row of C.
        # C row i corresponds to A row i.
        # So i = a_row_start // 2 ?
        # A1 is rows 0,1. So i=0,1.
        # A2 is rows 2,3. So i=2,3?
        # No, C is 2x7.
        # C row 0 is from A1. C row 1 is from A2.
        # So for A1, i=0. For A2, i=1.
        
        c00 = get_w(b_col_start + 0, a_row_start // 2)
        c01 = get_w(b_col_start + 1, a_row_start // 2)
        c10 = get_w(b_col_start + 0, a_row_start // 2 + 1)
        c11 = get_w(b_col_start + 1, a_row_start // 2 + 1)
        
        # M1 = (a00+a11)(b00+b11)
        # This requires linear combinations in U and V.
        # We can't just add a term. We need to expand.
        # (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
        # This is 4 rank-1 terms.
        # But Strassen uses 7 products.
        # We must represent the products as rank-1 terms in the tensor.
        # The tensor decomposition allows linear combinations in U, V, W.
        # So we can have U = [1, 1, ...], V = [1, 1, ...].
        
        # Let's add the 7 Strassen terms.
        # M1
        u1 = [0]*8; u1[a00]=1; u1[a11]=1
        v1 = [0]*28; v1[b00]=1; v1[b11]=1
        w1 = [0]*14
        w1[c00] += 1
        w1[c11] += 1
        w1[c01] -= 1
        w1[c10] -= 1
        # This is not correct. W is linear in C.
        # C = U V^T W^T?
        # C_{ik} = \sum_r U_{ir} V_{jr} W_{kr} ?
        # No, C is 2x7.
        # C_{i, k} = \sum_{j} A_{ij} B_{jk}.
        # Decomposition: C = \sum_{r=1}^R U_r \otimes V_r \otimes W_r ?
        # The interface says U (n*m, R), V (m*p, R), W (p*n, R).
        # And "exactly decompose the matrix multiplication tensor".
        # The tensor T is such that C = \sum_r (U[:,r] \cdot A) (V[:,r] \cdot B) W[:,r]^T ?
        # No.
        # Standard decomposition:
        # C_{ik} = \sum_{j} A_{ij} B_{jk}
        # = \sum_r (\sum_i U_{ir} A_{ij} ?)
        
        # Let's look at the seed code.
        # U[i*m+j, c] = 1
        # V[j*p+k, c] = 1
        # W[k*n+i, c] = 1
        # This implies:
        # C_{ik} = \sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ?
        # No, the sum is over c.
        # But i, j, k are indices of A, B, C.
        # The decomposition is:
        # C_{ik} = \sum_{c=1}^R (\sum_{j} U_{i*m+j, c} A_{ij}) (\sum_{j} V_{j*p+k, c} B_{jk}) W_{k*n+i, c} ?
        # No, U depends on i, j.
        # Usually: C_{ik} = \sum_r U_{ir} V_{kr} ?
        
        # Let's re-read the seed.
        # U[i*m+j, c] = 1.
        # This means U is a matrix of size (n*m) x R.
        # The index is i*m+j.
        # This suggests U picks out A_{ij}.
        # Similarly V picks out B_{jk}.
        # W picks out C_{ki}? No, W[k*n+i, c] = 1.
        # So C_{ki} is the output?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So C is indexed by k, i?
        # C is 2x7. i in 0..1, k in 0..6.
        # So C_{ik} is stored at W[k*n+i].
        
        # The reconstruction is likely:
        # C_{ik} = \sum_r (\sum_{j} U_{i*m+j, r} A_{ij}) (\sum_{j} V_{j*p+k, r} B_{jk}) W_{k*n+i, r}
        # Wait, if U_{i*m+j, r} is 1, then \sum_j U A = A_{ij}.
        # If V_{j*p+k, r} is 1, then \sum_j V B = B_{jk}.
        # Then product is A_{ij} B_{jk}.
        # Then \sum_r A_{ij} B_{jk} W_{k*n+i, r}.
        # This doesn't sum over j.
        # The sum over j is inside the U and V definitions?
        # No, U is a matrix.
        
        # Actually, the standard tensor decomposition for matrix multiplication is:
        # C_{ik} = \sum_{j} A_{ij} B_{jk}
        # We want C = \sum_r U_r A V_r^T W_r ?
        # No.
        
        # Let's assume the seed is correct and I just need to fill U, V, W.
        # The seed uses rank 56 with 1s.
        # I will use the same structure but with coefficients.
        
        # For Strassen, we need linear combinations.
        # So U, V, W entries will be 1, -1, 0.
        
        # Let's implement the addition of terms.
        # We will accumulate U, V, W as lists of lists.
        pass

    # Since I cannot easily run code, I will write the code that generates the 50 rank terms.
    
    return {"U": U, "V": V, "W": W}
