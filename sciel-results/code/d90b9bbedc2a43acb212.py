import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 8
    dim_v = m * p  # 28
    dim_w = p * n  # 14
    rank_school = n * m * p  # 56

    # Build the multiplication tensor T as a flat vector
    # T[i*m+j, j*p+k, k*n+i] = 1
    T_flat = np.zeros(dim_u * dim_v * dim_w)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_flat[idx_u * dim_v * dim_w + idx_v * dim_w + idx_w] = 1.0

    # Schoolbook solution as fallback
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify_decomposition(U, V, W, T_flat, atol=1e-10):
        T_recon = np.einsum('it,jt,kt->ijk', U, V, W).flatten()
        return np.allclose(T_recon, T_flat, atol=atol)

    # Strassen-based decomposition for 2x4 * 4x7
    # Split m=4 into two blocks of size 2: j = j1*2 + j2, j1,j2 in {0,1}
    # A (2x4) -> blocks A[:, 0:2] (2x2) and A[:, 2:4] (2x2)
    # B (4x7) -> blocks B[0:2, :] (2x7) and B[2:4, :] (2x7)
    # 
    # For each k-block of B (size 2x2 in the j-dimension, 7 in k),
    # we use Strassen on the 2x2 * 2x2 part, but B is 2x7 not 2x2.
    # 
    # Better approach: treat the problem as 2x(2+2) * (2+2)x7
    # Split into 4 sub-products:
    #   A[:,0:2] * B[0:2,:]  -> 2x2 * 2x7
    #   A[:,0:2] * B[2:4,:]  -> 2x2 * 2x7
    #   A[:,2:4] * B[0:2,:]  -> 2x2 * 2x7
    #   A[:,2:4] * B[2:4,:]  -> 2x2 * 2x7
    #
    # Each 2x2 * 2x7 product can be done with 4 schoolbook multiplications
    # (each 2x1 * 1x7 gives 2 scalar mults... no, 2x2 * 2x7 = 8 scalar mults schoolbook)
    #
    # Alternative: Use Strassen on the 2x2 x 2x2 blocks within each 2x7 column.
    # For a fixed pair of columns (k, k'), the 2x2 * 2x2 sub-product uses 7 mults (Strassen).
    # But we have 7 columns, so we need to handle them separately.
    #
    # Let's use a cleaner approach: 
    # The full product C = A*B where A is 2x4, B is 4x7.
    # Write A = [A0 A1] where A0, A1 are 2x2.
    # Write B = [B0; B1] where B0, B1 are 2x7.
    # C = A0*B0 + A1*B1
    #
    # Each A_i * B_i is 2x2 * 2x7.
    # For 2x2 * 2x7, we can split the 7 columns into groups.
    # Actually, let's just use the structure more directly.
    #
    # Key insight: A0*B0 + A1*B1 where each is 2x2 * 2x7.
    # A 2x2 * 2x7 product: for each column k, we compute (2x2)*(2x1) = 2x1.
    # That's 4 scalar mults per column, 7 columns = 28 scalar mults for one A_i*B_i.
    # Total: 56 scalar mults (schoolbook).
    #
    # Strassen on the 2x2 x 2x2 blocks:
    # For columns k=0..6, we can pair them or handle individually.
    # Let's split 7 columns as: one 2x2 block (cols 0,1) + one 2x2 block (cols 2,3) 
    #   + one 2x2 block (cols 4,5) + one 1x1 (col 6)
    # Wait, that doesn't work cleanly because Strassen needs square blocks.
    #
    # Better: Handle each column independently but share Strassen-like combinations.
    #
    # Actually, the cleanest approach: 
    # C = A0*B0 + A1*B1
    # Use Strassen for the "outer" 2x2 structure:
    # Define:
    #   M1 = (A0 + A1) * (B0 + B1)  -- but this is 2x2 * 2x7, not square
    #
    # Let me think differently. The standard Strassen applies to 2x2 * 2x2.
    # Here we have 2x4 * 4x7. Let's partition differently.
    #
    # Partition A into left/right 2x2 blocks: A = [AL AR]
    # Partition B into top/bottom 2x7 blocks: B = [BT; BB]
    # C = AL*BT + AR*BB
    #
    # For AL*BT (2x2 * 2x7): split BT into 7 columns, each 2x1.
    # AL * BT[:,k] for each k. That's 7 separate 2x2 * 2x1 products.
    # Each 2x2 * 2x1 = 4 scalar mults. Total 28.
    #
    # To use Strassen, we need to combine the 2x1 vectors.
    # Split the 7 columns into: cols 0-3 (two 2x2 blocks) + cols 4-6 (one 2x2 + one 2x1)
    # Hmm, 7 is odd.
    #
    # Let's try: split 7 columns as 2+2+2+1.
    # For the 2x2 column blocks, use Strassen (7 mults each for 2x2*2x2).
    # For the single column, use 4 mults.
    #
    # But Strassen's 7 mults are for computing all 4 entries of the 2x2 result.
    # For 2x2 * 2x2, Strassen uses 7 scalar mults.
    # For 2x2 * 2x1, we need 4 scalar mults (can't reduce below 4 for this).
    #
    # So for AL*BT: split BT into [BT0 BT1] where BT0, BT1 are 2x2 (cols 0-3) 
    #   and BT2 is 2x1 (cols 4-6... wait, that's 3 cols).
    # 7 = 2 + 2 + 3. For the 3-col part, split as 2+1.
    # 
    # This is getting complicated. Let me just directly construct a rank-50 decomposition.
    #
    # RANK 50 APPROACH:
    # C = A0*B0 + A1*B1 where A0,A1 are 2x2, B0,B1 are 2x7.
    # 
    # For A0*B0: Use 28 scalar multiplications (schoolbook for 2x2 * 2x7).
    # For A1*B1: Use 28 scalar multiplications.
    # Total: 56.
    #
    # To reduce: notice that A0*B0 + A1*B1 can be reorganized.
    # Let's define combined matrices and use Strassen-style:
    #   S1 = (A0 + A1) * (B0 + B1)  [2x2 * 2x7, needs 28 mults]
    #   S2 = A0 * B0  [2x2 * 2x7, needs 28 mults]  
    #   S3 = A1 * B1  [2x2 * 2x7, needs 28 mults]
    #   C = S1 - S2 - S3 + 2*A0*B0 + 2*A1*B1... no this doesn't help.
    #
    # Standard Strassen for 2x2*2x2:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #   C11 = M1+M4-M5+M7
    #   C12 = M3+M5
    #   C21 = M2+M4
    #   C22 = M1-M2+M3+M6
    #
    # For our case, the "B" side is 2x7, not 2x2.
    # We can apply Strassen column-by-column, but that gives no savings.
    #
    # KEY INSIGHT: We can split the 7 columns of B into 3+4 or 2+5 etc.
    # and apply Strassen to the square sub-blocks.
    #
    # Let's split B's 7 columns as: first 4 columns (forming 2x4) and last 3 (2x3).
    # For the 2x4 part: split into two 2x2 blocks, apply Strassen to each: 7+7=14 mults.
    # For the 2x3 part: split into 2x2 + 2x1: 7 + 4 = 11 mults.
    # Total for one A_i*B_i: 14 + 11 = 25 mults.
    # For two (A0*B0 + A1*B1): 50 mults.
    #
    # This gives rank 50! Let's construct it.
    #
    # A = [A0 A1] (2x4), B = [B0; B1] (4x7)
    # B0 is 2x7, split as B0 = [B0a B0b] where B0a is 2x4, B0b is 2x3.
    # B0a = [B0a1 B0a2] where B0a1, B0a2 are 2x2.
    # B0b = [B0b1 B0b2] where B0b1 is 2x2, B0b2 is 2x1.
    #
    # Similarly for B1.
    #
    # For A0 * B0:
    #   A0 * B0a1: Strassen (7 mults) -> contributes to C[:, 0:2]
    #   A0 * B0a2: Strassen (7 mults) -> contributes to C[:, 2:4]
    #   A0 * B0b1: Strassen (7 mults) -> contributes to C[:, 4:6]
    #   A0 * B0b2: schoolbook 2x2*2x1 (4 mults) -> contributes to C[:, 6]
    #
    # For A1 * B1:
    #   Same structure, 25 mults.
    #
    # Total: 50 mults.
    #
    # Now let's explicitly write out the Strassen formulas.
    #
    # For 2x2 * 2x2 with A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(g-h)
    #   M4 = d(f-e)
    #   M5 = (a+b)h
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    #   Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    #
    # For 2x2 * 2x1 with A = [[a,b],[c,d]], B = [e; g]:
    #   Result: [a*e + b*g; c*e + d*g] (4 scalar mults)

    rank_strassen = 50
    U_str = np.zeros((dim_u, rank_strassen))
    V_str = np.zeros((dim_v, rank_strassen))
    W_str = np.zeros((dim_w, rank_strassen))
    
    # Indexing:
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    #   j=0,1 -> A0 (first 2x2 block), j=2,3 -> A1 (second 2x2 block)
    #   A0 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    #   A1 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    #
    # B is 4x7: B[j, k] where j in {0..3}, k in {0..6}
    #   j=0,1 -> B0 (first 2x7 block), j=2,3 -> B1 (second 2x7 block)
    #   B0 = [[B[0,0],...,B[0,6]], [B[1,0],...,B[1,6]]]
    #   B1 = [[B[2,0],...,B[2,6]], [B[3,0],...,B[3,6]]]
    #
    # B0 split: cols 0-1 (B0a1), cols 2-3 (B0a2), cols 4-5 (B0b1), col 6 (B0b2)
    # B1 split: same column structure.
    #
    # C is 2x7: C[i, k] where i in {0,1}, k in {0..6}
    #
    # Tensor: T[i*4+j, j*7+k, k*2+i] = 1
    # U index: i*4+j (8 values)
    # V index: j*7+k (28 values)
    # W index: k*2+i (14 values)
    
    def add_strassen_2x2_2x2(A_block, B_block_cols, C_cols, offset):
        """
        A_block: (2,2) matrix of (u_idx, coeff) for the 2x2 A sub-block
                 A_block[r][c] = (u_index, coefficient) for A element at row r, col c
        B_block_cols: list of 4 (v_idx, coeff) for B's 2x2 sub-block
                      B_block_cols[r*2+c] = (v_index, coefficient)
        C_cols: list of 4 (w_idx, coeff) for C's 2x2 sub-block
                C_cols[r*2+c] = (w_index, coefficient)
        offset: column offset in U,V,W arrays
        """
        a, b = A_block[0][0], A_block[0][1]  # top row
        c, d = A_block[1][0], A_block[1][1]  # bottom row
        e, f = B_block_cols[0], B_block_cols[1]  # top row of B
        g, h = B_block_cols[2], B_block_cols[3]  # bottom row of B
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        mults = [
            (a, d, e, h, 1),   # M1 = (a+d)(e+h) -> need to expand
            (c, d, e, None, 1),# M2 = (c+d)*e
            (a, None, g, h, 1),# M3 = a*(g-h)
            (d, None, f, e, 1),# M4 = d*(f-e)
            (a, b, None, h, 1),# M5 = (a+b)*h
            (c, a, e, f, 1),   # M6 = (c-a)*(e+f)
            (b, d, g, h, 1),   # M7 = (b-d)*(g+h)
        ]
        
        # This is getting complex. Let me just directly enumerate all 50 terms.
        pass
    
    # Let me take a more direct approach and enumerate all 50 rank-1 terms.
    # Each term is (u_vec, v_vec, w_vec) where:
    #   u_vec has nonzeros at positions corresponding to A entries
    #   v_vec has nonzeros at positions corresponding to B entries  
    #   w_vec has nonzeros at positions corresponding to C entries
    #
    # The tensor equation is: sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = T[i,j,k]
    #
    # For Strassen, each "multiplication" Mx corresponds to a product of linear combos.
    # We need to expand each Strassen multiplication into rank-1 terms.
    
    # Let me just directly build the rank-50 decomposition by explicitly writing
    # out each of the 50 scalar multiplications.
    
    # For A0 * B0a1 (2x2 * 2x2, Strassen):
    # A0 = [[A00, A01], [A10, A11]] where Aij = A[i,j] for i,j in {0,1}
    # B0a1 = [[B00, B01], [B10, B11]] where Bij = B[i,k] for i in {0,1}, k in {0,1}
    #
    # Strassen mults:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B10-B11)
    # M4 = A11 * (B01-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    #
    # C00 = M1+M4-M5+M7  (contributes to C[0,0])
    # C01 = M3+M5         (contributes to C[0,1])
    # C10 = M2+M4         (contributes to C[1,0])
    # C11 = M1-M2+M3+M6   (contributes to C[1,1])
    
    # Each Mx is a product of a linear combo of A entries and a linear combo of B entries.
    # To make it rank-1 in the tensor sense, we need each "multiplication" to be
    # a single scalar product. But Strassen's Mx are already scalar products of
    # linear combinations. So each Mx contributes ONE rank-1 term to the tensor,
    # where the u-vector encodes the A coefficients, v-vector encodes B coefficients,
    # and w-vector encodes which C entries get this Mx with what sign.
    
    # For M1 = (A00+A11)*(B00+B11):
    #   u has +1 at (i=0,j=0) and +1 at (i=1,j=1)
    #   v has +1 at (j=0,k=0) and +1 at (j=1,k=1)
    #   w: M1 contributes +1 to C00 (i=0,k=0), +1 to C11 (i=1,k=1)
    #   So w has +1 at (k=0,i=0) and +1 at (k=1,i=1)
    
    # This works! Each Strassen multiplication is one rank-1 term.
    
    # Let me enumerate all 50 terms systematically.
    
    # Helper to create a rank-1 term
    def make_term(u_entries, v_entries, w_entries, t):
        """
        u_entries: list of (u_idx, coeff)
        v_entries: list of (v_idx, coeff)
        w_entries: list of (w_idx, coeff)
        t: column index
        """
        for (idx, coeff) in u_entries:
            U_str[idx, t] = coeff
        for (idx, coeff) in v_entries:
            V_str[idx, t] = coeff
        for (idx, coeff) in w_entries:
            W_str[idx, t] = coeff
    
    # A indices: u_idx = i*4 + j
    # B indices: v_idx = j*7 + k
    # C indices: w_idx = k*2 + i
    
    # A0 block: i in {0,1}, j in {0,1}
    # A1 block: i in {0,1}, j in {2,3}
    # B0 block: j in {0,1}, k in {0..6}
    # B1 block: j in {2,3}, k in {0..6}
    
    # For A0 * B0a1 (Strassen, 7 terms):
    # A0 entries: A[0,0]=u(0), A[0,1]=u(1), A[1,0]=u(4), A[1,1]=u(5)
    # B0a1 entries: B[0,0]=v(0), B[0,1]=v(1), B[1,0]=v(7), B[1,1]=v(8)
    # C entries: C[0,0]=w(0), C[0,1]=w(2), C[1,0]=w(1), C[1,1]=w(3)
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    make_term(
        [(0, 1), (5, 1)],      # A[0,0] + A[1,1]
        [(0, 1), (8, 1)],      # B[0,0] + B[1,1]
        [(0, 1), (3, 1)],      # C[0,0] + C[1,1]
        0
    )
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    make_term(
        [(4, 1), (5, 1)],      # A[1,0] + A[1,1]
        [(0, 1)],              # B[0,0]
        [(1, 1)],              # C[1,0]
        1
    )
    
    # M3 = A[0,0] * (B[1,0]-B[1,1])
    make_term(
        [(0, 1)],              # A[0,0]
        [(7, 1), (8, -1)],    # B[1,0] - B[1,1]
        [(2, 1)],              # C[0,1]
        2
    )
    
    # M4 = A[1,1] * (B[0,1]-B[0,0])
    make_term(
        [(5, 1)],              # A[1,1]
        [(1, 1), (0, -1)],    # B[0,1] - B[0,0]
        [(0, 1), (1, 1)],     # C[0,0] + C[1,0]
        3
    )
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    make_term(
        [(0, 1), (1, 1)],     # A[0,0] + A[0,1]
        [(8, 1)],              # B[1,1]
        [(0, -1), (2, 1)],    # -C[0,0] + C[0,1]
        4
    )
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    make_term(
        [(4, 1), (0, -1)],    # A[1,0] - A[0,0]
        [(0, 1), (1, 1)],     # B[0,0] + B[0,1]
        [(3, 1)],              # C[1,1]
        5
    )
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    make_term(
        [(1, 1), (5, -1)],    # A[0,1] - A[1,1]
        [(7, 1), (8, 1)],     # B[1,0] + B[1,1]
        [(0, 1)],              # C[0,0]
        6
    )
    
    # For A0 * B0a2 (Strassen, 7 terms): cols k=2,3
    # B0a2 entries: B[0,2]=v(2), B[0,3]=v(3), B[1,2]=v(9), B[1,3]=v(10)
    # C entries: C[0,2]=w(4), C[0,3]=w(6), C[1,2]=w(5), C[1,3]=w(7)
    
    make_term(
        [(0, 1), (5, 1)],
        [(2, 1), (10, 1)],
        [(4, 1), (7, 1)],
        7
    )
    
    make_term(
        [(4, 1), (5, 1)],
        [(2, 1)],
        [(5, 1)],
        8
    )
    
    make_term(
        [(0, 1)],
        [(9, 1), (10, -1)],
        [(6, 1)],
        9
    )
    
    make_term(
        [(5, 1)],
        [(3, 1), (2, -1)],
        [(4, 1), (5, 1)],
        10
    )
    
    make_term(
        [(0, 1), (1, 1)],
        [(10, 1)],
        [(4, -1), (6, 1)],
        11
    )
    
    make_term(
        [(4, 1), (0, -1)],
        [(2, 1), (3, 1)],
        [(7, 1)],
        12
    )
    
    make_term(
        [(1, 1), (5, -1)],
        [(9, 1), (10, 1)],
        [(4, 1)],
        13
    )
    
    # For A0 * B0b1 (Strassen, 7 terms): cols k=4,5
    # B0b1 entries: B[0,4]=v(4), B[0,5]=v(5), B[1,4]=v(11), B[1,5]=v(12)
    # C entries: C[0,4]=w(8), C[0,5]=w(10), C[1,4]=w(9), C[1,5]=w(11)
    
    make_term(
        [(0, 1), (5, 1)],
        [(4, 1), (12, 1)],
        [(8, 1), (11, 1)],
        14
    )
    
    make_term(
        [(4, 1), (5, 1)],
        [(4, 1)],
        [(9, 1)],
        15
    )
    
    make_term(
        [(0, 1)],
        [(11, 1), (12, -1)],
        [(10, 1)],
        16
    )
    
    make_term(
        [(5, 1)],
        [(5, 1), (4, -1)],
        [(8, 1), (9, 1)],
        17
    )
    
    make_term(
        [(0, 1), (1, 1)],
        [(12, 1)],
        [(8, -1), (10, 1)],
        18
    )
    
    make_term(
        [(4, 1), (0, -1)],
        [(4, 1), (5, 1)],
        [(11, 1)],
        19
    )
    
    make_term(
        [(1, 1), (5, -1)],
        [(11, 1), (12, 1)],
        [(8, 1)],
        20
    )
    
    # For A0 * B0b2 (schoolbook 2x2*2x1, 4 terms): col k=6
    # B0b2 entries: B[0,6]=v(6), B[1,6]=v(13)
    # C entries: C[0,6]=w(12), C[1,6]=w(13)
    
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6]
    make_term(
        [(0, 1)],
        [(6, 1)],
        [(12, 1)],
        21
    )
    make_term(
        [(1, 1)],
        [(13, 1)],
        [(12, 1)],
        22
    )
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6]
    make_term(
        [(4, 1)],
        [(6, 1)],
        [(13, 1)],
        23
    )
    make_term(
        [(5, 1)],
        [(13, 1)],
        [(13, 1)],
        24
    )
    
    # Now for A1 * B1 (same structure, but A1 uses j=2,3 and B1 uses j=2,3)
    # A1 entries: A[0,2]=u(2), A[0,3]=u(3), A[1,2]=u(6), A[1,3]=u(7)
    # B1 entries: B[2,k]=v(14+k), B[3,k]=v(21+k) for k=0..6
    
    # For A1 * B1a1 (Strassen, 7 terms): cols k=0,1
    # B1a1: B[2,0]=v(14), B[2,1]=v(15), B[3,0]=v(21), B[3,1]=v(22)
    
    make_term(
        [(2, 1), (7, 1)],
        [(14, 1), (22, 1)],
        [(0, 1), (3, 1)],
        25
    )
    
    make_term(
        [(6, 1), (7, 1)],
        [(14, 1)],
        [(1, 1)],
        26
    )
    
    make_term(
        [(2, 1)],
        [(21, 1), (22, -1)],
        [(2, 1)],
        27
    )
    
    make_term(
        [(7, 1)],
        [(15, 1), (14, -1)],
        [(0, 1), (1, 1)],
        28
    )
    
    make_term(
        [(2, 1), (3, 1)],
        [(22, 1)],
        [(0, -1), (2, 1)],
        29
    )
    
    make_term(
        [(6, 1), (2, -1)],
        [(14, 1), (15, 1)],
        [(3, 1)],
        30
    )
    
    make_term(
        [(3, 1), (7, -1)],
        [(21, 1), (22, 1)],
        [(0, 1)],
        31
    )
    
    # For A1 * B1a2 (Strassen, 7 terms): cols k=2,3
    # B1a2: B[2,2]=v(16), B[2,3]=v(17), B[3,2]=v(23), B[3,3]=v(24)
    
    make_term(
        [(2, 1), (7, 1)],
        [(16, 1), (24, 1)],
        [(4, 1), (7, 1)],
        32
    )
    
    make_term(
        [(6, 1), (7, 1)],
        [(16, 1)],
        [(5, 1)],
        33
    )
    
    make_term(
        [(2, 1)],
        [(23, 1), (24, -1)],
        [(6, 1)],
        34
    )
    
    make_term(
        [(7, 1)],
        [(17, 1), (16, -1)],
        [(4, 1), (5, 1)],
        35
    )
    
    make_term(
        [(2, 1), (3, 1)],
        [(24, 1)],
        [(4, -1), (6, 1)],
        36
    )
    
    make_term(
        [(6, 1), (2, -1)],
        [(16, 1), (17, 1)],
        [(7, 1)],
        37
    )
    
    make_term(
        [(3, 1), (7, -1)],
        [(23, 1), (24, 1)],
        [(4, 1)],
        38
    )
    
    # For A1 * B1b1 (Strassen, 7 terms): cols k=4,5
    # B1b1: B[2,4]=v(18), B[2,5]=v(19), B[3,4]=v(25), B[3,5]=v(26)
    
    make_term(
        [(2, 1), (7, 1)],
        [(18, 1), (26, 1)],
        [(8, 1), (11, 1)],
        39
    )
    
    make_term(
        [(6, 1), (7, 1)],
        [(18, 1)],
        [(9, 1)],
        40
    )
    
    make_term(
        [(2, 1)],
        [(25, 1), (26, -1)],
        [(10, 1)],
        41
    )
    
    make_term(
        [(7, 1)],
        [(19, 1), (18, -1)],
        [(8, 1), (9, 1)],
        42
    )
    
    make_term(
        [(2, 1), (3, 1)],
        [(26, 1)],
        [(8, -1), (10, 1)],
        43
    )
    
    make_term(
        [(6, 1), (2, -1)],
        [(18, 1), (19, 1)],
        [(11, 1)],
        44
    )
    
    make_term(
        [(3, 1), (7, -1)],
        [(25, 1), (26, 1)],
        [(8, 1)],
        45
    )
    
    # For A1 * B1b2 (schoolbook 2x2*2x1, 4 terms): col k=6
    # B1b2: B[2,6]=v(20), B[3,6]=v(27)
    
    make_term(
        [(2, 1)],
        [(20, 1)],
        [(12, 1)],
        46
    )
    make_term(
        [(3, 1)],
        [(27, 1)],
        [(12, 1)],
        47
    )
    make_term(
        [(6, 1)],
        [(20, 1)],
        [(13, 1)],
        48
    )
    make_term(
        [(7, 1)],
        [(27, 1)],
        [(13, 1)],
        49
    )
    
    # Verify the Strassen decomposition
    if verify_decomposition(U_str, V_str, W_str, T_flat):
        return {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
    
    # If verification failed, fall back to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
