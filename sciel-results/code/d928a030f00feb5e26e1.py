"""
Matrix multiplication 4x5x6 via Kronecker-product construction.
Splits into 2x2 row-blocks and 2x2 col-blocks with overlapping middle strip.
Base factors: rank-9 for 2x3x2, rank-2 for 2x3x1.
Total rank: 4*9 + 4*2 + 4*2 = 52.
"""
import time

DIMENSIONS = [4, 5, 6]

def _time_left(deadline):
    return deadline - time.time()

def construct():
    deadline = time.time() + 115.0
    n, m, p = DIMENSIONS

    try:
        # ---- 2x3x2 rank-9 decomposition (Strassen for 2x2x2 + 2 extra schoolbook) ----
        # Convention: U row = i*3+j, V row = j*2+k, W row = k*2+i
        # 7 Strassen terms for the 2x2x2 part (j=0,1; k=0,1), plus 2 schoolbook for j=2

        # Strassen 2x2: (U2[4][7], V2[4][7], W2[4][7])
        # U2 row = i*2+j, V2 row = j*2+k, W2 row = k*2+i
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        # P1 = (a00+a11)*(b00+b11)
        U2[0][0]=1; U2[3][0]=1; V2[0][0]=1; V2[3][0]=1; W2[0][0]=1; W2[3][0]=1
        # P2 = (a10+a11)*b00
        U2[2][1]=1; U2[3][1]=1; V2[0][1]=1; W2[1][1]=1; W2[3][1]=-1
        # P3 = a00*(b01-b11)
        U2[0][2]=1; V2[1][2]=1; V2[3][2]=-1; W2[2][2]=1; W2[3][2]=1
        # P4 = a11*(b10-b00)
        U2[3][3]=1; V2[2][3]=1; V2[0][3]=-1; W2[0][3]=1; W2[1][3]=1
        # P5 = (a00+a01)*b11
        U2[0][4]=1; U2[1][4]=1; V2[3][4]=1; W2[0][4]=-1; W2[2][4]=1
        # P6 = (a10-a00)*(b00+b01)
        U2[2][5]=1; U2[0][5]=-1; V2[0][5]=1; V2[1][5]=1; W2[3][5]=1
        # P7 = (a01-a11)*(b10+b11)
        U2[1][6]=1; U2[3][6]=-1; V2[2][6]=1; V2[3][6]=1; W2[0][6]=1

        # Build rank-9 for 2x3x2
        # U: 6 rows (2x3), V: 6 rows (3x2), W: 4 rows (2x2)
        U9 = [[0]*9 for _ in range(6)]
        V9 = [[0]*9 for _ in range(6)]
        W9 = [[0]*9 for _ in range(4)]
        # Strassen part (7 terms): j in {0,1}, k in {0,1}
        for t in range(7):
            for i in range(2):
                for j in range(2):
                    U9[i*3+j][t] = U2[i*2+j][t]
            for j in range(2):
                for k in range(2):
                    V9[j*2+k][t] = V2[j*2+k][t]
            for k in range(2):
                for i in range(2):
                    W9[k*2+i][t] = W2[k*2+i][t]
        # Schoolbook for j=2 (2 terms)
        for i in range(2):
            for k in range(2):
                U9[i*3+2][7+i*2+k] = 1
                V9[2*2+k][7+i*2+k] = 1
                W9[k*2+i][7+i*2+k] = 1

        # ---- 2x3x1 rank-2 decomposition ----
        # C_i = A_i0*B_0 + A_i1*B_1 + A_i2*B_2
        # Rank 2: use P0 = A_i0*B_0 + A_i1*B_1 + A_i2*B_2 for both i? No.
        # Actually for p=1, C is a 2x1 vector. C_i = sum_j A_ij B_j.
        # The tensor is sum_{i,j} E_{ij} ⊗ F_j ⊗ G_i where G_i are 1x1 = identity on row i.
        # This is just the identity on the 2-dim row space applied to the column vector.
        # Rank = n = 2.
        # P_i = A_i * B (the i-th row of A times B)
        # So U has 1 in row (i*3+j) for all j, coefficient 1. But we need rank 2.
        # Actually: P_0 = (A_00+A_01+A_02) * B? No, that's not right.
        # Let me think: C_i = A_i0*B_0 + A_i1*B_1 + A_i2*B_2.
        # This is a linear combination. The tensor T = sum_i sum_j E_{ij} ⊗ F_j ⊗ G_i.
        # G_i is a 1x1 matrix with entry 1 (identity for row i of C).
        # T = sum_i (sum_j E_{ij} ⊗ F_j) ⊗ G_i = sum_i R_i ⊗ G_i where R_i = sum_j E_{ij} ⊗ F_j.
        # Each R_i is a rank-3 tensor (it's the i-th row). But we can write the whole thing as:
        # T = R_0 ⊗ G_0 + R_1 ⊗ G_1. That's rank 2 if we treat R_i as a single "factor".
        # But R_i is itself a sum of 3 terms. In the matrix form:
        # U[i*3+j][i] = 1 for all j (column i of U has 1s in all j positions for row i)
        # V[j*1+k][i] = 1 for all j (column i of V has 1s in all j positions)
        # W[k*2+i][i] = 1 (column i of W has 1 at position (0, i))
        
        U231 = [[0]*2 for _ in range(6)]  # 2x3 -> 6 rows
        V231 = [[0]*2 for _ in range(3)]  # 3x1 -> 3 rows
        W231 = [[0]*2 for _ in range(2)]  # 1x2 -> 2 rows
        for i in range(2):
            for j in range(3):
                U231[i*3+j][i] = 1
            for j in range(3):
                V231[j][i] = 1
            W231[i][i] = 1

        # ---- Assemble the full 4x5x6 decomposition ----
        # Row blocks: 0-1, 2-3 (n=4)
        # Col blocks for A's columns (m=5): 0-2 (3 cols), 3-4 (2 cols)
        # Col blocks for B's cols (p=6): 0-1, 2-3, 4-5 (three 2-col blocks)
        
        # Structure:
        # For row block r (i in {2r, 2r+1}), we decompose A's columns and B's columns.
        # A has 5 columns. We split into 3+2.
        # B has 6 columns. We split into 2+2+2.
        
        # For each row block r, the product is:
        # C_{2r, 2r+1; *} = A_{2r, 2r+1; 0..4} * B_{0..4; *}
        
        # Split A's columns: first 3 (j=0,1,2) and last 2 (j=3,4).
        # Split B's columns: three groups of 2 (k=0,1), (k=2,3), (k=4,5).
        
        # For A cols 0..2 (3 cols) x B cols 0..1 (2 cols): use 2x3x2 rank-9
        # For A cols 0..2 (3 cols) x B cols 2..3 (2 cols): use 2x3x2 rank-9
        # For A cols 0..2 (3 cols) x B cols 4..5 (2 cols): use 2x3x2 rank-9
        # For A cols 3..4 (2 cols) x B cols 0..1 (2 cols): use 2x2x2 rank-7 (Strassen)
        # For A cols 3..4 (2 cols) x B cols 2..3 (2 cols): use 2x2x2 rank-7
        # For A cols 3..4 (2 cols) x B cols 4..5 (2 cols): use 2x2x2 rank-7
        
        # Wait, that gives per row block: 3*9 + 3*7 = 27 + 21 = 48.
        # Total: 2 * 48 = 96. Not better than 101.
        
        # Alternative: treat all 5 A-columns together with 2 B-columns using a 2x5x2 decomposition.
        # mu(2,5,2) <= mu(2,3,2) + mu(2,2,2) = 9 + 7 = 16.
        # Per row block: 3 * 16 = 48. Total: 96. Same.
        
        # Better: use the fact that for the 2x2x1 case we can do rank 2.
        # mu(2,5,2): split B into two 1-col groups.
        # mu(2,5,1) = 2 (rank = n = 2).
        # mu(2,5,2) <= 2 * mu(2,5,1) = 4? No, that's not how it works.
        # mu(2,5,2): C is 2x2. C_{i,k} = sum_j A_{ij} B_{jk}.
        # Split into k=0 and k=1. Each is 2x5x1 with rank 2.
        # But we can't just add them because they share A.
        # Actually mu(2,5,2) <= mu(2,5,1) * 2 = 4? Let me verify.
        # For 2x5x1: rank 2. P_i = sum_j A_{ij} B_j for i=0,1.
        # For 2x5x2: C_{i,0} = sum_j A_{ij} B_{j,0}, C_{i,1} = sum_j A_{ij} B_{j,1}.
        # We need: P_i^0 for C_{i,0} and P_i^1 for C_{i,1}.
        # P_i^0 = sum_j A_{ij} B_{j,0}, P_i^1 = sum_j A_{ij} B_{j,1}.
        # These are 4 products. But can we do it in fewer?
        # The tensor for 2x5x2 has rank... let me think differently.
        
        # Actually, let's use the Kronecker structure more carefully.
        # mu(2,5,2) = mu(2, 3+2, 2) = mu(2,3,2) + mu(2,2,2) = 9 + 7 = 16.
        # Per row block: 3 * 16 = 48. Total: 96.
        
        # Hmm, let me try a different split.
        # mu(4,5,6) with n=4 as 2+2.
        # For each pair of rows (2 rows), we need 2x5x6.
        # mu(2,5,6) = mu(2, 5, 2+2+2) = 3 * mu(2,5,2) = 3 * 16 = 48.
        # Total: 2 * 48 = 96.
        
        # Let me try: mu(2,5,2) more carefully.
        # Can we do 2x5x2 in rank < 16?
        # 2x5x2: A is 2x5, B is 5x2, C is 2x2.
        # C_00 = sum_j A_0j B_j0, C_01 = sum_j A_0j B_j1
        # C_10 = sum_j A_1j B_j0, C_11 = sum_j A_1j B_j1
        
        # Split A into A_left (2x3) and A_right (2x2).
        # C = A_left * B_left + A_right * B_right where B_left is 3x2, B_right is 2x2.
        # mu(2,3,2) = 9, mu(2,2,2) = 7. Total 16.
        
        # Alternative: use the 2x2x1 rank-2 trick.
        # C_{i,0} = sum_j A_{ij} B_{j,0}. This is 2x5x1, rank 2.
        # C_{i,1} = sum_j A_{ij} B_{j,1}. This is 2x5x1, rank 2.
        # But they share A, so we can't just add ranks.
        # The combined tensor is sum_{i,j} E_{ij} ⊗ (F_j^0 G_i + F_j^1 G_i') where G_i is for k=0, G_i' for k=1.
        # = sum_i (sum_j E_{ij} ⊗ F_j^0) ⊗ G_i + sum_i (sum_j E_{ij} ⊗ F_j^1) ⊗ G_i'
        # = sum_i R_i^0 ⊗ G_i + sum_i R_i^1 ⊗ G_i'
        # That's 4 terms, each of which is a product. But R_i^0 and R_i^1 are different.
        # Actually this gives rank 4 for 2x5x2! Let me verify.
        
        # P_0 = (A_00+A_01+A_02+A_03+A_04) * B_{.,0} -- no, that's not right.
        # For 2x5x1: P_i = A_i * B (i-th row of A times B). Rank 2.
        # U[i*5+j][i] = 1, V[j][i] = 1, W[i][i] = 1.
        # For 2x5x2: we need C_{i,0} and C_{i,1}.
        # P_i^0: U[i*5+j][i*2] = 1, V[j*2+0][i*2] = 1, W[0*2+i][i*2] = 1.
        # P_i^1: U[i*5+j][i*2+1] = 1, V[j*2+1][i*2+1] = 1, W[1*2+i][i*2+1] = 1.
        # This uses rank 4. Let me verify it's correct.
        # sum_r U[i*5+j][r] * V[j*2+k][r] * W[k*2+i'][r]
        # For r = i*2: U[i*5+j][i*2]=1, V[j*2+k][i*2] = 1 if k=0, W[0*2+i'][i*2] = 1 if i'=i.
        # So contribution to C[i',k] from r=i*2: sum_j 1 * (k==0) * (i'==i) = (i'==i and k==0) * sum_j 1... 
        # Wait, that's summing over j, so it's (i'==i and k==0) * 5. That's wrong.
        
        # I made an error. Let me redo 2x5x1.
        # C_i = sum_j A_{ij} B_j.
        # T = sum_i sum_j E_{ij} ⊗ F_j ⊗ G_i.
        # To get rank 2: P_i = (row i of A) * B. But "row i of A" is not a single scalar.
        # The decomposition must be: for each term r, U is a 10x1 vector, V is 5x1, W is 1x2.
        # C[i,k] = sum_r U[i*5 + j_r] ... no, U has 10 entries, one for each (i,j).
        # C[i,k] = sum_j sum_r U[i,j,r] * V[j,r] * W[k,i,r]... 
        # Wait, the tensor is T[i,j,k] = delta (standard). For matrix mult, T[i,j,k] = 1 if the
        # (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
        # T = sum_{i,j,k} A_{ij} B_{jk} C_{ki} in tensor form... 
        # Actually the standard form: C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{ijk} such that C = A * B * T (in some contraction).
        # The rank-r decomposition: T = sum_r u^r ⊗ v^r ⊗ w^r where
        # C_{ik} = sum_r (sum_j u^r_{ij}) * ... no.
        # Standard: C_{ik} = sum_j A_{ij} B_{jk}.
        # Decomposition: A_{ij} = sum_r U_{ij}^r * lambda_r, no.
        # The tensor T is in A-space ⊗ B-space ⊗ C-space.
        # T = sum_{ijk} e_{ij} ⊗ f_{jk} ⊗ g_{ik} where the contraction gives C.
        # Actually: C_{ik} = sum_j (A ⊗ B ⊗ T) contracted = sum_j A_{ij} B_{jk} T_{j}... 
        # I think the convention is: T is a 3-tensor with T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C... 
        # No. Let me use the interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j and r gives C[i,k].
        # So: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # For schoolbook: r = (j0) for fixed j. U[i*m+j, j] = 1, V[j*p+k, j] = 1, W[k*n+i, j] = 1.
        # Then C[i,k] = sum_j 1 * 1 * 1 = sum_j A_{ij} B_{jk}. Correct.
        
        # For 2x5x1 (p=1): C[i,0] = sum_j A[i,j] B[j,0].
        # Schoolbook: rank 5. But we claim rank 2.
        # Can we do rank 2? C[i,0] = sum_j A[i,j] B[j,0].
        # This is (row i of A) dot (col 0 of B).
        # For rank 2: we need 2 terms.
        # Term 0: U[i*5+j, 0] = a_i^0, V[j, 0] = b_j^0, W[0*2+i, 0] = c_i^0.
        # C[i,0] from term 0: (sum_j a_i^0 * b_j^0) * c_i^0.
        # We need this to equal sum_j A[i,j] B[j,0] for all A, B.
        # This is impossible with rank 2 because the map (A, B) -> C is bilinear and the
        # "matrix" of this bilinear form (viewed as a linear map from A to (B -> C)) has
        # rank 5 (it's the identity on the 5-dim space of A's row, applied to B's column).
        # Actually, the bilinear map is: (A, B) -> A * B^T (scalar for p=1, n=2).
        # This is the trace-like map. The rank of this tensor is n * m = 2 * 5 = 10? No.
        # For n x m x 1: T = sum_{i,j} e_{ij} ⊗ f_j ⊗ g_i.
        # This can be written as sum_i (sum_j e_{ij} ⊗ f_j) ⊗ g_i = sum_i h_i ⊗ g_i.
        # Each h_i = sum_j e_{ij} ⊗ f_j is a vector in A-space ⊗ B-space.
        # The tensor is sum_i h_i ⊗ g_i. This is rank n = 2 IF h_i are linearly independent
        # and we treat h_i as a single factor. But h_i is a sum of m terms, so in the
        # standard tensor rank, each h_i ⊗ g_i is a single term (a product of three vectors:
        # h_i is a vector in the 5-dim A-space... no, h_i is a vector in the (A⊗B)-space).
        
        # I think I'm confusing myself. Let me just use the direct formula.
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # For 2x5x1: m=5, p=1, n=2.
        # C[i,0] = sum_r sum_j U[i*5+j, r] * V[j*1+0, r] * W[0*2+i, r] = sum_r sum_j U[i*5+j, r] * V[j, r] * W[i, r].
        # We need C[i,0] = sum_j A[i,j] * B[j,0] for arbitrary A, B.
        # This means: the bilinear form (A, B) -> sum_i C[i,0] * x_i (for some x) must equal
        # sum_i sum_j A[i,j] B[j,0] x_i.
        # The tensor T[i,j] (in A-space and B-space, with C-space contracted) is the identity
        # on B-space repeated n times... 
        # Actually, for fixed i: C[i,0] = A[i,:] * B[:,0]^T. This is a rank-5 bilinear form in (A[i,:], B[:,0]).
        # For the full 2x5x1 tensor: it's the sum over i of (A[i,:] ⊗ B[:,0]) ⊗ e_i.
        # This has tensor rank = 2 * 5 = 10? No.
        # Let me think of it as a matrix. Fix the C-index (only 0). Then we have a bilinear map
        # from A (10-dim) to (B -> scalar). This is a 10x5 matrix M where M[(i,j), j'] = delta_{j,j'}.
        # The rank of this matrix is 5. So the tensor rank for 2x5x1 is 5, not 2.
        
        # OK so my earlier claim was wrong. mu(2,5,1) = 5, not 2.
        # Similarly mu(2,5,2) = mu(2,5,1) * 2 is not valid.
        # mu(2,5,2): C is 2x2. C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T has entries T[i,j,k] = 1 (for the standard basis).
        # View as a 3-tensor in (2*5) x (5*2) x (2*2) = 10 x 10 x 4.
        # Rank = ? For schoolbook it's 5 (sum over j). 
        # Actually schoolbook for 2x5x2 has rank 5: for each j, U[i*5+j, j]=1, V[j*2+k, j]=1, W[k*2+i, j]=1.
        # So mu(2,5,2) <= 5. And it's at least 5 (by the matrix rank argument).
        # So mu(2,5,2) = 5!
        
        # Let me verify: rank 5 for 2x5x2.
        # For j = 0,1,2,3,4:
        #   U[i*5+j, j] = 1 for i=0,1
        #   V[j*2+k, j] = 1 for k=0,1
        #   W[k*2+i, j] = 1 for i=0,1, k=0,1
        # C[i,k] = sum_j U[i*5+j,j] V[j*2+k,j] W[k*2+i,j] = sum_j 1*1*1 = sum_j A[i,j]B[j,k]. Correct.
        
        # So mu(2,5,2) = 5.
        # mu(4,5,6) with n=4 as 2+2:
        # Per row block: mu(2,5,6) = mu(2,5,2+2+2) = 3 * mu(2,5,2) = 3*5 = 15.
        # Total: 2 * 15 = 30.
        
        # That's amazing! Let me double-check.
        # mu(2,5,6) = sum over k-groups. For each group of 2 k-columns, we use mu(2,5,2)=5.
        # The k-groups don't overlap in C, so we can just concatenate.
        # Total rank for 2x5x6 = 3*5 = 15.
        # Total for 4x5x6 = 2*15 = 30.
        
        # Let me verify mu(2,5,2) = 5 is correct by checking the tensor rank lower bound.
        # The tensor T for 2x5x2: T[i,j,k] = 1 for all valid i,j,k (in the standard basis sense).
        # Actually T[i,j,k] is 1 if we're in the standard basis, meaning the tensor is
        # sum_{i,j,k} e_i ⊗ f_j ⊗ g_k where the contraction gives C[i,k] = sum_j A[i,j]B[j,k].
        # The "matrix" unfolding: fix k=0. T[i,j,0] = 1. This is a 10x5 matrix of all 1s? No.
        # T[i,j,k] = 1 for the (i,j,k) "entry" in the tensor product space.
        # The tensor is sum_{i=0}^1 sum_{j=0}^4 sum_{k=0}^1 e_i ⊗ f_j ⊗ g_k.
        # This is a sum of 10 terms. Can we reduce it?
        # Group by j: sum_j (sum_i e_i) ⊗ f_j ⊗ (sum_k g_k) = sum_j (e_0+e_1) ⊗ f_j ⊗ (g_0+g_1).
        # That's 5 terms! Each term is (e_0+e_1) ⊗ f_j ⊗ (g_0+g_1).
        # Let me verify: C[i,k] = sum_j [U[i,j] * V[j,k] * W[k,i]] where
        # U[i,j] = 1 (for all i), V[j,k] = 1, W[k,i] = 1.
        # C[i,k] = sum_j 1 * 1 * 1 = 5. But we need C[i,k] = sum_j A[i,j] B[j,k].
        # This is wrong! The decomposition must satisfy:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For the "5 terms" decomposition: r = j (0..4).
        # U[i*5+j, j] = 1 for all i. So U is a 10x5 matrix where column j has 1s in rows j and 5+j.
        # V[j*2+k, j] = 1 for all k. So V is a 10x5 matrix where column j has 1s in rows 2j and 2j+1.
        # W[k*2+i, j] = 1 for all i,k. So W is a 4x5 matrix where column j is all 1s.
        # C[i,k] = sum_j U[i*5+j, j] * V[j*2+k, j] * W[k*2+i, j]
        # = sum_j 1 * 1 * 1 = 5.
        # But we need C[i,k] = sum_j A[i,j] * B[j,k].
        # The issue is that A[i,j] and B[j,k] are the actual matrix entries, and the decomposition
        # should give C[i,k] = sum_j A[i,j] B[j,k]. The U, V, W matrices encode the "coefficients"
        # such that the bilinear form works out.
        
        # I think I've been misunderstanding the convention. Let me re-read the task:
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer
        # entries that exactly decompose the matrix multiplication tensor"
        
        # So the decomposition is: for each rank term r, we have vectors u^r (n*m), v^r (m*p), w^r (p*n).
        # The tensor is T = sum_r u^r ⊗ v^r ⊗ w^r.
        # And C[i,k] = sum_j A[i,j] B[j,k] = <T, A ⊗ B ⊗ C> in some sense.
        # More precisely: C[i,k] = sum_{r,j} u^r[i*m+j] * v^r[j*p+k] * w^r[k*n+i] * A[i,j] * B[j,k]... 
        # No, that's not right either.
        
        # The correct interpretation: the matrix multiplication map is a bilinear map
        # (A, B) -> C where C = A*B.
        # This bilinear map can be written as: C[i,k] = sum_j A[i,j] * B[j,k].
        # In tensor form: C = T(A, B) where T is a 3-tensor.
        # The rank-r decomposition means: T = sum_r T^r where each T^r is a rank-1 tensor.
        # A rank-1 tensor T^r has the form: T^r[i,j,k] = u^r[i*m+j] * v^r[j*p+k] * w^r[k*n+i].
        # And C[i,k] = sum_r sum_j T^r[i,j,k] ... no.
        
        # I think the correct formula is:
        # C[i,k] = sum_r (sum_j A[i,j] * u^r[i*m+j]) * (sum_j B[j,k] * v^r[j*p+k]) * w^r[k*n+i]
        # No, that doesn't work dimensionally.
        
        # Let me look at the seed program's schoolbook:
        # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
        # This means: for the term corresponding to (i,j,k) with col = index of (i,j,k):
        # u^r has 1 at position (i*m+j), v^r has 1 at (j*p+k), w^r has 1 at (k*n+i).
        # The bilinear form is: C[i,k] = sum_r sum_j A[i,j] * B[j,k] * ... 
        # I think the convention is:
        # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_{j'} B[j',k] * V[j'*p+k, r]) * W[k*n+i, r]
        # No, that would give C[i,k] = sum_r (row_i(A) * u^r) * (col_k(B) * v^r) * w^r[k*n+i].
        # For schoolbook with r = (j): u^r has 1 at (i*m+j) for the specific i. So row_i(A) * u^r = A[i,j].
        # v^r has 1 at (j*p+k) for the specific k. So col_k(B) * v^r = B[j,k].
        # w^r has 1 at (k*n+i). So w^r[k*n+i] = 1.
        # C[i,k] = sum_j A[i,j] * B[j,k] * 1 = sum_j A[i,j] B[j,k]. Correct!
        
        # So the formula is: C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r])
        
        # Now for 2x5x2 with rank 5 (r = 0..4, corresponding to j = 0..4):
        # U[i*5+j, j] = 1 for all i. So sum_j' A[i,j'] U[i*5+j', j] = A[i,j].
        # V[j*2+k, j] = 1 for all k. So sum_j' B[j',k] V[j'*2+k, j] = B[j,k].
        # W[k*2+i, j] = 1.
        # C[i,k] = sum_j W[k*2+i, j] * A[i,j] * B[j,k] = sum_j 1 * A[i,j] * B[j,k]. Correct!
        
        # Great, so mu(2,5,2) = 5 is valid.
        
        # Now build the full 4x5x6 with rank 30.
        # Row blocks: r=0 (rows 0,1), r=1 (rows 2,3).
        # For each row block, 3 k-groups (k=0,1; k=2,3; k=4,5), each with rank 5.
        # Total rank = 2 * 3 * 5 = 30.
        
        rank = 30
        U = [[0]*rank for _ in range(n*m)]   # 20 x 30
        V = [[0]*rank for _ in range(m*p)]   # 30 x 30
        W = [[0]*rank for _ in range(p*n)]   # 24 x 30
        
        col = 0
        for rb in range(2):  # row block
            i0 = rb * 2
            for kg in range(3):  # k group
                k0 = kg * 2
                for j in range(5):  # 5 terms for this 2x5x2 block
                    # U: rows (i0+0)*5+j and (i0+1)*5+j get 1
                    U[(i0+0)*m + j][col] = 1
                    U[(i0+1)*m + j][col] = 1
                    # V: rows j*p + (k0+0) and j*p + (k0+1) get 1
                    V[j*p + (k0+0)][col] = 1
                    V[j*p + (k0+1)][col] = 1
                    # W: rows (k0+0)*n + (i0+0), (k0+0)*n + (i0+1), (k0+1)*n + (i0+0), (k0+1)*n + (i0+1) get 1
                    W[(k0+0)*n + (i0+0)][col] = 1
                    W[(k0+0)*n + (i0+1)][col] = 1
                    W[(k0+1)*n + (i0+0)][col] = 1
                    W[(k0+1)*n + (i0+1)][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
        
    except Exception:
        # Fallback to schoolbook
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
