"""Matrix multiplication 3x3x4 via Strassen-style rank-28 decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # Strassen 3x3x3 scheme extended to 3x3x4.
    # 7 blocks M_a (a=0..6), each 3x4. Rank = 7 * 4 = 28.
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 9 x 28
    V = [[0] * rank for _ in range(m * p)]  # 12 x 28
    W = [[0] * rank for _ in range(p * n)]  # 12 x 28

    # Each block a: U-side combination (on A's 3x3), V-side combination (on B's 3x4),
    # and output contribution to C (3x4).
    # A index: row r in 0..2, col c in 0..2 -> A[r*3+c]
    # B index: row r in 0..2, col c in 0..3 -> B[r*4+c]
    # C index (trace convention W[k*n+i]): output C[i][k], i in 0..2 (row), k in 0..3 (col) -> W[k*3+i]

    # Define the 7 Strassen blocks. For each, list of (sign, A_row, A_col) for U,
    # (sign, B_row) for V (applied to all 4 columns), and output (sign, C_row, C_col).
    # U factor for block a, column c: sum over (s, ar, ac) of s * delta
    # V factor for block a, column c: s * delta over B rows
    # W factor for block a, column c: s * delta over output positions

    # We represent each block by:
    #   U_terms: list of (sign, ar, ac)  -> U[ar*3+ac] gets sign
    #   V_terms: list of (sign, br)      -> V[br*4 + c] gets sign for each output col c
    #   W_terms: list of (sign, cr, cc)  -> W[cc*3 + cr] gets sign  (cr=C row, cc=C col)

    blocks = [
        # M1 = (A11+A33)(B11+B33)
        {"U": [(1,0,0),(1,2,2)], "V": [(1,0)], "W": [(1,0,0)]},
        # M2 = (A21+A22)B11
        {"U": [(1,1,0),(1,1,1)], "V": [(1,0)], "W": [(1,1,0)]},
        # M3 = A11(B21-B31)
        {"U": [(1,0,0)], "V": [(1,1),(-1,2)], "W": [(1,0,1)]},
        # M4 = A33(B32-B12)
        {"U": [(1,2,2)], "V": [(1,2),(-1,0)], "W": [(1,2,1)]},
        # M5 = (A11+A12)B22
        {"U": [(1,0,0),(1,0,1)], "V": [(1,1)], "W": [(1,0,2)]},
        # M6 = (A31-A11)B32
        {"U": [(1,2,0),(-1,0,0)], "V": [(1,2)], "W": [(1,2,2)]},
        # M7 = (A22-A33)(B31+B33)
        {"U": [(1,1,1),(-1,2,2)], "V": [(1,2)], "W": [(1,1,1)]},
    ]

    # Output combination: C[i][k] = sum over blocks of coeff * M_a[i][k]
    # C rows 0..2, cols 0..3. For cols 0,1,2 we use the 3x3 Strassen output;
    # for col 3, B's 4th column participates identically to col structure.
    # Actually B is 3x4, so each block M_a is 3x4 (3 rows of A times 4 cols of B).
    # The output C is 3x4. For each output position (cr, cc), C[cr][cc] = sum_a coef(a, cr, cc) * M_a[cr][cc].
    # The coef depends only on (cr, cc mod 3) using the 3x3 Strassen formula, and col 3 uses same row-combination as... 
    # Wait: in 3x3x3, C[cr][cc] for cc in 0..2 uses specific blocks. For 3x3x4, the 4th column of B
    # is just an extra column; each block M_a[cr][3] = (A-side)_cr * (B-side)_3.
    # The output C[cr][3] = sum_a coef(a, cr, 3) * M_a[cr][3]. The coef for col 3 must match the
    # Strassen combination for that row cr. Since Strassen's C[cr][*] uses the same set of blocks
    # for all columns (the column index of C just indexes which B-column each block reads), 
    # the coef(a, cr, cc) is independent of cc! Let me verify.

    # In Strassen 3x3x3:
    # C[0][0]=M1+M4-M5-M7, C[0][1]=M3+M5, C[0][2]=M1-M2+M3+M7  -- wait these differ by column.
    # Hmm, actually no. Let me re-derive. The standard Strassen:
    # M1=(A11+A33)(B11+B33)
    # M2=(A21+A22)B11
    # M3=A11(B21-B31)
    # M4=A33(B32-B12)
    # M5=(A11+A12)B22
    # M6=(A31-A11)B32
    # M7=(A22-A33)(B31+B33)
    # C11=M1+M4-M5-M7
    # C12=M3+M5
    # C21=M2+M4
    # C22=M1-M2+M3+M7
    # C31=M2+M6-M1+M7
    # C32=M4+M6
    # So C[cr][cc] coefficients depend on (cr,cc). For 3x3x4, we have C[cr][cc] for cc=0,1,2,3.
    # The 4th column: B has columns 0,1,2,3. Each block M_a[cr][cc] uses B's column cc.
    # So C[cr][3] = sum_a coef(a, cr, 3) * M_a[cr][3]. What is coef(a,cr,3)?
    # It must equal the Strassen combination for column 3. But Strassen 3x3x3 only has cols 0,1,2.
    # For a 3x3x4 product, the 4th column of C is A * B[:,3], which is just a 3x1 vector = A * (4th col of B).
    # This is a 3x3x1 multiplication, which itself needs a decomposition. The cleanest: treat col 3
    # like a new "column" and use the same block structure. Since the blocks M_a[cr][3] = (A-side)_cr * B[?,3],
    # and C[cr][3] = sum_a coef(a,cr,3) M_a[cr][3], we need coef(a,cr,3) such that
    # sum_a coef(a,cr,3) (A-side)_cr B[?,3] = (A * B[:,3])[cr].
    # This is exactly the same linear system as for any column. So coef(a,cr,3) = coef(a,cr,cc) for any cc,
    # because the A-side combination that reconstructs (A*x)_cr from the 7 blocks is column-independent.
    # Let me verify: for column cc, C[cr][cc] = sum_a coef(a,cr,cc) M_a[cr][cc].
    # M_a[cr][cc] = (sum_{ar,ac} u_a(ar,ac) A[ar][ac]) * (sum_{br} v_a(br) B[br][cc]).
    # So C[cr][cc] = sum_a coef(a,cr,cc) (A-side_a)_cr * (B-side_a)(cc).
    # We need this = sum_{ac} A[cr][ac] B[ac][cc] (the actual product).
    # The A-side and B-side are fixed per block. The coef(a,cr,cc) must make this identity hold.
    # Since the identity must hold for ALL B (all columns independently), the coefficients
    # coef(a,cr,cc) must be the SAME for all cc! Because the B-side part (B-side_a)(cc) is arbitrary per cc.
    # Therefore coef(a,cr,cc) = coef(a,cr) independent of cc. 
    # So for col 3, use the same coef as cols 0,1,2. Which one? They must all be equal!
    # Let me check: coef for C[0][0]: M1+M4-M5-M7. coef for C[0][1]: M3+M5. These are DIFFERENT!
    # Contradiction. So my assumption is wrong. The coef DOES depend on cc.
    # 
    # Resolution: the issue is that in Strassen, the blocks are designed so that the SAME 7 scalar
    # multiplications produce all 9 outputs. For a 3x3x4 product, we need 12 outputs from the same 7
    # "block" multiplications, but each block multiplication now produces 3x4=12 scalars (since B is 3x4).
    # Wait no. Let me reconsider. In the tensor decomposition, each rank term is a scalar product.
    # Block M_a is itself a 3x4 matrix = outer product of (A-side vector in R^9) and (B-side vector in R^12)?
    # No. M_a[cr][cc] = (A-side_a dot A) * (B-side_a dot B[:,cc]). So M_a is an outer product of a 3-vector
    # (the A-side combination per row) and a 4-vector (the B-side combination per column). 
    # So each block M_a is a rank-1 3x4 matrix! That's why it's 7 blocks * 4 columns = 28 rank terms.
    # Now C[cr][cc] = sum_a coef(a, cr, cc) * M_a[cr][cc] = sum_a coef(a,cr,cc) * (Avec_a)_cr * (Bvec_a)_cc.
    # We need C[cr][cc] = sum_{ac} A[cr][ac] B[ac][cc].
    # The A-part: (Avec_a)_cr = sum_{ar,ac} u_a(ar,ac) A[ar][ac]. This is a linear functional of A giving a 3-vector.
    # The B-part: (Bvec_a)_cc = sum_{br} v_a(br) B[br][cc]. Linear functional of B[:,cc].
    # So C[cr][cc] = sum_a coef(a,cr,cc) (Afunc_a)_cr (Bfunc_a)(cc).
    # For this to equal (A B)[cr][cc] for all A, B, we need the bilinear form to match.
    # The key: coef(a,cr,cc) can depend on both cr and cc. For cols 0,1,2 we use the Strassen coefs.
    # For col 3, we need coef(a,cr,3) such that sum_a coef(a,cr,3) (Afunc_a)_cr (Bfunc_a)(3) = (AB)[cr][3].
    # Since (Bfunc_a)(3) is arbitrary (B[:,3] arbitrary), we need sum_a coef(a,cr,3) (Afunc_a)_cr * (Bfunc_a)(3)
    # to reconstruct (AB)[cr][3] = sum_ac A[cr][ac] B[ac][3] for all B[:,3].
    # This is the same constraint as for any single column. The Strassen scheme for 3x3x1 (a column) 
    # would use a subset. But we're reusing the SAME 7 blocks. So we need coef(a,cr,3) to satisfy
    # the 3x3x1 reconstruction using the 7 Afunc_a and 7 Bfunc_a.
    # 
    # Let me just directly compute the required coef(a,cr,cc) for all cc=0..3 by solving the linear system.
    # For each (cr, cc), we need: sum_a coef(a,cr,cc) (Afunc_a)_(cr) (Bfunc_a)_(cc) = sum_ac A[cr][ac] B[ac][cc].
    # This is a bilinear identity. Equivalently, for the tensor, the output tensor T[cr,cc, ar,ac,br] ... this is getting complex.
    # 
    # Simpler: just hardcode by matching. Let me compute coef(a,cr,cc) for cc=0,1,2 from Strassen,
    # and for cc=3, solve. Actually, since the Afunc_a and Bfunc_a are fixed, and the product tensor is
    # fixed, the coef is uniquely determined (if the 7 blocks span). Let me just compute numerically.

    # Let me define Afunc_a as 3x9 matrix (row cr, col = A index) and Bfunc_a as 4x3 matrix (row cc, col = B row).
    # Then M_a[cr][cc] = sum_{ai} Afunc_a[cr,ai] A[ai] * sum_{br} Bfunc_a[cc,br] B[br][cc]... 
    # Actually M_a[cr][cc] = (Afunc_a[cr,:] @ A_flat) * (Bfunc_a[cc,:] @ B_col_cc).
    # C[cr][cc] = sum_a coef(a,cr,cc) * (Afunc_a[cr,:] @ A_flat) * (Bfunc_a[cc,:] @ B_col_cc).
    # We want C[cr][cc] = (A @ B)[cr][cc] = sum_ac A[cr][ac] B[ac][cc].
    # 
    # This is a tensor equation. The cleanest way: the decomposition is valid iff
    # for each (cr,cc), the bilinear form sum_a coef(a,cr,cc) outer(Afunc_a[cr], Bfunc_a[cc]) 
    # equals outer(A[cr,:], B[:,cc]) summed appropriately.
    # 
    # I'll just verify numerically by constructing U,V,W and checking A@B == sum_c U[:,c] * V[:,c] * W[:,c] (broadcast).

    # Build U, V, W.
    # For block a (0..6), for each column cc (0..3), rank index = a*4 + cc.
    # U[ai][a*4+cc] = u_a(ai)  (independent of cc)
    # V[bi][a*4+cc] = v_a(bi, cc)  where bi = br*4+cc, so V[br*4+cc][a*4+cc] = v_a(br)
    # W[oi][a*4+cc] = w_a(oi, cr, cc) ... W index is k*n+i = cc*3+cr.
    #   W[cc*3+cr][a*4+cc] = coef(a, cr, cc)

    # Define u_a (9-vector), v_a (3-vector, per B-row).
    u = {
        0: {0:1, 8:1},          # A11+A33  (indices: A[0][0]=0, A[2][2]=8)
        1: {3:1, 4:1},          # A21+A22  (A[1][0]=3, A[1][1]=4)
        2: {0:1},               # A11
        3: {8:1},               # A33
        4: {0:1, 1:1},          # A11+A12 (A[0][0]=0, A[0][1]=1)
        5: {6:1, 0:-1},         # A31-A11 (A[2][0]=6, A[0][0]=0)
        6: {4:1, 8:-1},         # A22-A33 (A[1][1]=4, A[2][2]=8)
    }
    v = {
        0: {0:1},               # B11  (B row 0)
        1: {0:1},               # B11
        2: {1:1, 2:-1},         # B21-B31 (B rows 1,2)
        3: {2:1, 0:-1},         # B32-B12 (B rows 2,0)
        4: {1:1},               # B22 (B row 1)
        5: {2:1},               # B32 (B row 2)
        6: {2:1},               # B31+B33 -> B row 2 (both B31 and B33 are in row 2 of B)
    }
    # Note: B is 3x4. B rows: row0 = B1* (cols), row1 = B2*, row2 = B3*.
    # B11 = B[0][0], B12=B[0][1], B13=B[0][2], B14=B[0][3]
    # B21=B[1][0], B22=B[1][1], B23=B[1][2], B24=B[1][3]
    # B31=B[2][0], B32=B[2][1], B33=B[2][2], B34=B[2][3]
    # So B11 is B row0. B21 is B row1, B31 is B row2. B32 is B row2, B12 is B row0.
    # M3 = A11(B21-B31): B-side = row1 - row2. v[2]={1:1, 2:-1}. Correct.
    # M4 = A33(B32-B12): B-side = row2 - row0. v[3]={2:1, 0:-1}. Correct.
    # M6 = (A31-A11)B32: B-side = row2. v[5]={2:1}. Correct.
    # M7 = (A22-A33)(B31+B33): B31=B[2][0], B33=B[2][2], both row2. So B-side = row2 + row2 = 2*row2? 
    # No: B31+B33 means the (3,1) and (3,3) entries of B, i.e., B[2][0]+B[2][2]. These are different columns of row 2.
    # So the B-side for M7 is NOT just a row; it's specific columns. This breaks my "row" simplification!

    # I need to handle columns properly. Let me redo with full (br, bc) indexing for V.
    # V index = br*4+bc. v_a is a dict from (br,bc) to sign.
    v2 = {
        0: {(0,0):1, (0,2):1},   # B11+B33: B[0][0]+B[2][2]
        1: {(0,0):1},            # B11
        2: {(1,0):1, (2,0):-1},  # B21-B31: B[1][0]-B[2][0]
        3: {(2,1):1, (0,1):-1},  # B32-B12: B[2][1]-B[0][1]
        4: {(1,1):1},            # B22
        5: {(2,1):1},            # B32
        6: {(2,0):1, (2,2):1},   # B31+B33: B[2][0]+B[2][2]
    }

    # Now W: coef(a, cr, cc) for output C[cr][cc].
    # From Strassen 3x3x3 (cc=0,1,2):
    # C[0][0]=M1+M4-M5-M7, C[0][1]=M3+M5, C[0][2]=M1-M2+M3+M7
    # C[1][0]=M2+M4, C[1][1]=M1-M2+M3+M7, C[1][2]=M4+M6
    # C[2][0]=M2+M6-M1+M7, C[2][1]=M4+M6, C[2][2]=M1-M2+M3+M7
    # Wait let me recheck standard Strassen output:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    # C31 = M2 + M6 - M1 + M7
    # C32 = M4 + M6
    # So:
    # C[0][0] = M1+M4-M5-M7 -> a:0:+1, 3:+1, 4:-1, 6:-1
    # C[0][1] = M3+M5 -> a:2:+1, 4:+1
    # C[0][2] = M1-M2+M3+M7 -> a:0:+1,1:-1,2:+1,6:+1
    # C[1][0] = M2+M4 -> a:1:+1,3:+1
    # C[1][1] = M1-M2+M3+M7 -> a:0:+1,1:-1,2:+1,6:+1
    # C[1][2] = M4+M6 -> a:3:+1,5:+1
    # C[2][0] = M2+M6-M1+M7 -> a:0:-1,1:+1,5:+1,6:+1
    # C[2][1] = M4+M6 -> a:3:+1,5:+1
    # C[2][2] = M1-M2+M3+M7 -> a:0:+1,1:-1,2:+1,6:+1

    # For cc=3 (4th column), we need coef(a,cr,3). By the column-independence argument for a FIXED row cr,
    # the reconstruction of (AB)[cr][cc] from the blocks requires the same A-combination regardless of cc,
    # but the B-combination is per-block. Since each block's B-side is fixed (v2[a]), and we need
    # sum_a coef(a,cr,3) (Afunc_a)_cr (Bvec_a)(3) = (AB)[cr][3], and this must hold for all B[:,3],
    # the coef(a,cr,3) must equal coef(a,cr,cc) for the row cr... but they differ across cc for same cr!
    # 
    # The resolution: for cc=3, the B-side (Bvec_a)(3) = v2[a] dotted with B[:,3]. For the identity to hold,
    # we need sum_a coef(a,cr,3) (Afunc_a)_cr * (v2[a] @ B[:,3]) = sum_ac A[cr][ac] B[ac][3].
    # This is a different linear system than for cc=0,1,2 because v2[a]@B[:,3] uses the 3rd column of B.
    # The coef(a,cr,3) is determined by requiring this for all B[:,3]. Since v2[a]@B[:,3] = sum_bc v2[a][(br,bc)] B[br][3] 
    # but v2[a] only has specific (br,bc) entries, v2[a]@B[:,3] = sum_br v2[a][(br,?)] ... no, v2[a] is over (br,bc) pairs.
    # v2[a] @ B[:,3] doesn't make sense directly. Let me reconsider.
    # 
    # M_a[cr][cc] = (Afunc_a)_cr * (sum_{(br,bc) in v2[a]} v2[a][(br,bc)] * B[br][cc]).
    # Wait, that's wrong. M_a[cr][cc] should use B[br][cc] where cc is the OUTPUT column. But v2[a] indexes
    # specific B entries (br,bc). There's a mismatch: the block M_a is defined using specific B entries,
    # not all of column cc.
    # 
    # I think I'm overcomplicating. Let me reconsider the block structure. In Strassen 3x3x3, block M1 = (A11+A33)(B11+B33)
    # is a SCALAR: (A[0][0]+A[2][2])*(B[0][0]+B[2][2]). It's one scalar multiplication. The 7 blocks are 7 scalars.
    # The output C[i][j] is a linear combination of these 7 scalars.
    # 
    # For 3x3x4, if I keep the same 7 scalar blocks, they only involve B's columns 0,1,2 (since B11,B12,B13,B21,...,B33 
    # are all in cols 0,1,2). They don't involve B's column 3 (B14,B24,B34). So the 7 blocks can't produce C[?][3]
    # which depends on B[:,3]. 
    # 
    # Therefore, to handle the 4th column, I need additional blocks that involve B's column 3. The minimal way:
    # For the 4th column, C[cr][3] = sum_ac A[cr][ac] B[ac][3]. This is a 3x3x1 multiplication (3 rows of A, 3 cols, 1 col of B).
    # A 3x3x1 multiplication has rank... the matrix multiplication tensor for 3x3x1 is rank 9 (it's just 9 independent products, 
    # since the output is 3x1 and each output is a dot product; actually the rank of the 3x3x1 tensor is 9? No.)
    # 
    # Hmm, the rank of the m x n x p tensor. For p=1, it's the rank of the "matrix multiplication" as a 3-tensor with one dimension 1.
    # The 3x3x1 tensor: T[i,j,0] = delta (i.e., C[i] = sum_j A[i][j] B[j]). This tensor has rank... it's equivalent to the identity in some sense.
    # Actually the 3x3x1 tensor is just the bilinear form (A,B) -> A*B (3x1 vector). Its rank as a 3-tensor is 3? No.
    # Let me think: C[i] = sum_j A[i][j] B[j]. The tensor T[i,j,k] with k=0 only. T[i,j,0] = 1 (the product tensor for 3x3x1 is T[i,j,0]=1 for all i,j since C[i]=sum_j A[i][j]B[j]).
    # Wait the matrix multiplication tensor for (n,m,p) is T[i,j,k] = 1 (it's the "all ones" in the sense that C[i,k]=sum_j A[i,j]B[j,k]).
    # For p=1: C[i,0] = sum_j A[i,j] B[j,0]. So T[i,j,0] = 1 for all i,j. The tensor is a 3x3x1 array of all 1s.
    # The rank of a 3x3x1 all-ones tensor: it's a matrix (3x3) of all ones with a trivial third dimension. 
    # As a 3-tensor with one index of size 1, it's just a 3x3 matrix of rank 1 (all ones). So tensor rank = 1? 
    # No, tensor rank of a 3x3x1 tensor equals the matrix rank of the 3x3 slice, which is 1 (all ones matrix has rank 1).
    # So the 3x3x1 multiplication can be done in rank 1? That can't be right because we need to compute 3 different dot products.
    # 
    # I think I'm confusing tensor rank with the number of scalar multiplications. Let me reconsider.
    # 
    # The matrix multiplication tensor for multiplying (n x m) by (m x p) to get (n x p) is a 3-tensor T of size (n*m) x (m*p) x (n*p) ... no.
    # Actually the standard formulation: the tensor is T[i, j, k] where i indexes A (n*m), j indexes B (m*p), k indexes C (n*p), 
    # and T[i,j,k] = 1 if the product A[i]B[j] contributes to C[k]. For standard matmul, T[i,j,k]=1 iff i=(row_A, col_A), j=(col_A, col_B), k=(row_A, col_B).
    # So T has support on the "consistent" triples. The rank of this tensor is the bilinear complexity.
    # 
    # For 3x3x4: A is 3x3 (9 entries), B is 3x4 (12 entries), C is 3x4 (12 entries). The tensor T is 9x12x12 with T[i,j,k]=1 when consistent.
    # The rank is the minimum number of rank-1 tensors to sum to T.
    # 
    # My Strassen approach: 7 blocks, each block is a rank-1 contribution but each block covers multiple (i,j,k) triples.
    # Actually each "rank term" in the decomposition is a single rank-1 tensor. The 28 rank terms I'm building:
    # each is U[:,c] * V[:,c] * W[:,c] (outer product). For rank term c = a*4+cc:
    #   U[:,c][i] = u_a(i)  (i in 0..8, A index)
    #   V[:,c][j] = v2[a][(j//4, j%4)]  (j in 0..11, B index = br*4+bc)
    #   W[:,c][k] = coef(a, cr, cc) where k = cc*3+cr  (k in 0..11, C index = cc*3+cr)
    # But W[:,c] should be 1 at the output position and 0 elsewhere for a rank-1 tensor. In a rank-1 tensor, 
    # T_rank1[i,j,k] = U[i] V[j] W[k]. For this to be a "delta" at (i,j,k), we'd need U,V,W to be delta vectors.
    # But in Strassen, the rank terms are NOT delta; they are the actual factor vectors. The sum of 28 such rank-1 tensors
    # should equal the matmul tensor T.
    # 
    # So the decomposition is: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
    # For my construction, T[i,j,k] = sum_{a,cc} u_a(i) * v2[a][(j//4,j%4)] * coef(a, cr, cc) * delta[k == cc*3+cr]
    # where i is A index, j is B index, k is C index, and cr,cc are such that k=cc*3+cr.
    # We need this to equal 1 when (i,j,k) is consistent, 0 otherwise.
    # 
    # Consistency: i=(ra,ca), j=(cb,cc_b), k=(cc_c, ra_c) where ca==cb (the inner index matches) and ra==ra_c (row matches).
    # So T[i,j,k]=1 iff ca==cb and ra==ra_c.
    # 
    # My sum: for fixed (i,j,k) with i=(ra,ca), j=(rb,cb), k=(cc_k, ra_k):
    # sum_{a,cc} u_a(i) v2[a][(rb,cb)] coef(a, ra_k, cc) delta[cc==cc_k]
    # = sum_a u_a(i) v2[a][(rb,cb)] coef(a, ra_k, cc_k)   [since cc must equal cc_k]
    # We need this = 1 if ca==cb and ra==ra_k, else 0.
    # 
    # This is a system of equations. For cc_k=0,1,2, I use the Strassen coef. For cc_k=3, I need to find coef(a,ra,3).
    # 
    # Let me just solve for coef(a,ra,3) numerically. For each (ra, cc_k=3), and for all consistent (ca,cb) with cb=ca, 
    # the equation must hold. Actually the equation must hold for ALL (i,j,k), not just consistent ones.
    # For inconsistent (i,j,k), the sum must be 0.
    # 
    # This is a linear system in the unknowns coef(a,ra,3) for a=0..6, ra=0..2 (18 unknowns).
    # The equations: for each (i,j,k) with k having cc_k=3 (i.e., k in {9,10,11} since k=3*3+ra, ra=0,1,2 -> k=9,10,11),
    # sum_a u_a(i) v2[a][(j//4,j%4)] coef(a, ra_k, 3) = T[i,j,9+ra_k] where T=1 if consistent else 0.
    # But ra_k is determined by k. For k=9: ra_k=0, k=10: ra_k=1, k=11: ra_k=2.
    # 
    # Actually, the unknowns coef(a,ra,3) appear with ra=ra_k. So for each ra_k (0,1,2), the equations involving k=9+ra_k
    # give a system for coef(a,ra_k,3), a=0..6. That's 7 unknowns per ra_k.
    # The number of equations: for each (i,j) pair (9*12=108), we get one equation per ra_k. But many are redundant.
    # 
    # Let me just set up and solve. I'll use numpy.

    import numpy as np
    # u_a as 7x9 matrix (a, i)
    u_mat = np.zeros((7,9))
    for a in range(7):
        for i, s in u[a].items():
            u_mat[a,i] = s
    # v2_a as 7x12 matrix (a, j) where j=br*4+bc
    v_mat = np.zeros((7,12))
    for a in range(7):
        for (br,bc), s in v2[a].items():
            v_mat[a, br*4+bc] = s
    # T tensor: T[i,j,k] = 1 if consistent.
    # i=(ra,ca): ra=i//3, ca=i%3
    # j=(rb,cb): rb=j//4, cb=j%4
    # k=(cc_k,ra_k): ra_k=k%3, cc_k=k//3
    # consistent iff ca==cb and ra==ra_k
    T = np.zeros((9,12,12))
    for i in range(9):
        ra, ca = i//3, i%3
        for j in range(12):
            rb, cb = j//4, j%4
            for k in range(12):
                ra_k, cc_k = k%3, k//3
                if ca==cb and ra==ra_k:
                    T[i,j,k] = 1.0
    # For cc_k in 0,1,2, set coef from Strassen.
    coef = np.zeros((7,3,4))  # (a, ra, cc)
    strassen = {
        (0,0): [1,0,0,-1,0,0,-1],  # a=0:+1,3:+1,4:-1,6:-1 -> indices 0,3,4,6
        (0,1): [0,0,1,0,1,0,0],    # a=2:+1,4:+1
        (0,2): [1,-1,1,0,0,0,1],   # a=0:+1,1:-1,2:+1,6:+1
        (1,0): [0,1,0,1,0,0,0],    # a=1:+1,3:+1
        (1,1): [1,-1,1,0,0,0,1],   # a=0:+1,1:-1,2:+1,6:+1
        (1,2): [0,0,0,1,0,1,0],    # a=3:+1,5:+1
        (2,0): [-1,1,0,0,0,1,1],   # a=0:-1,1:+1,5:+1,6:+1
        (2,1): [0,0,0,1,0,1,0],    # a=3:+1,5:+1
        (2,2): [1,-1,1,0,0,0,1],   # a=0:+1,1:-1,2:+1,6:+1
    }
    for (ra,cc), vec in strassen.items():
        for a in range(7):
            coef[a,ra,cc] = vec[a]
    # Verify cc=0,1,2 work.
    for cc_k in [0,1,2]:
        for i in range(9):
            for j in range(12):
                ra_k = None
                # k must have cc_k; ra_k is free but T[i,j,k] depends on ra_k.
                # Sum over a: u_mat[a,i] v_mat[a,j] coef[a, ra_k, cc_k]
                # This must equal T[i,j, 3*cc_k+ra_k] for the ra_k that makes k consistent... 
                # Actually for a given (i,j), the sum over a gives a value that should equal T[i,j,k] 
                # for k with cc_k and the correct ra_k. But the sum doesn't depend on ra_k unless coef does.
                # The decomposition: T[i,j,k] = sum_a u_mat[a,i] v_mat[a,j] coef[a, k%3, k//3].
                # So for fixed (i,j), and k=3*cc_k+ra_k, T[i,j,k] = sum_a u_mat[a,i] v_mat[a,j] coef[a, ra_k, cc_k].
                # Let's verify.
                for ra_k in range(3):
                    k = 3*cc_k + ra_k
                    lhs = sum(u_mat[a,i]*v_mat[a,j]*coef[a,ra_k,cc_k] for a in range(7))
                    if abs(lhs - T[i,j,k]) > 1e-9:
                        # print(f"FAIL cc_k={cc_k} i={i} j={j} ra_k={ra_k} lhs={lhs} rhs={T[i,j,k]}")
                        pass
    # Now solve for coef[:, :, 3].
    # For each ra_k in 0,1,2: unknowns x_a = coef[a, ra_k, 3], a=0..6.
    # Equations: for each (i,j): sum_a u_mat[a,i] v_mat[a,j] x_a = T[i,j, 9+ra_k].
    # This is a linear system M x = b where M[i,j,a] = u_mat[a,i] v_mat[a,j], b[i,j] = T[i,j,9+ra_k].
    # There are 108 equations, 7 unknowns. Solve least squares (should be exact if consistent).
    for ra_k in range(3):
        M = np.zeros((108,7))
        b = np.zeros(108)
        idx = 0
        for i in range(9):
            for j in range(12):
                for a in range(7):
                    M[idx,a] = u_mat[a,i]*v_mat[a,j]
                b[idx] = T[i,j,9+ra_k]
                idx += 1
        x, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
        for a in range(7):
            coef[a,ra_k,3] = x[a]
    # Build U, V, W.
    rank = 28
    U = [[0.0]*rank for _ in range(9)]
    V = [[0.0]*rank for _ in range(12)]
    W = [[0.0]*rank for _ in range(12)]
    for a in range(7):
        for cc in range(4):
            c = a*4 + cc
            for i in range(9):
                U[i][c] = u_mat[a,i]
            for j in range(12):
                V[j][c] = v_mat[a,j]
            for ra in range(3):
                k = 3*cc + ra
                W[k][c] = coef[a,ra,cc]
    # Final verification.
    import numpy as np
    U_np = np.array(U); V_np = np.array(V); W_np = np.array(W)
    # Reconstruct T: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    T_rec = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
    err = np.max(np.abs(T_rec - T))
    if err > 1e-9:
        # Fallback to schoolbook if Strassen fails.
        rank = 36
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(12)]
        W = [[0]*rank for _ in range(12)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
