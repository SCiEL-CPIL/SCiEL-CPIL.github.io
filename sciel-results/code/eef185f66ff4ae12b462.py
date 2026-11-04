"""
Exact rank-28 tensor decomposition for 2x3 * 3x5 matrix multiplication.

Strategy:
1. Use Strassen's algorithm for the 2x2x2 sub-block (rows 0-1 of A, cols 0-1 of B,
   rows 0-1 of A, cols 0-1 of C). This uses 7 multiplications instead of 4.
2. For the remaining terms involving column 2 of A (index j=2) or columns 2-4 of B (index k=2,3,4),
   use schoolbook multiplication.

Let's count the terms:
- Strassen block (2x2x2): 7 terms.
  These handle C_{i, k} for i in {0,1}, k in {0,1} using A_{i, j} for j in {0,1} and B_{j, k} for j in {0,1}.
  The contributions from A_{i, 2} and B_{2, k} are NOT included in Strassen.

- Terms involving A_{i, 2}:
  For each i in {0,1} and k in {0,4}, we have term A_{i, 2} * B_{2, k}.
  There are 2 * 5 = 10 such terms.
  Wait, the Strassen block only covers B_{0,0}, B_{0,1}, B_{1,0}, B_{1,1}.
  It does not cover B_{2, k} for any k.
  It does not cover A_{i, 2} for any k.
  
  So we need to add all terms involving A_{i, 2} or B_{2, k} that are not already covered?
  No, the tensor is sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
  We can partition the sum based on j (the inner dimension).
  
  Sum = Sum_{j=0}^{1} (Sum_{i,k} A_{i,j} B_{j,k} C_{i,k}) + Sum_{j=2} (Sum_{i,k} A_{i,2} B_{2,k} C_{i,k})
  
  The second part (j=2) is simply 2 * 5 = 10 terms of schoolbook.
  A_{0,2}B_{2,0}C_{0,0}, ..., A_{1,2}B_{2,4}C_{1,4}.
  
  The first part is Sum_{j=0}^{1} ...
  This can be further split by k.
  For k in {0,1}, we have a 2x2x2 block (i in 0,1; j in 0,1; k in 0,1).
  We can use Strassen for this block.
  Strassen reduces the 4 terms (i,j,k in {0,1}) to 7 terms?
  Wait, the standard Strassen is for 2x2 * 2x2 -> 2x2.
  Here we have A (2x2), B (2x2), C (2x2).
  The tensor is T_{i,j,k} = delta_{j,k}? No.
  The multiplication is C = A B.
  For the sub-block, C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k}.
  This is exactly the 2x2x2 matrix multiplication tensor.
  Its rank is 7.
  
  So for the sub-block (i in 0,1, j in 0,1, k in 0,1), we use 7 terms.
  
  What about k in {2,3,4} for j in {0,1}?
  For each k in {2,3,4}, we have terms A_{i,0}B_{0,k}C_{i,k} + A_{i,1}B_{1,k}C_{i,k}.
  This is a 2x1x1 tensor? No, i goes 0..1, j goes 0..1, k is fixed.
  For fixed k, the terms are:
  A_{0,0}B_{0,k}C_{0,k} + A_{1,0}B_{0,k}C_{1,k} + A_{0,1}B_{1,k}C_{0,k} + A_{1,1}B_{1,k}C_{1,k}.
  This is 4 terms per k.
  There are 3 such k values (2,3,4). So 3 * 4 = 12 terms.
  
  And for j=2, we have 10 terms as calculated before.
  
  Total rank = 7 (Strassen for k=0,1) + 12 (Schoolbook for k=2,3,4, j=0,1) + 10 (Schoolbook for j=2)
  Total = 7 + 12 + 10 = 29.
  
  Can we do better?
  The 12 terms for k=2,3,4 and j=0,1 can be viewed as:
  For each k in {2,3,4}, we are multiplying a 2x2 matrix (A[:, 0:2]) by a 2x1 vector (B[0:2, k]) to get a 2x1 vector (C[:, k]).
  This is the rank of the 2x2x1 tensor.
  R(2,2,1) = 4?
  C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k}.
  This is a linear combination of B_{0,k} and B_{1,k}.
  The tensor is sum_{i,j} A_{i,j} B_{j,k} C_{i,k}.
  For fixed k, this is the tensor for 2x2 * 2x1.
  Is R(2,2,1) = 4?
  Yes, because the output has 2 entries, and each depends on 2 inputs from A and 1 from B?
  Actually, B is a vector of length 2.
  The map is (A, b) -> A b.
  This is bilinear.
  The rank of the tensor T_{i,j,k} (where k is fixed) is the rank of the 2x2 matrix of coefficients?
  Let's check R(2,2,1).
  Schoolbook: 4 terms.
  Can we do 3?
  C_0 = A_00 B_0 + A_10 B_1
  C_1 = A_01 B_0 + A_11 B_1
  This is equivalent to multiplying a 2x2 matrix by a 2x1 vector.
  This is the same as extracting two rows of a 2x2 matrix multiplication?
  No, it's just a linear map.
  The tensor is T = A_{00} \otimes B_0 \otimes C_0 + A_{10} \otimes B_1 \otimes C_0 + A_{01} \otimes B_0 \otimes C_1 + A_{11} \otimes B_1 \otimes C_1.
  Group by B:
  B_0 (A_{00} C_0 + A_{01} C_1) + B_1 (A_{10} C_0 + A_{11} C_1).
  The term (A_{00} C_0 + A_{01} C_1) is a rank 2 tensor in A \otimes C space?
  It is the matrix [[1, 0], [0, 1]]? No.
  A_{00} C_0 + A_{01} C_1.
  This is the tensor U = e_0 \otimes e_0 + e_1 \otimes e_1 in A \otimes C space?
  Indices of A are (row, col). Here A is 2x2.
  A_{00} is row 0, col 0. A_{01} is row 0, col 1.
  So the vector in A-space is (1, 1, 0, 0) if flattened?
  The vector in C-space is (1, 1) if flattened?
  The tensor is sum of 2 rank-1 terms in A \otimes C.
  So the total rank is 2 (for B_0) + 2 (for B_1) = 4.
  So R(2,2,1) = 4.
  
  Thus, the 12 terms cannot be reduced below 12 if we treat each k independently.
  Total 29.
  
  Is there a way to mix k=0,1 with k=2,3,4?
  Strassen works on the 2x2 block.
  If we view the whole thing as 2x3 * 3x5.
  We used Strassen on the 2x2x2 part.
  The remaining parts are "rectangular" and seem to have rank equal to schoolbook.
  
  Let's check if R(2,3,5) is indeed 28.
  If R(2,3,5) = 28, we need to save 1 more rank.
  Maybe the 12 terms (3 * 4) can be reduced?
  We have 3 columns of C (k=2,3,4).
  For each column, we compute C[:, k] = A[:, 0:2] * B[0:2, k].
  This is 3 independent 2x2x1 multiplications.
  Total rank 12.
  Can we do better for 2x2x1?
  No, rank is 4.
  
  Wait, what if we use Strassen on a different block?
  Or maybe the rank is actually 29?
  The prompt says "Find a rank-minimal exact tensor decomposition".
  And the seed score is 1.0 (rank 30).
  The inspiration says "Measured: 1 -> 1.071".
  1.071 corresponds to rank 28?
  Score = seed / value.
  If seed is rank 30, and score is 1.0, value is 30.
  If score is 1.071, value is 30 / 1.071 = 28.01.
  So rank 28 gives score 1.071.
  So R(2,3,5) = 28.
  
  How to get 28?
  We have 7 (Strassen) + 12 (Schoolbook k=2,3,4, j=0,1) + 10 (Schoolbook j=2) = 29.
  We need to save 1.
  
  Let's look at the 10 terms for j=2.
  A_{i,2} B_{2,k} C_{i,k}.
  This is 2x1 * 1x5 -> 2x5.
  Rank is 10?
  C_{i,k} = A_{i,2} B_{2,k}.
  This is an outer product.
  The tensor is sum_{i,k} A_{i,2} B_{2,k} C_{i,k}.
  This is the rank of the 2x5 identity? No.
  It is the tensor for the map (a, b) -> a b^T.
  This is the "diagonal" tensor?
  Rank is 2*5 = 10?
  Yes, because each C_{i,k} depends on distinct A_{i,2} and B_{2,k}.
  Actually, A_{i,2} is shared for all k. B_{2,k} is shared for all i.
  But the pairs (i,k) are distinct.
  Can we use fewer than 10?
  T = sum_{i,k} A_i B_k C_{ik}.
  This is the tensor of the identity map from R^{2x5} to R^{2x5}? No.
  It is the tensor for the bilinear map (u, v) -> u v^T.
  The rank of this tensor is n*m.
  Proof: The tensor is sum_{i=1}^n sum_{j=1}^m e_i \otimes e_j \otimes e_{ij}.
  These are n*m linearly independent terms?
  Yes, because the C-index (i,j) is unique for each term.
  So rank is 10.
  
  So where is the saving?
  Maybe in the 12 terms?
  We have 3 columns.
  Maybe we can combine them?
  Let's check the rank of 2x2x3?
  No, we have 2x2 (A) * 2x3 (B) -> 2x3 (C).
  This is the tensor for 2x2 * 2x3.
  Schoolbook rank: 2*2*3 = 12.
  Can we do better?
  R(2,2,3).
  We can split B into B1 (2x1) and B2 (2x1) and B3 (2x1).
  R(2,2,1) = 4.
  So R(2,2,3) <= 4+4+4 = 12.
  Is R(2,2,3) < 12?
  Known results: R(2,2,2) = 7.
  R(2,2,3) ?
  If we view it as 2x2 * 2x3.
  We can use Strassen for the first 2 columns?
  Strassen is for 2x2 * 2x2.
  If we use Strassen for cols 0,1 of B, we get 7 terms for C[:, 0:2].
  Then 1 column left (col 2 of B, which is col 2 of C).
  That is 2x2 * 2x1. Rank 4.
  Total 7 + 4 = 11.
  
  Ah!
  So for the part j=0,1 (A is 2x2, B is 2x5, C is 2x5):
  We can split B into B_left (2x2, cols 0,1) and B_right (2x3, cols 2,3,4).
  
  Part 1: A (2x2) * B_left (2x2) -> C_left (2x2).
  Use Strassen. Rank 7.
  
  Part 2: A (2x2) * B_right (2x3) -> C_right (2x3).
  We can split B_right into B_col2 (2x1), B_col3 (2x1), B_col4 (2x1).
  Each is 2x2 * 2x1. Rank 4.
  Total 12.
  
  Total for j=0,1 part: 7 + 12 = 19.
  
  Part 3: A_col2 (2x1) * B (3x5) -> C (2x5).
  This is 2x1 * 3x5.
  Wait, A is 2x3. A_col2 is the 3rd column (index 2).
  So A_sub is 2x1.
  B is 3x5.
  The multiplication is C = A_col2 * B_row2?
  No.
  The full multiplication is C = A B.
  C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k} + A_{i,2}B_{2,k}.
  
  We handled A_{i,0}B_{0,k} + A_{i,1}B_{1,k} in Part 1 and 2.
  Part 1 handled k=0,1.
  Part 2 handled k=2,3,4.
  
  Now we have the term A_{i,2}B_{2,k}.
  This is a 2x1 vector (A_col2) multiplied by a 1x5 vector (B_row2)?
  No, B_{2,k} is the 3rd row of B.
  So we are computing the outer product of A_col2 (2x1) and B_row2 (1x5).
  Result is 2x5.
  Rank of this tensor is 2*5 = 10.
  
  Total rank = 7 (Strassen) + 12 (Schoolbook for k=2,3,4) + 10 (Outer product) = 29.
  
  Still 29.
  Where is the 28?
  
  Maybe R(2,2,3) is 11?
  If R(2,2,3) = 11, then Part 2 is 11.
  Total = 7 + 11 + 10 = 28.
  
  Is R(2,2,3) = 11?
  We established R(2,2,2)=7.
  R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
  Can it be 10?
  Lower bound for R(2,2,3)?
  The output is 2x3 = 6 entries.
  The rank must be at least 6.
  Is it 11?
  Actually, there is a known result that R(2,2,n) = 4n - 3 for n>=2?
  If so, R(2,2,3) = 4*3 - 3 = 9?
  Let's check.
  If R(2,2,3) = 9, then Total = 7 + 9 + 10 = 26.
  
  Let's verify R(2,2,3).
  Multiplying 2x2 by 2x3.
  C1 = A11 B11 + A12 B12
  C2 = A21 B11 + A22 B12
  C3 = A11 B13 + A12 B13
  C4 = A21 B13 + A22 B13
  (Indices: C1 is C00, C2 is C10, C3 is C01, C4 is C11? No, C is 2x3.
  C00, C01, C02, C10, C11, C12.
  C00 = A00 B00 + A01 B10
  C01 = A00 B01 + A01 B11
  C02 = A00 B02 + A01 B12
  C10 = A10 B00 + A11 B10
  C11 = A10 B01 + A11 B11
  C12 = A10 B02 + A11 B12
  
  This is 3 columns.
  Each column is 2x2 * 2x1.
  Rank 4 each. Total 12.
  
  Can we do better?
  Strassen for the first 2 columns (2x2 * 2x2) gives 7 terms for C00, C01, C10, C11.
  Then 1 column left (C02, C12).
  This is 2x2 * 2x1. Rank 4.
  Total 11.
  
  So R(2,2,3) <= 11.
  Is it 11?
  If R(2,2,3) = 11, then total rank is 7 + 11 + 10 = 28.
  
  So the plan is:
  1. Strassen for A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]. (7 terms)
  2. Schoolbook/Strassen hybrid for A[:, 0:2] * B[0:2, 2:5] -> C[:, 2:5].
     Split B[0:2, 2:5] into B[0:2, 2:4] (2x2) and B[0:2, 4] (2x1).
     Use Strassen for B[0:2, 2:4] -> C[:, 2:4]. (7 terms)
     Use Schoolbook for B[0:2, 4] -> C[:, 4]. (4 terms)
     Total 11 terms.
  3. Schoolbook for A[:, 2] * B[2, :] -> C[:, :]. (10 terms)
  
  Total 7 + 11 + 10 = 28.
  
  Let's implement this.
  
  Indices:
  A is 2x3. Flattened: A[0]=A00, A[1]=A01, A[2]=A02, A[3]=A10, A[4]=A11, A[5]=A12.
  B is 3x5. Flattened: B[0]=B00, B[1]=B01, B[2]=B02, B[3]=B03, B[4]=B04,
                         B[5]=B10, B[6]=B11, B[7]=B12, B[8]=B13, B[9]=B14,
                         B[10]=B20, B[11]=B21, B[12]=B22, B[13]=B23, B[14]=B24.
  C is 2x5. Flattened: C[0]=C00, C[1]=C01, C[2]=C02, C[3]=C03, C[4]=C04,
                         C[5]=C10, C[6]=C11, C[7]=C12, C[8]=C13, C[9]=C14.
  Note: W index is k*n + i.
  C_{i,k} -> W[k*2 + i].
  C00 -> W[0], C01 -> W[2], C02 -> W[4], C03 -> W[6], C04 -> W[8].
  C10 -> W[1], C11 -> W[3], C12 -> W[5], C13 -> W[7], C14 -> W[9].
  
  Step 1: Strassen for A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2].
  A_sub: A00, A01, A10, A11. (Indices 0,1,3,4 in flat A)
  B_sub: B00, B01, B10, B11. (Indices 0,1,5,6 in flat B)
  C_sub: C00, C01, C10, C11. (Indices 0,2,1,3 in flat W)
  
  Strassen terms:
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  M3 = A00(B01-B11)
  M4 = A11(B10-B00)
  M5 = (A00+A01)B11
  M6 = (A10-A00)(B00+B01)
  M7 = (A01-A11)(B10+B11)
  
  C00 = M1 + M4 - M5 + M7
  C01 = M3 + M5
  C10 = M2 + M4
  C11 = M1 - M2 + M3 + M6
  
  Step 2: Strassen for A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4].
  A_sub: Same as above.
  B_sub: B02, B03, B12, B13. (Indices 2,3,7,8 in flat B)
  C_sub: C02, C03, C12, C13. (Indices 4,6,5,7 in flat W)
  
  Strassen terms (reuse A coefficients, new B coefficients):
  M8 = (A00+A11)(B02+B12)
  M9 = (A10+A11)B02
  M10 = A00(B03-B13)
  M11 = A11(B12-B02)
  M12 = (A00+A01)B13
  M13 = (A10-A00)(B02+B03)
  M14 = (A01-A11)(B12+B13)
  
  C02 = M8 + M11 - M12 + M14
  C03 = M10 + M12
  C12 = M9 + M11
  C13 = M8 - M9 + M10 + M13
  
  Step 3: Schoolbook for A[:, 0:2] * B[0:2, 4] -> C[:, 4].
  A_sub: A00, A01, A10, A11.
  B_sub: B04, B14. (Indices 4,9 in flat B)
  C_sub: C04, C14. (Indices 8,9 in flat W)
  
  C04 = A00 B04 + A01 B14
  C14 = A10 B04 + A11 B14
  
  Terms:
  M15 = A00 B04
  M16 = A01 B14
  M17 = A10 B04
  M18 = A11 B14
  
  C04 = M15 + M16
  C14 = M17 + M18
  
  Step 4: Schoolbook for A[:, 2] * B[2, :] -> C[:, :].
  A_sub: A02, A12. (Indices 2,5 in flat A)
  B_sub: B20..B24. (Indices 10..14 in flat B)
  C_sub: C00..C04, C10..C14.
  
  C00 += A02 B20
  C01 += A02 B21
  C02 += A02 B22
  C03 += A02 B23
  C04 += A02 B24
  C10 += A12 B20
  C11 += A12 B21
  C12 += A12 B22
  C13 += A12 B23
  C14 += A12 B24
  
  Terms:
  M19 = A02 B20
  M20 = A02 B21
  M21 = A02 B22
  M22 = A02 B23
  M23 = A02 B24
  M24 = A12 B20
  M25 = A12 B21
  M26 = A12 B22
  M27 = A12 B23
  M28 = A12 B24
  
  Total 28 terms.
  
  Now we just need to construct U, V, W matrices.
  U has shape (6, 28).
  V has shape (15, 28).
  W has shape (10, 28).
  
  We will fill them column by column.
  
  Helper function to add a term:
  def add_term(U, V, W, col, a_idx, b_idx, c_idx, sign=1):
      U[a_idx][col] += sign
      V[b_idx][col] += sign
      W[c_idx][col] += sign
      
  But Strassen terms involve sums/differences.
  We need to be careful with the coefficients.
  
  Let's list the terms explicitly with their U, V, W contributions.
  
  Term 1: M1 = (A00+A11)(B00+B11)
  U: A00 (idx 0) +1, A11 (idx 4) +1
  V: B00 (idx 0) +1, B11 (idx 6) +1
  Wait, this is not a rank-1 term.
  M1 is a product of sums.
  (A00+A11)(B00+B11) = A00 B00 + A00 B11 + A11 B00 + A11 B11.
  This is 4 rank-1 terms.
  Strassen replaces 4 terms with 1 multiplication, but in the tensor decomposition,
  the "multiplication" is the rank-1 term u \otimes v \otimes w.
  The coefficients in u and v can be sums.
  So u = e_0 + e_4. v = e_0 + e_6. w = ...
  The term is u \otimes v \otimes w.
  This contributes u_i v_j w_k to T_{i,j,k}.
  
  So for M1:
  u = e_0 + e_4
  v = e_0 + e_6
  w = ?
  M1 contributes to C00 (+1) and C11 (+1).
  So w = e_C00 + e_C11.
  C00 is W[0], C11 is W[3].
  So w = e_0 + e_3.
  
  Let's verify.
  Contribution to T_{A00, B00, C00} = 1*1*1 = 1. Correct.
  Contribution to T_{A00, B11, C00} = 1*1*1 = 1. Correct.
  Contribution to T_{A11, B00, C00} = 1*1*1 = 1. Correct.
  Contribution to T_{A11, B11, C00} = 1*1*1 = 1. Correct.
  Contribution to T_{A00, B00, C11} = 1*1*1 = 1. Correct.
  ...
  
  So we just need to define the vectors u, v, w for each of the 28 terms.
  
  Let's define the basis indices.
  A: 0:A00, 1:A01, 2:A02, 3:A10, 4:A11, 5:A12
  B: 0:B00, 1:B01, 2:B02, 3:B03, 4:B04, 5:B10, 6:B11, 7:B12, 8:B13, 9:B14, 10:B20, 11:B21, 12:B22, 13:B23, 14:B24
  W: 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12, 6:C03, 7:C13, 8:C04, 9:C14
  
  Term 1 (M1):
  u: [1, 0, 0, 0, 1, 0] (A00, A11)
  v: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] (B00, B11)
  w: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] (C00, C11)
  
  Term 2 (M2): (A10+A11)B00
  u: [0, 0, 0, 1, 1, 0] (A10, A11)
  v: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (B00)
  w: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] (C10)
  
  Term 3 (M3): A00(B01-B11)
  u: [1, 0, 0, 0, 0, 0] (A00)
  v: [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0] (B01, -B11)
  w: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (C01)
  
  Term 4 (M4): A11(B10-B00)
  u: [0, 0, 0, 0, 1, 0] (A11)
  v: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] - [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  w: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] (C00, C10)
  
  Term 5 (M5): (A00+A01)B11
  u: [1, 1, 0, 0, 0, 0] (A00, A01)
  v: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] (B11)
  w: [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0] (-C00, C01)
  
  Term 6 (M6): (A10-A00)(B00+B01)
  u: [0, 0, 0, 1, 0, 0] - [1, 0, 0, 0, 0, 0] = [-1, 0, 0, 1, 0, 0]
  v: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (B00, B01)
  w: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (C11)
  
  Term 7 (M7): (A01-A11)(B10+B11)
  u: [0, 1, 0, 0, -1, 0]
  v: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] (B10, B11)
  w: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] (C00)
  
  Term 8 (M8): (A00+A11)(B02+B12)
  u: [1, 0, 0, 0, 1, 0]
  v: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (B02, B12)
  w: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (C02)
  
  Term 9 (M9): (A10+A11)B02
  u: [0, 0, 0, 1, 1, 0]
  v: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (B02)
  w: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] (C12)
  
  Term 10 (M10): A00(B03-B13)
  u: [1, 0, 0, 0, 0, 0]
  v: [0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0] (B03, -B13)
  w: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] (C03)
  
  Term 11 (M11): A11(B12-B02)
  u: [0, 0, 0, 0, 1, 0]
  v: [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  w: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0] (C02, C12)
  
  Term 12 (M12): (A00+A01)B13
  u: [1, 1, 0, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (B13)
  w: [0, 0, 0, 0, -1, 0, 1, 0, 0, 0] (-C02, C03)
  
  Term 13 (M13): (A10-A00)(B02+B03)
  u: [-1, 0, 0, 1, 0, 0]
  v: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  w: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] (C13)
  
  Term 14 (M14): (A01-A11)(B12+B13)
  u: [0, 1, 0, 0, -1, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  w: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (C02)
  
  Term 15 (M15): A00 B04
  u: [1, 0, 0, 0, 0, 0]
  v: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (B04)
  w: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] (C04)
  
  Term 16 (M16): A01 B14
  u: [0, 1, 0, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (B14)
  w: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] (C04)
  
  Term 17 (M17): A10 B04
  u: [0, 0, 0, 1, 0, 0]
  v: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  w: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] (C14)
  
  Term 18 (M18): A11 B14
  u: [0, 0, 0, 0, 1, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  w: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] (C14)
  
  Term 19 (M19): A02 B20
  u: [0, 0, 1, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] (B20)
  w: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] (C00)
  
  Term 20 (M20): A02 B21
  u: [0, 0, 1, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] (B21)
  w: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (C01)
  
  Term 21 (M21): A02 B22
  u: [0, 0, 1, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] (B22)
  w: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (C02)
  
  Term 22 (M22): A02 B23
  u: [0, 0, 1, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] (B23)
  w: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] (C03)
  
  Term 23 (M23): A02 B24
  u: [0, 0, 1, 0, 0, 0]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] (B24)
  w: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] (C04)
  
  Term 24 (M24): A12 B20
  u: [0, 0, 0, 0, 0, 1]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  w: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] (C10)
  
  Term 25 (M25): A12 B21
  u: [0, 0, 0, 0, 0, 1]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  w: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (C11)
  
  Term 26 (M26): A12 B22
  u: [0, 0, 0, 0, 0, 1]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  w: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] (C12)
  
  Term 27 (M27): A12 B23
  u: [0, 0, 0, 0, 0, 1]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  w: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] (C13)
  
  Term 28 (M28): A12 B24
  u: [0, 0, 0, 0, 0, 1]
  v: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  w: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] (C14)
  
  We can construct the matrices by summing these contributions.
  
  U is 6x28.
  V is 15x28.
  W is 10x28.
  
  We will create lists of the vectors for each term and then build the matrices.
"""

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    
    # Dimensions
    dim_U = n * m  # 6
    dim_V = m * p  # 15
    dim_W = p * n  # 10
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    # Helper to add a vector to a matrix column
    def add_vec(mat, vec, col):
        for i in range(len(vec)):
            mat[i][col] += vec[i]
            
    # Define the 28 terms
    # Each term is (u_vec, v_vec, w_vec)
    
    # Indices
    # A: 0:A00, 1:A01, 2:A02, 3:A10, 4:A11, 5:A12
    # B: 0:B00, 1:B01, 2:B02, 3:B03, 4:B04, 5:B10, 6:B11, 7:B12, 8:B13, 9:B14, 10:B20, 11:B21, 12:B22, 13:B23, 14:B24
    # W: 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12, 6:C03, 7:C13, 8:C04, 9:C14
    
    terms = []
    
    # Term 1: M1
    u = [1, 0, 0, 0, 1, 0]
    v = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 2: M2
    u = [0, 0, 0, 1, 1, 0]
    v = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 3: M3
    u = [1, 0, 0, 0, 0, 0]
    v = [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 4: M4
    u = [0, 0, 0, 0, 1, 0]
    v = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 5: M5
    u = [1, 1, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 6: M6
    u = [-1, 0, 0, 1, 0, 0]
    v = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 7: M7
    u = [0, 1, 0, 0, -1, 0]
    v = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 8: M8
    u = [1, 0, 0, 0, 1, 0]
    v = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 9: M9
    u = [0, 0, 0, 1, 1, 0]
    v = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 10: M10
    u = [1, 0, 0, 0, 0, 0]
    v = [0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 11: M11
    u = [0, 0, 0, 0, 1, 0]
    v = [0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 12: M12
    u = [1, 1, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, -1, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 13: M13
    u = [-1, 0, 0, 1, 0, 0]
    v = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    terms.append((u, v, w))
    
    # Term 14: M14
    u = [0, 1, 0, 0, -1, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 15: M15
    u = [1, 0, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # Term 16: M16
    u = [0, 1, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # Term 17: M17
    u = [0, 0, 0, 1, 0, 0]
    v = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    # Term 18: M18
    u = [0, 0, 0, 0, 1, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    # Term 19: M19
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    w = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 20: M20
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    w = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 21: M21
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    w = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 22: M22
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    w = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 23: M23
    u = [0, 0, 1, 0, 0, 0]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    w = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    terms.append((u, v, w))
    
    # Term 24: M24
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    w = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 25: M25
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    w = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 26: M26
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    w = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    terms.append((u, v, w))
    
    # Term 27: M27
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    w = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    terms.append((u, v, w))
    
    # Term 28: M28
    u = [0, 0, 0, 0, 0, 1]
    v = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    terms.append((u, v, w))
    
    # Fill matrices
    for col in range(rank):
        u, v, w = terms[col]
        add_vec(U, u, col)
        add_vec(V, v, col)
        add_vec(W, w, col)
        
    return {"U": U, "V": V, "W": W}

