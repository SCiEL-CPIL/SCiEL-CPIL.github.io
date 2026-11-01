"""Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy: Use Strassen's 7-term algorithm for 2x2 matrix multiplication.
View the 2x4 * 4x8 product as four independent 2x2 block multiplications:
  C[0,0] = A[0,0:2] * B[0:2,0:2]
  C[0,1] = A[0,0:2] * B[0:2,2:4]
  C[1,0] = A[1,0:2] * B[2:4,0:2]
  C[1,1] = A[1,0:2] * B[2:4,4:6]
Wait, that's not right. Let me reconsider the block structure.

Actually, 2x4 * 4x8:
  A is 2x4, B is 4x8, C is 2x8.
  
  We can partition B into four 4x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
  Then C[:, 0:2] = A * B[:, 0:2], etc.
  
  Each A * B[:, 2j:2j+2] is a 2x4 * 4x2 = 2x2 product.
  
  So we have four independent 2x2 multiplications, each of rank 7 (Strassen).
  Total rank = 4 * 7 = 28.
  
  Wait, but the inner dimension is 4, which can be split into two 2-dim blocks.
  So A * B[:, 2j:2j+2] where A is 2x4 and B block is 4x2.
  
  Actually, let's think of it differently. The 4x8 matrix B can be viewed as
  four 4x2 blocks. The 2x4 matrix A times each 4x2 block gives a 2x2 result.
  
  For each such 2x2 product: A (2x4) * B_block (4x2) = C_block (2x2)
  
  But A is 2x4, not 2x2. So we can't directly use Strassen on 2x2.
  
  Let me reconsider. The inner dimension m=4. We can split the summation over j
  into two parts: j=0,1 and j=2,3.
  
  C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
  
  Each partial sum is a 2x2 * 2x8 product... no, that's not right either.
  
  Let me think again. The standard approach for rank reduction uses the fact that
  matrix multiplication tensor has a lower rank than the trivial decomposition.
  
  For 2x4 * 4x8, the trivial rank is 2*4*8 = 64.
  
  One approach: view this as multiplying a 2x4 by 4x8. We can reshape A as 2x(2x2)
  and B as (2x2)x8, but that doesn't directly help.
  
  Alternative: Use the fact that 2x4 * 4x8 can be decomposed using Strassen-like
  techniques. 
  
  Actually, let's use a different approach. The tensor for 2x4*4x8 multiplication
  can be viewed as a composition of smaller tensors. 
  
  Consider: C = A * B where A is 2x4, B is 4x8.
  
  We can write A as a 2x2 block matrix where each block is 2x2:
    A = [A00 A01]
        [A10 A11]
  where Aij is 2x2.
  
  Similarly, B can be written as a 4x8 matrix, which is a 2x2 block matrix where
  each block is 2x4:
    B = [B00 B01]
        [B10 B11]
  where Bij is 2x4.
  
  Then C = A * B is 2x8:
    C = [C00 C01]
        [C10 C11]
  where Cij is 2x4.
  
  C00 = A00*B00 + A01*B10
  C01 = A00*B01 + A01*B11
  C10 = A10*B00 + A11*B10
  C11 = A10*B01 + A11*B11
  
  Each term like A00*B00 is a 2x2 * 2x4 = 2x4 product.
  
  This is getting complicated. Let me try a more direct approach.
  
  The key insight: the matrix multiplication tensor for (n,m,p) has rank at most
  n*p*ceil(m/2) if we can pair up the inner dimension. But that gives 2*8*2 = 32,
  which is already better than 64.
  
  Wait, the current program has rank 64, which is the trivial bound n*m*p.
  
  Let me try to achieve rank 32 by pairing the inner dimension.
  
  Actually, re-reading the current program, it says "Total rank = n * p * (m/2) * 2 = n * p * m = 64"
  which is just the trivial bound. The pairing doesn't reduce the rank in that formulation.
  
  To actually reduce the rank, we need to use Strassen-like techniques.
  
  For the 2x4 * 4x8 case, one approach is to use the fact that we can apply
  Strassen to the 2x2 subproblems.
  
  Let me try: view the multiplication as four 2x2 * 2x2 products after appropriate
  reshaping.
  
  A (2x4) can be seen as two 2x2 matrices side by side: A = [A_L | A_R]
  B (4x8) can be seen as a 2x2 block matrix with 2x4 blocks: B = [[B_TL, B_TR], [B_BL, B_BR]]
  
  Then C = A * B = [A_L | A_R] * [[B_TL, B_TR], [B_BL, B_BR]]
               = [A_L*B_TL + A_R*B_BL, A_L*B_TR + A_R*B_BR]
  
  Each term like A_L*B_TL is 2x2 * 2x4 = 2x4.
  
  This is still not directly 2x2.
  
  Let me try yet another approach. The simplest rank reduction comes from using
  the Strassen algorithm on the 2x2 blocks if we can formulate it that way.
  
  Actually, I think the cleanest approach is:
  
  The 2x4 * 4x8 multiplication can be viewed as computing 16 output entries, each
  being a sum of 4 products. The trivial decomposition has 64 terms.
  
  To reduce the rank, we can use the fact that certain combinations of terms can
  be shared. This is essentially what Strassen does for 2x2.
  
  For a 2x2 * 2x2 multiplication, Strassen uses 7 multiplications instead of 8.
  
  For our case, we can partition the problem into four 2x2 * 2x2 subproblems:
  - C[0,0:2] depends on A[0,0:2] and B[0:2,0:2] (but B is 4x8, so this isn't quite right)
  
  Let me be more careful. C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k].
  
  We can split this as:
  C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
  
  The first sum involves A[0:2, 0:2] (which is 2x2) and B[0:2, :] (which is 2x8).
  The second sum involves A[0:2, 2:4] (which is 2x2) and B[2:4, :] (which is 2x8).
  
  So C = A_top * B_top + A_bottom * B_bottom
  where A_top = A[0:2, 0:2] (2x2), B_top = B[0:2, :] (2x8)
        A_bottom = A[0:2, 2:4] (2x2), B_bottom = B[2:4, :] (2x8)
  
  This is a sum of two 2x2 * 2x8 products.
  
  Each 2x2 * 2x8 product can be viewed as four 2x2 * 2x2 products.
  
  For each 2x2 * 2x2 product, we can use Strassen (rank 7).
  
  So total rank = 2 (for the two sums) * 4 (2x2 blocks) * 7 (Strassen) = 56.
  
  That's better than 64!
  
  Let me implement this.
  
  For a 2x2 * 2x2 product X*Y = Z:
  Strassen's 7 multiplications:
  M1 = (X00+X11)*(Y00+Y11)
  M2 = (X10+X11)*Y00
  M3 = X00*(Y01-Y11)
  M4 = X11*(Y10-Y00)
  M5 = (X00+X01)*Y11
  M6 = (X10-X00)*(Y00+Y01)
  M7 = (X01-X11)*(Y10+Y11)
  
  Z00 = M1 - M4 + M5 + M6
  Z01 = M3 + M5
  Z10 = M2 + M4
  Z11 = M1 - M2 + M3 + M7
  
  Now, for the full 2x4 * 4x8 problem:
  
  C = A_top * B_top + A_bottom * B_bottom
  
  Where:
  A_top = A[0:2, 0:2] = [[A00, A01], [A10, A11]] (using A[i,j] notation)
  B_top = B[0:2, 0:8] = [[B00, B01, B02, B03, B04, B05, B06, B07],
                         [B10, B11, B12, B13, B14, B15, B16, B17]]
  
  A_bottom = A[0:2, 2:4] = [[A02, A03], [A12, A13]]
  B_bottom = B[2:4, 0:8] = [[B20, B21, B22, B23, B24, B25, B26, B27],
                            [B30, B31, B32, B33, B34, B35, B36, B37]]
  
  A_top * B_top is 2x8. Let's call its (i,k) entry (A_top*B_top)[i,k].
  This equals sum_{j=0}^{1} A_top[i,j] * B_top[j,k] = sum_{j=0}^{1} A[i,j] * B[j,k].
  
  Similarly for A_bottom * B_bottom.
  
  Now, A_top * B_top can be viewed as:
  For each pair of columns (2k, 2k+1) in B_top, we have a 2x2 * 2x2 product.
  
  B_top[:, 2k:2k+2] is 2x2: [[B[0,2k], B[0,2k+1]], [B[1,2k], B[1,2k+1]]]
  
  So (A_top * B_top)[:, 2k:2k+2] = A_top * B_top[:, 2k:2k+2] is a 2x2 product.
  
  Similarly for A_bottom * B_bottom.
  
  So we have 2 (top/bottom) * 4 (column pairs) = 8 independent 2x2 * 2x2 products.
  
  Each uses 7 Strassen multiplications, so total rank = 8 * 7 = 56.
  
  Let me implement this.
  
  The U matrix has shape (n*m, rank) = (8, 56).
  U[i*m + j, col] corresponds to coefficient of A[i,j] in the col-th rank-1 term.
  
  The V matrix has shape (m*p, rank) = (32, 56).
  V[j*p + k, col] corresponds to coefficient of B[j,k] in the col-th rank-1 term.
  
  The W matrix has shape (p*n, rank) = (16, 56).
  W[k*n + i, col] corresponds to coefficient of C[i,k] in the col-th rank-1 term.
  
  Wait, the problem says W[k*n+i, r]. So W is indexed by (k*n+i) for row, and r for column.
  C[i,k] is at position k*n + i in W's row index.
  
  Let me verify: n=2, p=8. p*n = 16. W has 16 rows.
  C[i,k] where i in 0..1, k in 0..7. Index = k*n + i = 2k + i.
  So C[0,0] -> 0, C[1,0] -> 1, C[0,1] -> 2, C[1,1] -> 3, etc.
  
  Now, for each of the 8 Strassen subproblems, we have 7 rank-1 terms.
  
  Let me define the 8 subproblems:
  For s in {0, 1} (0=top, 1=bottom):
    For t in {0, 1, 2, 3} (column pair index):
      X = A[0:2, 2*s:2*s+2]  (2x2)
      Y = B[2*s:2*s+2, 2*t:2*t+2]  (2x2)
      Z = C[0:2, 2*t:2*t+2]  (2x2)
      
      Strassen on X*Y = Z.
  
  For each Strassen term, we need to express it as a rank-1 tensor.
  
  Let me work out the first Strassen term for subproblem (s, t):
  M1 = (X00+X11)*(Y00+Y11)
  
  X00 = A[0, 2*s], X11 = A[1, 2*s+1]
  Y00 = B[2*s, 2*t], Y11 = B[2*s+1, 2*t+1]
  
  M1 = (A[0, 2*s] + A[1, 2*s+1]) * (B[2*s, 2*t] + B[2*s+1, 2*t+1])
  
  This expands to:
  A[0, 2*s]*B[2*s, 2*t] + A[0, 2*s]*B[2*s+1, 2*t+1] + A[1, 2*s+1]*B[2*s, 2*t] + A[1, 2*s+1]*B[2*s+1, 2*t+1]
  
  In the tensor decomposition, this rank-1 term contributes to C[i,k] with coefficient
  determined by how M1 appears in the Strassen formulas.
  
  For subproblem (s, t), Z is C[0:2, 2*t:2*t+2]:
  Z00 = C[0, 2*t]
  Z01 = C[0, 2*t+1]
  Z10 = C[1, 2*t]
  Z11 = C[1, 2*t+1]
  
  M1 appears in:
  Z00 with coefficient +1
  Z11 with coefficient +1
  
  So the rank-1 term for M1 contributes:
  To C[0, 2*t]: +1
  To C[1, 2*t+1]: +1
  
  In W: 
  C[0, 2*t] is at row (2*t)*n + 0 = 4t
  C[1, 2*t+1] is at row (2*t+1)*n + 1 = 4t + 3
  
  In U:
  A[0, 2*s] is at row 0*m + 2*s = 2s
  A[1, 2*s+1] is at row 1*m + 2*s+1 = 4 + 2*s + 1 = 5 + 2*s
  
  In V:
  B[2*s, 2*t] is at row 2*s*p + 2*t = 16s + 2t
  B[2*s+1, 2*t+1] is at row (2*s+1)*p + 2*t+1 = 16s + 16 + 2t + 1 = 17s + 17 + 2t... wait, p=8, so (2s+1)*8 + 2t+1 = 16s + 8 + 2t + 1 = 16s + 2t + 9
  
  Let me be more careful with indices.
  
  U has rows indexed by i*m + j, where i in 0..1, j in 0..3.
  So row = 4*i + j.
  
  V has rows indexed by j*p + k, where j in 0..3, k in 0..7.
  So row = 8*j + k.
  
  W has rows indexed by k*n + i, where k in 0..7, i in 0..1.
  So row = 2*k + i.
  
  Now, for subproblem (s, t), the 7 Strassen terms:
  
  Let X = [[A[0, 2s], A[0, 2s+1]], [A[1, 2s], A[1, 2s+1]]]
  Let Y = [[B[2s, 2t], B[2s, 2t+1]], [B[2s+1, 2t], B[2s+1, 2t+1]]]
  
  M1 = (X00+X11)*(Y00+Y11) = (A[0,2s]+A[1,2s+1])*(B[2s,2t]+B[2s+1,2t+1])
  M2 = (X10+X11)*Y00 = (A[1,2s]+A[1,2s+1])*B[2s,2t]
  M3 = X00*(Y01-Y11) = A[0,2s]*(B[2s,2t+1]-B[2s+1,2t+1])
  M4 = X11*(Y10-Y00) = A[1,2s+1]*(B[2s+1,2t]-B[2s,2t])
  M5 = (X00+X01)*Y11 = (A[0,2s]+A[0,2s+1])*B[2s+1,2t+1]
  M6 = (X10-X00)*(Y00+Y01) = (A[1,2s]-A[0,2s])*(B[2s,2t]+B[2s,2t+1])
  M7 = (X01-X11)*(Y10+Y11) = (A[0,2s+1]-A[1,2s+1])*(B[2s+1,2t]+B[2s+1,2t+1])
  
  Z00 = C[0,2t] = M1 - M4 + M5 + M6
  Z01 = C[0,2t+1] = M3 + M5
  Z10 = C[1,2t] = M2 + M4
  Z11 = C[1,2t+1] = M1 - M2 + M3 + M7
  
  Now, each Mj is a product of a linear form in A and a linear form in B.
  Each rank-1 term in the decomposition corresponds to one such product.
  
  For the rank-1 term corresponding to Mj in subproblem (s,t):
  - The A-linear form gives the U coefficients
  - The B-linear form gives the V coefficients
  - The contribution to C gives the W coefficients
  
  Let me implement this.
  
  rank = 8 * 7 = 56
  
  I'll iterate over s in {0,1}, t in {0,1,2,3}, and for each, add 7 rank-1 terms.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56  # 8 subproblems * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 16 rows
    
    col = 0
    for s in range(2):  # 0=top, 1=bottom
        for t in range(4):  # column pair index
            # X = A[0:2, 2s:2s+2], Y = B[2s:2s+2, 2t:2t+2]
            # X00=A[0,2s], X01=A[0,2s+1], X10=A[1,2s], X11=A[1,2s+1]
            # Y00=B[2s,2t], Y01=B[2s,2t+1], Y10=B[2s+1,2t], Y11=B[2s+1,2t+1]
            
            # M1 = (X00+X11)*(Y00+Y11)
            # U: A[0,2s] + A[1,2s+1]
            U[0*m + 2*s][col] = 1
            U[1*m + 2*s+1][col] = 1
            # V: B[2s,2t] + B[2s+1,2t+1]
            V[2*s*p + 2*t][col] = 1
            V[(2*s+1)*p + 2*t+1][col] = 1
            # W: Z00=M1..., Z11=M1... => C[0,2t]+=1, C[1,2t+1]+=1
            W[2*t*n + 0][col] = 1
            W[(2*t+1)*n + 1][col] = 1
            col += 1
            
            # M2 = (X10+X11)*Y00
            # U: A[1,2s] + A[1,2s+1]
            U[1*m + 2*s][col] = 1
            U[1*m + 2*s+1][col] = 1
            # V: B[2s,2t]
            V[2*s*p + 2*t][col] = 1
            # W: Z10=M2+... => C[1,2t]+=1; Z11=-M2+... => C[1,2t+1]+=-1
            W[2*t*n + 1][col] = 1
            W[(2*t+1)*n + 1][col] = -1
            col += 1
            
            # M3 = X00*(Y01-Y11)
            # U: A[0,2s]
            U[0*m + 2*s][col] = 1
            # V: B[2s,2t+1] - B[2s+1,2t+1]
            V[2*s*p + 2*t+1][col] = 1
            V[(2*s+1)*p + 2*t+1][col] = -1
            # W: Z01=M3+... => C[0,2t+1]+=1; Z11=M3+... => C[1,2t+1]+=1
            W[2*t*n + 1][col] = 1
            W[(2*t+1)*n + 1][col] = 1
            col += 1
            
            # M4 = X11*(Y10-Y00)
            # U: A[1,2s+1]
            U[1*m + 2*s+1][col] = 1
            # V: B[2s+1,2t] - B[2s,2t]
            V[(2*s+1)*p + 2*t][col] = 1
            V[2*s*p + 2*t][col] = -1
            # W: Z00=-M4+... => C[0,2t]+=-1; Z10=M4+... => C[1,2t]+=1
            W[2*t*n + 0][col] = -1
            W[2*t*n + 1][col] = 1
            col += 1
            
            # M5 = (X00+X01)*Y11
            # U: A[0,2s] + A[0,2s+1]
            U[0*m + 2*s][col] = 1
            U[0*m + 2*s+1][col] = 1
            # V: B[2s+1,2t+1]
            V[(2*s+1)*p + 2*t+1][col] = 1
            # W: Z00=M5+... => C[0,2t]+=1; Z01=M5+... => C[0,2t+1]+=1
            W[2*t*n + 0][col] = 1
            W[2*t*n + 1][col] = 1
            col += 1
            
            # M6 = (X10-X00)*(Y00+Y01)
            # U: A[1,2s] - A[0,2s]
            U[1*m + 2*s][col] = 1
            U[0*m + 2*s][col] = -1
            # V: B[2s,2t] + B[2s,2t+1]
            V[2*s*p + 2*t][col] = 1
            V[2*s*p + 2*t+1][col] = 1
            # W: Z00=M6+... => C[0,2t]+=1
            W[2*t*n + 0][col] = 1
            col += 1
            
            # M7 = (X01-X11)*(Y10+Y11)
            # U: A[0,2s+1] - A[1,2s+1]
            U[0*m + 2*s+1][col] = 1
            U[1*m + 2*s+1][col] = -1
            # V: B[2s+1,2t] + B[2s+1,2t+1]
            V[(2*s+1)*p + 2*t][col] = 1
            V[(2*s+1)*p + 2*t+1][col] = 1
            # W: Z11=M7+... => C[1,2t+1]+=1
            W[(2*t+1)*n + 1][col] = 1
            col += 1
    
    assert col == rank, f"column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
