"""Rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication
    using Strassen's 2x2 block algorithm.
    
    The 2x4 matrix A is split into two 2x2 blocks A1, A2.
    The 4x6 matrix B is split into two 4x3 blocks B1, B2.
    
    Using Strassen's 7 multiplications:
      M1 = (A1 + A2) * (B1 + B2)  -- 2x2 * 4x3
      M2 = (A1 - A2) * B1         -- 2x2 * 4x3
      M3 = A2 * B2                 -- 2x2 * 4x3
      M4 = A1 * (B2 - B1)         -- 2x2 * 4x3
      M5 = A1 * (B1 + B2)         -- 2x2 * 4x3
      M6 = (A2 - A1) * B1         -- 2x2 * 4x3
      M7 = A2 * (B2 - B1)         -- 2x2 * 4x3
      
    Then:
      C1 = M5 + M4 - M1 + M3    (2x3)
      C2 = M1 + M2             (2x3)
      
    Wait, let me reconsider. The standard Strassen for 2x2 * 2x2 gives:
      C11 = M5 + M4 - M1 + M3
      C12 = M1 + M2
      C21 = M6 + M7
      C22 = M5 - M6 + M1 - M2
      
    But our blocks are 2x2 * 4x3, so we need to think more carefully.
    
    Let me use a different approach: decompose the 4x6 into 2x2 sub-blocks.
    B = [B11 B12 B13; B21 B22 B23] where Bij are 2x2.
    A = [A1 A2] where A1, A2 are 2x2.
    
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    So C has three 2x2 blocks. For each column block j, C_j = A1*B1j + A2*B2j.
    
    For each j, use Strassen on the 2x2 * 2x2 multiplication:
    For fixed j:
      M1j = (A1 + A2) * (B1j + B2j)
      M2j = (A1 - A2) * B1j
      M3j = A2 * B2j
      M4j = A1 * (B2j - B1j)
      M5j = A1 * (B1j + B2j)
      M6j = (A2 - A1) * B1j
      M7j = A2 * (B2j - B1j)
      
    C_j1 (top 2 rows of C_j) = M5j + M4j - M1j + M3j
    C_j2 (bottom 2 rows of C_j) = M1j + M2j
    
    Total multiplications: 3 * 7 = 21 of type 2x2*2x2
    Each 2x2*2x2 multiplication can be done with 7 scalar mults (Strassen),
    so total scalar rank = 21 * 7 = 147? That's worse.
    
    Actually, the rank is the number of scalar multiplications. Let me reconsider.
    
    For a 2x4 * 4x6 multiplication, the schoolbook rank is 2*4*6 = 48.
    
    Using Strassen at the 2x2 block level:
    A is 2x4, split as A = [A_left A_right] where A_left, A_right are 2x2.
    B is 4x6, split as B = [B_top; B_bottom] where B_top, B_bottom are 2x6.
    
    C = A_left * B_top + A_right * B_bottom  (2x6)
    
    Now we have two 2x2 * 2x6 multiplications to add.
    Each 2x2 * 2x6 = sum over k=0..5 of (2x2)*(1x2)^T... no.
    
    2x2 * 2x6: Let B_top = [b0 b1 b2 b3 b4 b5] where bi are 2x1 columns.
    A_left * B_top = [A_left*b0, A_left*b1, ..., A_left*b5]
    
    So each column is a 2x2 * 1 vector = 4 scalar mults. 6 columns * 4 = 24.
    Similarly for A_right * B_bottom: 24.
    Total = 48, same as schoolbook.
    
    To use Strassen, we need 2x2 * 2x2 blocks. Let me split B into 2x2 blocks.
    B is 4x6 = [B11 B12 B13; B21 B22 B23] with 2x2 blocks.
    A is 2x4 = [A1 A2] with 2x2 blocks.
    
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    C is 2x6, so C = [C1 C2 C3] where Cj are 2x2.
    
    For each j: Cj = A1*B1j + A2*B2j.
    
    Now, A1*B1j is 2x2*2x2 = 4 mults schoolbook, 7 with Strassen (worse for small).
    A2*B2j similarly.
    
    So for each j: 4+4 = 8 schoolbook, or 7+7=14 Strassen.
    Total: 3*8 = 24 schoolbook... wait that's not right either.
    
    Hmm, 2x2 * 2x2 schoolbook is 4 scalar mults. 3 pairs = 12 mults? No.
    
    Let me recount. A is 2x4, B is 4x6. C is 2x6.
    Schoolbook: C[i][k] = sum_j A[i][j] * B[j][k].
    i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    Total = 2*4*6 = 48 scalar multiplications.
    
    For Strassen, we need to exploit the tensor structure. The 2x4*4x6 tensor
    can be viewed as: for each (i,j) in A and (j,k) in B, we get a term in C[i,k].
    
    Let me try a different block decomposition. Split the 4 middle dimension into two 2s.
    A = [A1 | A2] (2x2 each)
    B = [B1; B2] (2x6 each)
    C = A1*B1 + A2*B2
    
    Now A1*B1 is 2x2 * 2x6. Split B1 into three 2x2 blocks: B1 = [B11 B12 B13].
    A1*B1 = [A1*B11 A1*B12 A1*B13]
    
    Similarly A2*B2 = [A2*B21 A2*B22 A2*B23].
    
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    Each A1*B1j is 2x2 * 2x2. Schoolbook = 4 mults.
    Total = 3 * 2 * 4 = 24? No, 6 matrices of 2x2*2x2, each 4 mults = 24.
    But wait, that's only 24, but schoolbook was 48. I'm double counting or missing something.
    
    Ah I see the issue. 2x2 * 2x2 schoolbook: C[i][k] = sum_j A[i][j]*B[j][k], i,k in {0,1}, j in {0,1}.
    That's 4 scalar multiplications. Yes.
    
    So 6 such products = 24. But schoolbook for 2x4*4x6 is 48. The discrepancy is because
    in the block form, A1*B11 involves A[0][0]*B[0][0], A[0][1]*B[1][0], A[1][0]*B[0][0], A[1][1]*B[1][0].
    And A2*B21 involves A[0][2]*B[2][0], A[0][3]*B[3][0], A[1][2]*B[2][0], A[1][3]*B[3][0].
    Together they give C[0][0] and C[1][0]. That's 8 mults for one column of C.
    6 columns * 8 = 48. OK so 6 * 4 = 24 is wrong; each 2x2*2x2 is 4 mults, and we have 6 of them, total 24. But that can't be right.
    
    Wait: A1 is 2x2, B11 is 2x2. A1*B11 gives a 2x2 result with 4 entries, each being a sum of 2 products. So 4 entries * 2 = 8 scalar multiplications? No, 2x2*2x2:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Scalar multiplications: A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11 = 8.
    
    Ah, I was wrong. 2x2*2x2 schoolbook is 8 scalar multiplications, not 4. The "4" is the number of entries in the result, but each entry requires 2 mults.
    
    So 6 * 8 = 48. Correct.
    
    Now Strassen for 2x2*2x2 uses 7 mults instead of 8. So 6 * 7 = 42.
    
    That gives rank 42! Let me construct this.
    
    For each j in {0,1,2} (the three column blocks of B):
    Let B1j be the top 2x2 block of column j of B.
    Let B2j be the bottom 2x2 block.
    
    Strassen products:
    P1 = (A1 + A2) * (B1j + B2j)
    P2 = (A1 - A2) * B1j
    P3 = A2 * (B2j - B1j)
    P4 = A1 * (B1j + B2j)
    P5 = A2 * B2j
    P6 = (A1 - A2) * (B2j + B1j)  -- wait, standard Strassen:
    
    Standard Strassen for A*B where A,B are 2x2:
    A = [a b; c d], B = [e f; g h]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [M1+M4-M5+M7, M5+M3; M2+M6, M1-M2+M3+M4]
    
    So for each column block j:
    A1 = [a b; c d] (rows 0,1 of A, cols 0,1)
    A2 = [e f; g h] (rows 0,1 of A, cols 2,3)
    B1j = top 2x2 of B column block j
    B2j = bottom 2x2 of B column block j
    
    M1j = (A1+A2) * (B1j+B2j)  -- element-wise? No, matrix product.
    
    Actually, Strassen operates on the sub-matrices. Let me be more careful.
    
    A (2x4) = [A_left | A_right] where A_left = A[0:2,0:2], A_right = A[0:2,2:4]
    B (4x6) = [B_top; B_bottom] where B_top = B[0:2,:], B_bottom = B[2:4,:]
    
    C = A_left * B_top + A_right * B_bottom  (2x6)
    
    Now split B_top into three 2x2 blocks: B_top = [BT1 BT2 BT3]
    B_bottom = [BB1 BB2 BB3]
    
    C = [A_left*BT1 + A_right*BB1, A_left*BT2 + A_right*BB2, A_left*BT3 + A_right*BB3]
    
    For each block j: Cj = A_left*BTj + A_right*BBj (2x2)
    
    Now apply Strassen to compute A_left*BTj and A_right*BBj separately? No, that doesn't help.
    
    The trick is: Cj = A_left*BTj + A_right*BBj. We can't directly apply Strassen to this sum.
    
    Alternative: View the whole thing differently.
    
    Let me use the approach from the knowledge: "Using Strassen's algorithm on 2x2 blocks reduces the rank from 48 to 42."
    
    One way: embed 2x4*4x6 into a larger problem. Or use the fact that:
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    
    Split j into j1 in {0,1} and j2 in {2,3}:
    C[i,k] = sum_{j1=0}^{1} A[i,j1] B[j1,k] + sum_{j2=0}^{1} A[i,2+j2] B[2+j2,k]
    
    = (A_left * B_top)[i,k] + (A_right * B_bottom)[i,k]
    
    Now, A_left * B_top is 2x2 * 2x6. This can be seen as three 2x2*2x2 products:
    A_left * [BT1 BT2 BT3] = [A_left*BT1, A_left*BT2, A_left*BT3]
    
    Each A_left*BTj is 2x2*2x2, requiring 8 mults schoolbook.
    
    Similarly A_right * B_bottom = [A_right*BB1, A_right*BB2, A_right*BB3], each 8 mults.
    
    Total: 6 * 8 = 48.
    
    To use Strassen, we need to restructure. One approach:
    
    For each column block j, compute Cj = A_left*BTj + A_right*BBj.
    
    Note that if we had A' = [A_left; A_right] (4x2) and B' = [BTj BBj] (4x2), then Cj = A' * B'.
    A' is 4x2, B' is 2x2. Cj = A'*B' is 4x2 * 2x2 = 4x2.
    
    A'*B'[i,k] = sum_{l=0}^{1} A'[i,l] B'[l,k]
    
    This is 4*2*2 = 16 mults for each j, total 48. Same.
    
    OK let me just directly implement the rank-42 Strassen-based decomposition.
    
    The key insight: we can use 7 Strassen multiplications for each 2x2*2x2 sub-product, but we need to share them across the three column blocks.
    
    Actually, here's a cleaner approach. Consider the 2x4 * 4x6 product as:
    
    A (2x4) times B (4x6). Think of B as a 4x6 matrix.
    
    Split the 4 rows of B into two groups of 2: B = [B_a; B_b] where B_a, B_b are 2x6.
    Split A into A = [A_a | A_b] where A_a, A_b are 2x2.
    
    C = A_a * B_a + A_b * B_b  (2x6)
    
    Now, A_a * B_a is 2x2 * 2x6. Write B_a = [b0 b1 b2 b3 b4 b5] (columns, each 2x1).
    A_a * B_a = [A_a*b0, A_a*b1, ..., A_a*b5].
    Each A_a*bi is 2x2 * 2x1 = 4 mults. 6 columns = 24 mults.
    Similarly A_b * B_b = 24 mults. Total 48.
    
    For Strassen, I need to group these. Let me try a different strategy.
    
    Since 42 = 6*7, maybe I can identify 6 independent 2x2*2x2 Strassen applications.
    
    Here's the plan:
    - Split B's 6 columns into 3 pairs: (col0,col1), (col2,col3), (col4,col5).
    - For each pair, the relevant part of B is a 4x2 matrix.
    - C's corresponding 2x2 block = A * (4x2 block of B).
    
    For column pair j (j=0,1,2), let B_j be the 4x2 submatrix.
    C_j = A * B_j (2x2).
    
    A is 2x4, B_j is 4x2. C_j[i,k] = sum_{l=0}^{3} A[i,l] B_j[l,k].
    
    Split A = [A_a | A_b] (2x2 each), B_j = [Bj_a; Bj_b] (2x2 each).
    C_j = A_a * Bj_a + A_b * Bj_b.
    
    Now apply Strassen-like decomposition to C_j = A_a*Bj_a + A_b*Bj_b.
    
    Define:
    S1 = A_a + A_b  (2x2)
    S2 = A_a - A_b  (2x2)
    T1 = Bj_a + Bj_b  (2x2)
    T2 = Bj_a - Bj_b  (2x2)
    
    M1 = S1 * T1  (2x2 * 2x2, 7 mults Strassen)
    M2 = S2 * Bj_a  (2x2 * 2x2, 7 mults)
    M3 = A_b * T2  (2x2 * 2x2, 7 mults)
    M4 = A_a * T2  (2x2 * 2x2, 7 mults)
    
    Wait, I need exactly the right combination. Let me derive:
    
    C_j = A_a*Bj_a + A_b*Bj_b
    
    Using: A_a*Bj_a = ((A_a+A_b) - A_b) * ((Bj_a+Bj_b) - Bj_b)
    = (A_a+A_b)*(Bj_a+Bj_b) - (A_a+A_b)*Bj_b - A_b*(Bj_a+Bj_b) + A_b*Bj_b
    
    So A_a*Bj_a + A_b*Bj_b = (A_a+A_b)*(Bj_a+Bj_b) - (A_a+A_b)*Bj_b - A_b*(Bj_a+Bj_b) + 2*A_b*Bj_b
    
    That uses 4 matrix products. Not better.
    
    Standard Strassen computes P*Q with 7 mults where P,Q are 2x2.
    Here I have a sum of two products. 
    
    Alternative: C_j = A * B_j where A is 2x4, B_j is 4x2.
    This is equivalent to a 2x4 * 4x2 multiplication, which has rank 16 schoolbook.
    Can it be done in fewer? 2*4*2 = 16.
    
    Total rank = 3 * 16 = 48 schoolbook.
    
    For Strassen: 2x4 * 4x2. Split the 4 middle dimension:
    A = [A1|A2] (2x2), B_j = [Bj1; Bj2] (2x2).
    A*B_j = A1*Bj1 + A2*Bj2.
    
    This is the same as before. The rank of 2x4*4x2 is 16 (it's a standard result that the rank equals n*m*p for small cases, but for 2x2*2x2 it's 7).
    
    Actually, the tensor rank of 2x4 * 4x2: this is the multiplication tensor for M_{2,4} * M_{4,2}.
    The rank is known to be... let me think. It's the same as computing a 2x2 result where each entry is a dot product of length 4.
    
    I think the rank of 2x4*4x2 is 16 (no known improvement for this specific shape in standard references, though I'm not 100% sure).
    
    Let me just go with the rank-42 construction from the measured knowledge. The knowledge says "Using Strassen's algorithm on 2x2 blocks reduces the rank from 48 to 42." This suggests there IS a valid rank-42 decomposition.
    
    Let me construct it explicitly by identifying 6 groups of 7 Strassen multiplications.
    
    Here's my approach: I'll use the fact that the 2x4 * 4x6 tensor can be decomposed using 6 Strassen 2x2*2x2 decompositions.
    
    The 48 scalar multiplications in schoolbook are:
    For i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}: A[i,j]*B[j,k]
    
    I'll group these into 6 sets of 8, and replace each set of 8 with 7 Strassen multiplications.
    
    Group 1: i=0, k=0: A[0,0]*B[0,0], A[0,1]*B[1,0], A[0,2]*B[2,0], A[0,3]*B[3,0] -- 4 mults
    Group 2: i=1, k=0: A[1,0]*B[0,0], A[1,1]*B[1,0], A[1,2]*B[2,0], A[1,3]*B[3,0] -- 4 mults
    ...
    
    That gives 12 groups of 4, not 6 groups of 8.
    
    Let me try: group by (i, k) pairs where i in {0,1} and k in {0,1,2,3,4,5}: 12 groups of 4.
    
    Or group differently: for each j in {0,1,2,3}, and for the 2x6 = 12 terms A[i,j]*B[j,k].
    4 groups of 12.
    
    Hmm, 42 = 48 - 6. So I need to save 6 multiplications total.
    
    Actually, 42 = 6 * 7. So it's 6 Strassen blocks of 7.
    
    Let me try this grouping:
    - Block 1: terms involving A[0,0], A[0,1], A[1,0], A[1,1] and B[0,0:2], B[1,0:2] (i.e., B[0,0], B[0,1], B[1,0], B[1,1])
      This computes C[0,0], C[0,1], C[1,0], C[1,1] partially.
      8 mults schoolbook, 7 Strassen.
      
    - Block 2: terms involving A[0,2], A[0,3], A[1,2], A[1,3] and B[2,0:2], B[3,0:2]
      8 mults, 7 Strassen.
      
    - Block 3: terms involving A[0,0], A[0,1], A[1,0], A[1,1] and B[0,2:4], B[1,2:4]
      8 mults, 7 Strassen.
      
    - Block 4: A[0,2..3] and B[2..3, 2:4]: 7
    - Block 5: A[0,0..1] and B[0..1, 4:6]: 7
    - Block 6: A[0,2..3] and B[2..3, 4:6]: 7
    
    Total: 6 * 7 = 42. 
    
    Now I need to write the actual factor matrices. For each Strassen block, I have 7 scalar multiplications. Each scalar multiplication corresponds to one column in U, V, W.
    
    For a Strassen 2x2*2x2 block with matrices:
    A_block = [[a, b], [c, d]] (from A's 2x2 sub-block)
    B_block = [[e, f], [g, h]] (from B's 2x2 sub-block)
    
    The 7 Strassen products:
    m1 = (a+d)(e+h)
    m2 = (c+d)(e)
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)(h)
    m6 = (c-a)(e+f)
    m7 = (b-d)(g+h)
    
    C_block = [[m1+m4-m5+m7, m5+m3], [m2+m6, m1-m2+m3+m4]]
    
    Each mk is a product of two linear combinations. In the tensor decomposition:
    U encodes the coefficients for A, V for B, W for C.
    
    For m1 = (a+d)(e+h):
    U column has 1 in positions for a and d.
    V column has 1 in positions for e and h.
    W column has +1 in positions for C00 and C11 (from m1's contribution).
    
    Wait, I need to be more careful. Let me define the indexing.
    
    A is 2x4, indexed as A[i,j] where i in {0,1}, j in {0,1,2,3}.
    U has shape (2*4, rank) = (8, rank). U[i*4+j, c] corresponds to A[i,j].
    
    B is 4x6, indexed as B[j,k] where j in {0,1,2,3}, k in {0,1,2,3,4,5}.
    V has shape (4*6, rank) = (24, rank). V[j*6+k, c] corresponds to B[j,k].
    
    C is 2x6, indexed as C[i,k] where i in {0,1}, k in {0,1,2,3,4,5}.
    W has shape (6*2, rank) = (12, rank). W[k*2+i, c] corresponds to C[i,k].
    
    For each Strassen block, I'll create 7 columns.
    
    Let me implement this step by step.
    
    Block 1: A sub-block [[A00, A01], [A10, A11]] = rows 0,1 cols 0,1 of A.
    B sub-block [[B00, B01], [B10, B11]] = rows 0,1 cols 0,1 of B.
    C sub-block: contributes to C[0,0], C[0,1], C[1,0], C[1,1].
    
    Strassen:
    m1 = (A00+A10)*(B00+B11)  -- wait, (a+d) where a=A00, d=A10? No.
    
    Let me be precise. A_block = [[a, b], [c, d]] where:
    a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1].
    B_block = [[e, f], [g, h]] where:
    e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1].
    
    m1 = (a+d)*(e+h) = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    m2 = (c+d)*e = (A[1,0]+A[1,1])*B[0,0]
    m3 = a*(f-h) = A[0,0]*(B[0,1]-B[1,1])
    m4 = d*(g-e) = A[1,1]*(B[1,0]-B[0,0])
    m5 = (a+b)*h = (A[0,0]+A[0,1])*B[1,1]
    m6 = (c-a)*(e+f) = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    m7 = (b-d)*(g+h) = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    C[0,0] += m1+m4-m5+m7
    C[0,1] += m5+m3
    C[1,0] += m2+m6
    C[1,1] += m1-m2+m3+m4
    
    Now for each mk, I create one column in U, V, W.
    
    For m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    U: U[0*4+0] = 1, U[1*4+1] = 1  (A[0,0] and A[1,1])
    V: V[0*6+0] = 1, V[1*6+1] = 1  (B[0,0] and B[1,1])
    W: W[0*2+0] = 1, W[1*2+1] = 1  (C[0,0] gets +1, C[1,1] gets +1)
    
    For m2 = (A[1,0]+A[1,1])*B[0,0]:
    U: U[1*4+0] = 1, U[1*4+1] = 1
    V: V[0*6+0] = 1
    W: W[0*2+1] = 1  (C[1,0] gets +1)
    
    For m3 = A[0,0]*(B[0,1]-B[1,1]):
    U: U[0*4+0] = 1
    V: V[0*6+1] = 1, V[1*6+1] = -1
    W: W[1*2+0] = 1, W[1*2+1] = 1  (C[0,1] gets +1, C[1,1] gets +1)
    
    For m4 = A[1,1]*(B[1,0]-B[0,0]):
    U: U[1*4+1] = 1
    V: V[1*6+0] = 1, V[0*6+0] = -1
    W: W[0*2+0] = 1, W[1*2+1] = 1  (C[0,0] gets +1, C[1,1] gets +1)
    
    For m5 = (A[0,0]+A[0,1])*B[1,1]:
    U: U[0*4+0] = 1, U[0*4+1] = 1
    V: V[1*6+1] = 1
    W: W[0*2+0] = -1, W[0*2+1] = 1  (C[0,0] gets -1, C[0,1] gets +1)
    
    For m6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1]):
    U: U[1*4+0] = 1, U[0*4+0] = -1
    V: V[0*6+0] = 1, V[0*6+1] = 1
    W: W[0*2+1] = 1  (C[1,0] gets +1)
    
    For m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1]):
    U: U[0*4+1] = 1, U[1*4+1] = -1
    V: V[1*6+0] = 1, V[1*6+1] = 1
    W: W[0*2+0] = 1, W[1*2+1] = -1  (C[0,0] gets +1, C[1,1] gets -1)
    
    Let me verify C[0,0] contributions: m1(+1) + m4(+1) + m5(-1) + m7(+1) = m1+m4-m5+m7. Correct.
    C[0,1]: m5(+1) + m3(+1) = m5+m3. Correct.
    C[1,0]: m2(+1) + m6(+1) = m2+m6. Correct.
    C[1,1]: m1(+1) + m3(+1) + m4(+1) + m7(-1) = m1-m2+m3+m4? Wait, m7 contributes -1 to C[1,1].
    m1: +1, m3: +1, m4: +1, m7: -1. Sum = m1+m3+m4-m7.
    But Strassen says C[1,1] = m1-m2+m3+m4. I have m1+m3+m4-m7 instead of m1-m2+m3+m4.
    
    I made an error. Let me recheck. m7 contributes to C[0,0] and C[1,1].
    C[0,0] = m1+m4-m5+m7. So m7 contributes +1 to C[0,0].
    C[1,1] = m1-m2+m3+m4. So m7 does NOT contribute to C[1,1].
    
    I incorrectly assigned m7 to C[1,1]. Let me fix:
    m7 = (b-d)*(g+h) = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    This only contributes to C[0,0] with coefficient +1.
    
    So W for m7: W[0*2+0] = 1 only.
    
    Let me redo all W assignments:
    m1: C[0,0]+1, C[1,1]+1
    m2: C[1,0]+1
    m3: C[0,1]+1, C[1,1]+1
    m4: C[0,0]+1, C[1,1]+1
    m5: C[0,0]-1, C[0,1]+1
    m6: C[1,0]+1
    m7: C[0,0]+1
    
    Check:
    C[0,0] = m1+m4-m5+m7: +1+1-1+1 = m1+m4-m5+m7. Correct.
    C[0,1] = m5+m3: +1+1. Correct.
    C[1,0] = m2+m6: +1+1. Correct.
    C[1,1] = m1+m3+m4: +1+1+1. But should be m1-m2+m3+m4. I'm missing -m2.
    
    m2 should contribute -1 to C[1,1]. Let me fix m2's W:
    m2: C[1,0]+1, C[1,1]-1
    
    Check C[1,0] = m2+m6: +1+1. Correct.
    C[1,1] = m1-m2+m3+m4: +1-1+1+1. Correct.
    
    Good. Now I'll write the full implementation for all 6 blocks.
    
    The 6 blocks are:
    Block 1: A rows 0,1 cols 0,1; B rows 0,1 cols 0,1 -> C cols 0,1
    Block 2: A rows 0,1 cols 2,3; B rows 2,3 cols 0,1 -> C cols 0,1
    Block 3: A rows 0,1 cols 0,1; B rows 0,1 cols 2,3 -> C cols 2,3
    Block 4: A rows 0,1 cols 2,3; B rows 2,3 cols 2,3 -> C cols 2,3
    Block 5: A rows 0,1 cols 0,1; B rows 0,1 cols 4,5 -> C cols 4,5
    Block 6: A rows 0,1 cols 2,3; B rows 2,3 cols 4,5 -> C cols 4,5
    
    For each block, the Strassen indices are:
    a = A[i0, j0], b = A[i0, j0+1], c = A[i0+1, j0], d = A[i0+1, j0+1]
    e = B[r0, k0], f = B[r0, k0+1], g = B[r0+1, k0], h = B[r0+1, k0+1]
    where i0=0, j0 = 0 or 2; r0 = 0 or 2, k0 = 0, 2, or 4.
    
    And the C indices: C[0, k0], C[0, k0+1], C[1, k0], C[1, k0+1].
    
    Let me code this up.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Define the 6 Strassen blocks
    # Each block: (a_row0, a_col0, b_row0, b_col0)
    # a: A[a_row0:a_row0+2, a_col0:a_col0+2]
    # b: B[b_row0:b_row0+2, b_col0:b_col0+2]
    # c: C[0:2, b_col0:b_col0+2]
    blocks = [
        (0, 0, 0, 0),  # Block 1
        (0, 2, 2, 0),  # Block 2
        (0, 0, 0, 2),  # Block 3
        (0, 2, 2, 2),  # Block 4
        (0, 0, 0, 4),  # Block 5
        (0, 2, 2, 4),  # Block 6
    ]
    
    rank = 6 * 7  # 42
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    col = 0
    for (ar, ac, br, bc) in blocks:
        # A indices: a=A[ar,ac], b=A[ar,ac+1], c=A[ar+1,ac], d=A[ar+1,ac+1]
        a_idx = (ar, ac)       # A row ar, col ac
        b_idx = (ar, ac + 1)   # A row ar, col ac+1
        c_idx = (ar + 1, ac)   # A row ar+1, col ac
        d_idx = (ar + 1, ac + 1)
        
        # B indices: e=B[br,bc], f=B[br,bc+1], g=B[br+1,bc], h=B[br+1,bc+1]
        e_idx = (br, bc)
        f_idx = (br, bc + 1)
        g_idx = (br + 1, bc)
        h_idx = (br + 1, bc + 1)
        
        # C indices
        c00 = (0, bc)      # C[0, bc]
        c01 = (0, bc + 1)  # C[0, bc+1]
        c10 = (1, bc)      # C[1, bc]
        c11 = (1, bc + 1)  # C[1, bc+1]
        
        def u_pos(i, j):
            return i * m + j
        
        def v_pos(i, j):
            return i * p + j
        
        def w_pos(i, j):
            return j * n + i
        
        # m1 = (a+d)*(e+h)
        U[u_pos(*a_idx)][col] = 1
        U[u_pos(*d_idx)][col] = 1
        V[v_pos(*e_idx)][col] = 1
        V[v_pos(*h_idx)][col] = 1
        W[w_pos(*c00)][col] = 1
        W[w_pos(*c11)][col] = 1
        col += 1
        
        # m2 = (c+d)*e
        U[u_pos(*c_idx)][col] = 1
        U[u_pos(*d_idx)][col] = 1
        V[v_pos(*e_idx)][col] = 1
        W[w_pos(*c10)][col] = 1
        W[w_pos(*c11)][col] = -1
        col += 1
        
        # m3 = a*(f-h)
        U[u_pos(*a_idx)][col] = 1
        V[v_pos(*f_idx)][col] = 1
        V[v_pos(*h_idx)][col] = -1
        W[w_pos(*c01)][col] = 1
        W[w_pos(*c11)][col] = 1
        col += 1
        
        # m4 = d*(g-e)
        U[u_pos(*d_idx)][col] = 1
        V[v_pos(*g_idx)][col] = 1
        V[v_pos(*e_idx)][col] = -1
        W[w_pos(*c00)][col] = 1
        W[w_pos(*c11)][col] = 1
        col += 1
        
        # m5 = (a+b)*h
        U[u_pos(*a_idx)][col] = 1
        U[u_pos(*b_idx)][col] = 1
        V[v_pos(*h_idx)][col] = 1
        W[w_pos(*c00)][col] = -1
        W[w_pos(*c01)][col] = 1
        col += 1
        
        # m6 = (c-a)*(e+f)
        U[u_pos(*c_idx)][col] = 1
        U[u_pos(*a_idx)][col] = -1
        V[v_pos(*e_idx)][col] = 1
        V[v_pos(*f_idx)][col] = 1
        W[w_pos(*c10)][col] = 1
        col += 1
        
        # m7 = (b-d)*(g+h)
        U[u_pos(*b_idx)][col] = 1
        U[u_pos(*d_idx)][col] = -1
        V[v_pos(*g_idx)][col] = 1
        V[v_pos(*h_idx)][col] = 1
        W[w_pos(*c00)][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
