"""Matrix multiplication 2x4x5: rank-minimal exact tensor decomposition.

Uses a Winograd-style / polynomial evaluation-interpolation approach combined
with Strassen-like 2x2x2 sub-block decomposition to reduce rank below schoolbook.

The strategy:
- Decompose the 2x4x5 multiplication into sub-blocks.
- For each sub-block, use the optimal known rank decomposition.
- Combine via Kronecker/product structure where possible.

Known results:
- 2x2x2 matrix multiplication has rank 7 (Strassen).
- 2x2x3 has rank 8 (known optimal).
- 2x3x3 has rank 9 (known optimal).

For 2x4x5:
We can view A (2x4) * B (4x5) = C (2x5).

One approach: split B into B1 (4x2) and B2 (4x3), giving C1 = A*B1 (2x2) and C2 = A*B2 (2x3).
- C1 uses rank 7 (Strassen for 2x2x2)
- C2: 2x4x3. We can further split: A (2x4) * B2 (4x3). Split B2 into two 4x? blocks...
  Actually 2x4x3: split the 4 into 2+2: A = [A1 A2] where A1, A2 are 2x2.
  C2 = A1 * B2a + A2 * B2b where B2a, B2b are 2x3.
  Each 2x2x3 has rank 8. So total for C2: 8 + 8 = 16.
  Total: 7 + 16 = 23.

But we can do better by sharing structure. Let's use a more careful decomposition.

Alternative: Use the fact that 2x4x5 can be decomposed using polynomial methods.
Represent the multiplication as polynomial evaluation.

Actually, let me use a direct Strassen-based approach on the 2x4x5 problem.

View A as 2x4 = [A1 A2] with A1, A2 in R^{2x2}.
View B as 4x5 = [B11 B12; B21 B22] where B11, B12 are 2x? and B21, B22 are 2x?.
Wait, B is 4x5. Split rows: top 2 rows, bottom 2 rows.
B_top = B[0:2, :] (2x5), B_bot = B[2:4, :] (2x5).

C = A1 * B_top + A2 * B_bot.

Now A1*B_top is 2x2 * 2x5 = 2x5. This is a 2x2x5 multiplication.
A2*B_bot is also 2x2x5.

For 2x2x5: we can use Strassen-like. 2x2x5 can be done with rank... 
Actually, let's think of 2x2xn. The rank of 2x2xn matrix multiplication tensor.
For n=2: rank 7.
For n=3: rank 8 (2x2x3).
For n=4: rank 10 (2x2x4, using two 2x2x2 Strassen with sharing? No, 2x2x4 = two independent 2x2x2, so 14? Or can share? Actually 2x2x4: A(2x2)*B(2x4)=C(2x4). This is two 2x2x2 multiplications side by side, so rank 14 unless we share. But they're independent, so rank 14.
Wait no - 2x2x4: the tensor is M_{2,2,4}. Its rank... Let me think. It's the direct sum of two M_{2,2,2} tensors (one for each column of B/C). So rank = 7+7 = 14.

Similarly 2x2x5 = M_{2,2,5} = M_{2,2,2} ⊕ M_{2,2,2} ⊕ M_{2,2,1}. Rank = 7+7+1 = 15.

So total rank = 15 + 15 = 30. That's better than 40 but not great.

Let me try a different split. 

Better approach: Use the known result that M_{2,4,5} has rank at most 23 or so.

Let me try: Split A (2x4) into two 2x2 blocks: A = [A1 A2].
Split B (4x5) into top 2 rows B1 (2x5) and bottom 2 rows B2 (2x5).
C = A1*B1 + A2*B2.

Each term is 2x2 * 2x5. As computed, rank 15 each, total 30.

Can we do better? What if we use a shared Strassen decomposition?

Alternative: Think of it as computing 2x5 output from 2x4 input times 4x5 matrix.

Let me try a polynomial/evaluation approach.

Actually, the simplest improvement from schoolbook (40) is to use Strassen on the 2x2 sub-blocks.

Let me carefully construct a rank-30 decomposition and see if we can shave more.

For now, let me implement the rank-30 decomposition and verify it.
"""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Decomposition:
    # A (2x4) = [A1 A2], each 2x2
    # B (4x5) = [B1; B2], each 2x5
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x5):
    #   B1 = [b1 b2 b3 b4 b5] (5 columns, each 2x1)
    #   Use Strassen for first two columns (2x2x2, rank 7)
    #   Use Strassen for next two columns (2x2x2, rank 7)  
    #   Use schoolbook for last column (2x2x1, rank 1)
    #   Total: 15
    #
    # Similarly for A2*B2: 15
    # Total rank: 30
    
    # Strassen for 2x2x2:
    # Given X (2x2), Y (2x2), compute Z = X*Y
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = [[z11, z12], [z21, z22]]
    #
    # Strassen 7 products:
    # m1 = (x11+x22)*(y11+y22)
    # m2 = (x21+x22)*y11
    # m3 = x11*(y12-y22)
    # m4 = x22*(y21-y11)
    # m5 = (x11+x12)*y22
    # m6 = (x21-x11)*(y11+y12)
    # m7 = (x12-x22)*(y21+y22)
    #
    # z11 = m1 + m4 - m5 + m7
    # z12 = m3 + m5
    # z21 = m2 + m4
    # z22 = m1 - m2 + m3 + m6
    
    def strassen_2x2x2():
        """Return U,V,W entries for one 2x2x2 Strassen block.
        U is 4x7, V is 4x7, W is 4x7.
        U rows indexed by (i,j) of X: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        V rows indexed by (j,k) of Y: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        W rows indexed by (k,i) of Z: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # m1 = (x11+x22)*(y11+y22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # z11 += m1, z22 += m1
        
        # m2 = (x21+x22)*y11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        # z21 += m2, z22 -= m2
        
        # m3 = x11*(y12-y22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # z12 += m3, z22 += m3
        
        # m4 = x22*(y21-y11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # z11 += m4, z21 += m4
        
        # m5 = (x11+x12)*y22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        # z11 -= m5, z12 += m5
        
        # m6 = (x21-x11)*(y11+y12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # z22 += m6
        
        # m7 = (x12-x22)*(y21+y22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        # z11 += m7
        
        return U, V, W
    
    # Schoolbook for 2x2x1: rank 1
    # X (2x2) * Y (2x1) = Z (2x1)
    # z1 = x11*y1 + x12*y2
    # z2 = x21*y1 + x22*y2
    # Actually this is rank 2, not 1! 2x2x1 has rank 2.
    # U: 4x2, V: 2x2, W: 2x2
    def schoolbook_2x2x1():
        U = [[0]*2 for _ in range(4)]
        V = [[0]*2 for _ in range(2)]
        W = [[0]*2 for _ in range(2)]
        # z1 = x11*y1 + x12*y2
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # z2 = x21*y1 + x22*y2
        U[2][0] = 1; V[0][0] = 1; W[1][0] = 1
        U[3][1] = 1; V[1][1] = 1; W[1][1] = 1
        return U, V, W
    
    # Now build the full decomposition.
    # Total rank: 15 + 15 = 30
    # 
    # U is (n*m) x rank = 8 x 30
    # V is (m*p) x rank = 20 x 30
    # W is (p*n) x rank = 10 x 30
    
    rank = 30
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(10)]
    
    # A indices: i in {0,1}, j in {0,1,2,3} -> row = i*4+j
    #   A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B indices: j in {0,1,2,3}, k in {0,1,2,3,4} -> row = j*5+k
    #   B1 = B[0:2, :], B2 = B[2:4, :]
    # C indices: k in {0,1,2,3,4}, i in {0,1} -> row = k*2+i (trace convention W[k*n+i])
    
    # For A1*B1:
    #   A1 has rows i*4+j for i in {0,1}, j in {0,1}
    #   B1 has rows j*5+k for j in {0,1}, k in {0,1,2,3,4}
    #   C has rows k*2+i
    
    # Strassen block 1: A1 * B1[:, 0:2] (2x2 * 2x2)
    U_s, V_s, W_s = strassen_2x2x2()
    col = 0
    for a in range(4):
        for c in range(7):
            # U_s[a][c] contributes to U[row_A1(a)][col+c]
            # A1 row a: a=0->(0,0)->row 0, a=1->(0,1)->row 1, a=2->(1,0)->row 4, a=3->(1,1)->row 5
            row_a = a  # for j in {0,1}, row = i*4+j; a=0->i=0,j=0->0; a=1->i=0,j=1->1; a=2->i=1,j=0->4; a=3->i=1,j=1->5
            row_a = [0, 1, 4, 5][a]
            U[row_a][col+c] += U_s[a][c]
    for b in range(4):
        for c in range(7):
            # V_s[b][c]: b=0->(0,0), b=1->(0,1), b=2->(1,0), b=3->(1,1)
            # B1 row: j*5+k; b=0->j=0,k=0->0; b=1->j=0,k=1->1; b=2->j=1,k=0->5; b=3->j=1,k=1->6
            row_b = [0, 1, 5, 6][b]
            V[row_b][col+c] += V_s[b][c]
    for w in range(4):
        for c in range(7):
            # W_s[w][c]: w=0->(0,0), w=1->(0,1), w=2->(1,0), w=3->(1,1)
            # C row: k*2+i; w=0->k=0,i=0->0; w=1->k=0,i=1->1; w=2->k=1,i=0->2; w=3->k=1,i=1->3
            row_w = [0, 1, 2, 3][w]
            W[row_w][col+c] += W_s[w][c]
    col += 7
    
    # Strassen block 2: A1 * B1[:, 2:4] (2x2 * 2x2)
    U_s, V_s, W_s = strassen_2x2x2()
    for a in range(4):
        for c in range(7):
            row_a = [0, 1, 4, 5][a]
            U[row_a][col+c] += U_s[a][c]
    for b in range(4):
        for c in range(7):
            # B1 columns 2,3: j=0,k=2->2; j=0,k=3->3; j=1,k=2->7; j=1,k=3->8
            row_b = [2, 3, 7, 8][b]
            V[row_b][col+c] += V_s[b][c]
    for w in range(4):
        for c in range(7):
            # C columns 2,3: k=2,i=0->4; k=2,i=1->5; k=3,i=0->6; k=3,i=1->7
            row_w = [4, 5, 6, 7][w]
            W[row_w][col+c] += W_s[w][c]
    col += 7
    
    # Schoolbook block 3: A1 * B1[:, 4] (2x2 * 2x1)
    U_sb, V_sb, W_sb = schoolbook_2x2x1()
    for a in range(4):
        for c in range(2):
            row_a = [0, 1, 4, 5][a]
            U[row_a][col+c] += U_sb[a][c]
    for b in range(2):
        for c in range(2):
            # B1 column 4: j=0,k=4->4; j=1,k=4->9
            row_b = [4, 9][b]
            V[row_b][col+c] += V_sb[b][c]
    for w in range(2):
        for c in range(2):
            # C column 4: k=4,i=0->8; k=4,i=1->9
            row_w = [8, 9][w]
            W[row_w][col+c] += W_sb[w][c]
    col += 2
    
    # Now A2*B2: same structure but A2 uses j in {2,3} and B2 uses j in {2,3}
    # A2 rows: i*4+j for i in {0,1}, j in {2,3}: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B2 rows: j*5+k for j in {2,3}, k in {0,1,2,3,4}: 2*5+0=10,...,3*5+4=19
    
    # Strassen block 4: A2 * B2[:, 0:2]
    U_s, V_s, W_s = strassen_2x2x2()
    for a in range(4):
        for c in range(7):
            row_a = [2, 3, 6, 7][a]
            U[row_a][col+c] += U_s[a][c]
    for b in range(4):
        for c in range(7):
            # B2 columns 0,1: j=2,k=0->10; j=2,k=1->11; j=3,k=0->15; j=3,k=1->16
            row_b = [10, 11, 15, 16][b]
            V[row_b][col+c] += V_s[b][c]
    for w in range(4):
        for c in range(7):
            row_w = [0, 1, 2, 3][w]
            W[row_w][col+c] += W_s[w][c]
    col += 7
    
    # Strassen block 5: A2 * B2[:, 2:4]
    U_s, V_s, W_s = strassen_2x2x2()
    for a in range(4):
        for c in range(7):
            row_a = [2, 3, 6, 7][a]
            U[row_a][col+c] += U_s[a][c]
    for b in range(4):
        for c in range(7):
            # B2 columns 2,3: j=2,k=2->12; j=2,k=3->13; j=3,k=2->17; j=3,k=3->18
            row_b = [12, 13, 17, 18][b]
            V[row_b][col+c] += V_s[b][c]
    for w in range(4):
        for c in range(7):
            row_w = [4, 5, 6, 7][w]
            W[row_w][col+c] += W_s[w][c]
    col += 7
    
    # Schoolbook block 6: A2 * B2[:, 4]
    U_sb, V_sb, W_sb = schoolbook_2x2x1()
    for a in range(4):
        for c in range(2):
            row_a = [2, 3, 6, 7][a]
            U[row_a][col+c] += U_sb[a][c]
    for b in range(2):
        for c in range(2):
            # B2 column 4: j=2,k=4->14; j=3,k=4->19
            row_b = [14, 19][b]
            V[row_b][col+c] += V_sb[b][c]
    for w in range(2):
        for c in range(2):
            # C column 4: k=4,i=0->8; k=4,i=1->9
            row_w = [8, 9][w]
            W[row_w][col+c] += W_sb[w][c]
    col += 2
    
    # Verify: col should be 30
    assert col == 30, f"Expected 30 columns, got {col}"
    
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
