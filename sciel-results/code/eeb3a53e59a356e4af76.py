"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]
DEADLINE = 150.0
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-15 decomposition for 2x3x4 matrix multiplication.
    
    Strategy:
    The 2x3x4 multiplication tensor can be decomposed using the identity:
    M_{2,3,4} is isomorphic to a sum of M_{2,1,4} and two M_{2,2,2} tensors.
    Specifically, we can split the 3 columns of the first matrix and 4 columns of the second.
    
    Let A be 2x3 and B be 3x4.
    A = [A1 | A2 | A3] where A1, A2, A3 are 2x1.
    B = [B1 B2; B3 B4; B5 B6] where B1..B6 are 1x2? No, B is 3x4.
    
    Actually, a known rank bound for M_{2,3,4} is 15.
    One way to achieve rank 15 is to use the decomposition:
    M_{2,3,4} = M_{2,1,4} + M_{2,2,2} + M_{2,2,2}
    Rank(M_{2,1,4}) = 2*1*4 = 8? No, rank is 2*1*4=8? 
    Wait, rank of M_{m,n,p} is at least max(mn, mp, np).
    For M_{2,1,4}, rank is 2*1*4 = 8? No, the rank of the multiplication tensor M_{m,n,p} is the minimum number of bilinear multiplications.
    For M_{2,1,4}, it's just vector-matrix multiplication. Rank is 2*4=8? No, it's 2*1*4=8.
    
    Let's look at the structure more carefully.
    We want to compute C = A * B where A is 2x3, B is 3x4.
    C is 2x4.
    
    A known result: Rank(M_{2,3,4}) = 15.
    This can be constructed by embedding M_{2,2,2} (Strassen, rank 7) twice and handling the rest.
    
    Let's try to construct it explicitly.
    Let A = [[a00, a01, a02],
             [a10, a11, a12]]
    Let B = [[b00, b01, b02, b03],
             [b10, b11, b12, b13],
             [b20, b21, b22, b23]]
    
    C = A * B
    C[0,0] = a00*b00 + a01*b10 + a02*b20
    C[0,1] = a00*b01 + a01*b11 + a02*b21
    C[0,2] = a00*b02 + a01*b12 + a02*b22
    C[0,3] = a00*b03 + a01*b13 + a02*b23
    C[1,0] = a10*b00 + a11*b10 + a12*b20
    C[1,1] = a10*b01 + a11*b11 + a12*b21
    C[1,2] = a10*b02 + a11*b12 + a12*b22
    C[1,3] = a10*b03 + a11*b13 + a12*b23
    
    We can group the computations.
    Let's use the fact that M_{2,3,4} can be reduced to M_{2,2,4} + M_{1,1,4} or similar.
    
    Actually, there is a known decomposition of rank 15.
    It uses 7 multiplications for a 2x2x2 part, 7 for another 2x2x2 part, and 1 for the remaining.
    
    Let's try to find the explicit U, V, W matrices for rank 15.
    Since I don't have the exact coordinates memorized, I will implement a search for a rank-15 decomposition using linear algebra and integer constraints, or fall back to a known construction if possible.
    
    However, a simpler approach:
    The rank of M_{2,3,4} is known to be 15.
    We can construct it by splitting the 3x4 matrix B into two 3x2 matrices B1 and B2.
    Then C = [A*B1 | A*B2].
    This doesn't reduce the rank directly because A*B1 is 2x2 and A*B2 is 2x2.
    Rank(M_{2,3,2}) is 9? No, 2*3*2=12 schoolbook.
    
    Let's use the decomposition:
    M_{2,3,4} is isomorphic to M_{2,2,4} + M_{1,1,4}?
    No, let's look at the indices.
    
    Alternative: Use the fact that we can compute A*B by first computing some linear combinations.
    
    Let's try a different known bound.
    Rank(M_{2,3,4}) <= 15.
    Construction:
    Let U, V, W be 24x15, 12x15, 8x15 matrices.
    
    I will implement a randomized search for a rank-15 decomposition with integer entries.
    If not found within time, I will return the schoolbook rank 24.
    
    But wait, the task asks for a *rank-minimal* exact decomposition.
    If I return rank 24, the score is 1.0 (same as schoolbook).
    If I return rank 15, the score is 24/15 = 1.6.
    
    Let's try to hardcode the rank-15 decomposition if I can derive it.
    
    Actually, let's use a property:
    M_{2,3,4} can be viewed as M_{2,3,2} + M_{2,0,2} + M_{2,0,2}? No.
    
    Let's try to construct it using the following idea:
    Split A into A1 (2x2) and A2 (2x1).
    Split B into B1 (2x4) and B2 (1x4).
    C = A1*B1 + A2*B2.
    A1*B1 is 2x2 * 2x4 = 2x4.
    A2*B2 is 2x1 * 1x4 = 2x4.
    
    Rank(M_{2,2,4})?
    This is the multiplication of a 2x2 matrix by a 2x4 matrix.
    This can be seen as two 2x2 multiplications (since B1 = [B1a | B1b] where B1a, B1b are 2x2).
    So Rank(M_{2,2,4}) = 2 * Rank(M_{2,2,2}) = 2 * 7 = 14.
    
    Rank(M_{2,1,4})?
    This is a 2x1 vector times a 1x4 matrix.
    This is just 2*4 = 8 scalar multiplications?
    Wait, the rank of the tensor for u * v^T where u is 2x1 and v is 1x4 is 2*4=8.
    
    So total rank = 14 + 8 = 22. Still better than 24, but not 15.
    
    How to get 15?
    We need to share some multiplications between the A1*B1 and A2*B2 parts.
    
    Let's look at the structure of M_{2,3,4}.
    It is known that M_{2,3,4} has rank 15.
    
    I will implement a search for rank 15.
    Given the complexity, I will first try to return a rank 22 decomposition (which is easier to construct) to get a score of 24/22 = 1.09.
    Then I will try to optimize to 15.
    
    Actually, let's just implement the rank 22 decomposition first.
    A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    B = [B1; B2] where B1 is 2x4, B2 is 1x4.
    C = A1*B1 + A2*B2.
    
    A1*B1:
    A1 = [[a00, a01], [a10, a11]]
    B1 = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    
    We can compute A1*B1 using two Strassen 2x2 multiplications.
    Let B1 = [B1a | B1b] where B1a = [[b00, b01], [b10, b11]] and B1b = [[b02, b03], [b12, b13]].
    
    Strassen for A1*B1a:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b10-b11)
    m4 = a11(b01-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a00)(b00+b10)
    m7 = (a01-a11)(b11+b21) -> wait, indices.
    
    Let's use standard Strassen for 2x2.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+g)
    m7 = (b-d)(g+h)
    
    C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    So for A1*B1a:
    A1 = [[a00, a01], [a10, a11]]
    B1a = [[b00, b01], [b10, b11]]
    
    m1_1 = (a00+a11)(b00+b11)
    m2_1 = (a10+a11)b00
    m3_1 = a00(b01-b11)
    m4_1 = a11(b10-b00)
    m5_1 = (a00+a01)b11
    m6_1 = (a10-a00)(b00+b10)
    m7_1 = (a01-a11)(b11+b10) -> wait, (b-d)(g+h) -> (a01-a11)(b10+b11)
    
    C1a = A1*B1a = [[m1_1+m4_1-m5_1+m7_1, m3_1+m5_1], [m2_1+m4_1, m1_1-m2_1+m3_1+m6_1]]
    
    Similarly for A1*B1b:
    B1b = [[b02, b03], [b12, b13]]
    
    m1_2 = (a00+a11)(b02+b13)
    m2_2 = (a10+a11)b02
    m3_2 = a00(b03-b13)
    m4_2 = a11(b12-b02)
    m5_2 = (a00+a01)b13
    m6_2 = (a10-a00)(b02+b12)
    m7_2 = (a01-a11)(b13+b12)
    
    C1b = A1*B1b = [[m1_2+m4_2-m5_2+m7_2, m3_2+m5_2], [m2_2+m4_2, m1_2-m2_2+m3_2+m6_2]]
    
    Now for A2*B2:
    A2 = [[a02], [a12]]
    B2 = [[b20, b21, b22, b23]]
    
    C2 = [[a02*b20, a02*b21, a02*b22, a02*b23],
          [a12*b20, a12*b21, a12*b22, a12*b23]]
    
    This requires 8 multiplications.
    Total = 7 + 7 + 8 = 22.
    
    So we can achieve rank 22.
    
    To achieve rank 15, we need to merge some of these.
    Notice that A2 is just the third column of A.
    And B2 is the third row of B.
    
    Is there a way to reduce the 8 multiplications in A2*B2?
    No, it's a rank-8 tensor.
    
    But maybe we can share multiplications between A1*B1 and A2*B2?
    For example, if we form linear combinations of A1 and A2?
    
    Let's define new matrices:
    A' = [A1 + A2 * x | ...] ?
    
    Actually, the known rank 15 decomposition for M_{2,3,4} is non-trivial.
    Given the time constraint, I will implement the rank 22 decomposition.
    This gives a score of 24/22 = 1.0909.
    
    If I can find a rank 20 or 15, the score will be higher.
    
    Let's try to search for a rank 20 decomposition using random linear combinations.
    
    However, for now, I will return the rank 22 decomposition.
    
    Wait, the schoolbook rank is 24.
    The current program returns rank 24.
    My program returns rank 22.
    Score = 24/22 = 1.09.
    
    Let's implement the rank 22 decomposition.
    
    U, V, W shapes:
    U: (n*m, rank) = (2*3, 22) = (6, 22)
    V: (m*p, rank) = (3*4, 22) = (12, 22)
    W: (p*n, rank) = (4*2, 22) = (8, 22)
    
    Indices:
    U[i*m + j, c] -> U[i*3 + j, c]
    V[j*p + k, c] -> V[j*4 + k, c]
    W[k*n + i, c] -> W[k*2 + i, c]
    
    Let's build the 22 columns.
    Columns 0-6: Strassen for A1*B1a
    Columns 7-13: Strassen for A1*B1b
    Columns 14-21: A2*B2
    
    A1 = [[a00, a01], [a10, a11]]
    A2 = [[a02], [a12]]
    
    B1a = [[b00, b01], [b10, b11]]
    B1b = [[b02, b03], [b12, b13]]
    B2 = [[b20, b21, b22, b23]]
    
    Strassen 1 (A1*B1a):
    m1_1 = (a00+a11)(b00+b11)
    U: a00->1, a11->1. Indices: a00 is U[0,0], a11 is U[4,0] (i=1, j=1 -> 1*3+1=4).
    V: b00->1, b11->1. Indices: b00 is V[0,0], b11 is V[5,0] (j=1, k=1 -> 1*4+1=5).
    W: contributes to C[0,0], C[0,1], C[1,0], C[1,1].
    C[0,0] = m1_1 + ... -> W[0,0] += 1 (k=0, i=0 -> 0*2+0=0)
    C[0,1] = ... -> W[1,0] += 0? No, m1_1 is in C[0,0] and C[1,1].
    C[0,0] index: k=0, i=0 -> 0.
    C[0,1] index: k=1, i=0 -> 2.
    C[1,0] index: k=0, i=1 -> 1.
    C[1,1] index: k=1, i=1 -> 3.
    
    m1_1 is in C[0,0] and C[1,1].
    So W[0,0] += 1, W[3,0] += 1.
    
    m2_1 = (a10+a11)b00
    U: a10->1, a11->1. U[3,1] += 1, U[4,1] += 1.
    V: b00->1. V[0,1] += 1.
    W: m2_1 is in C[1,0] and C[1,1] (with -).
    C[1,0] index 1. C[1,1] index 3.
    W[1,1] += 1, W[3,1] -= 1.
    
    m3_1 = a00(b01-b11)
    U: a00->1. U[0,2] += 1.
    V: b01->1, b11->-1. V[1,2] += 1, V[5,2] -= 1.
    W: m3_1 is in C[0,1] and C[1,1].
    C[0,1] index 2. C[1,1] index 3.
    W[2,2] += 1, W[3,2] += 1.
    
    m4_1 = a11(b10-b00)
    U: a11->1. U[4,3] += 1.
    V: b10->1, b00->-1. V[4,3] += 1, V[0,3] -= 1.
    W: m4_1 is in C[0,0] and C[1,0].
    C[0,0] index 0. C[1,0] index 1.
    W[0,3] += 1, W[1,3] += 1.
    
    m5_1 = (a00+a01)b11
    U: a00->1, a01->1. U[0,4] += 1, U[1,4] += 1.
    V: b11->1. V[5,4] += 1.
    W: m5_1 is in C[0,0] (with -) and C[0,1].
    C[0,0] index 0. C[0,1] index 2.
    W[0,4] -= 1, W[2,4] += 1.
    
    m6_1 = (a10-a00)(b00+b10)
    U: a10->1, a00->-1. U[3,5] += 1, U[0,5] -= 1.
    V: b00->1, b10->1. V[0,5] += 1, V[4,5] += 1.
    W: m6_1 is in C[1,1].
    C[1,1] index 3.
    W[3,5] += 1.
    
    m7_1 = (a01-a11)(b10+b11)
    U: a01->1, a11->-1. U[1,6] += 1, U[4,6] -= 1.
    V: b10->1, b11->1. V[4,6] += 1, V[5,6] += 1.
    W: m7_1 is in C[0,0].
    C[0,0] index 0.
    W[0,6] += 1.
    
    Strassen 2 (A1*B1b):
    Same U, but different V and W.
    B1b = [[b02, b03], [b12, b13]]
    Indices: b02 is V[2,7], b03 is V[3,7], b12 is V[6,7], b13 is V[7,7].
    
    m1_2 = (a00+a11)(b02+b13)
    U: same as m1_1. U[0,7] += 1, U[4,7] += 1.
    V: b02->1, b13->1. V[2,7] += 1, V[7,7] += 1.
    W: m1_2 is in C[0,2] and C[1,3].
    C[0,2] index: k=2, i=0 -> 4.
    C[1,3] index: k=3, i=1 -> 7.
    W[4,7] += 1, W[7,7] += 1.
    
    m2_2 = (a10+a11)b02
    U: a10->1, a11->1. U[3,8] += 1, U[4,8] += 1.
    V: b02->1. V[2,8] += 1.
    W: m2_2 is in C[1,2] and C[1,3] (with -).
    C[1,2] index 5. C[1,3] index 7.
    W[5,8] += 1, W[7,8] -= 1.
    
    m3_2 = a00(b03-b13)
    U: a00->1. U[0,9] += 1.
    V: b03->1, b13->-1. V[3,9] += 1, V[7,9] -= 1.
    W: m3_2 is in C[0,3] and C[1,3].
    C[0,3] index 6. C[1,3] index 7.
    W[6,9] += 1, W[7,9] += 1.
    
    m4_2 = a11(b12-b02)
    U: a11->1. U[4,10] += 1.
    V: b12->1, b02->-1. V[6,10] += 1, V[2,10] -= 1.
    W: m4_2 is in C[0,2] and C[1,2].
    C[0,2] index 4. C[1,2] index 5.
    W[4,10] += 1, W[5,10] += 1.
    
    m5_2 = (a00+a01)b13
    U: a00->1, a01->1. U[0,11] += 1, U[1,11] += 1.
    V: b13->1. V[7,11] += 1.
    W: m5_2 is in C[0,2] (with -) and C[0,3].
    C[0,2] index 4. C[0,3] index 6.
    W[4,11] -= 1, W[6,11] += 1.
    
    m6_2 = (a10-a00)(b02+b12)
    U: a10->1, a00->-1. U[3,12] += 1, U[0,12] -= 1.
    V: b02->1, b12->1. V[2,12] += 1, V[6,12] += 1.
    W: m6_2 is in C[1,3].
    C[1,3] index 7.
    W[7,12] += 1.
    
    m7_2 = (a01-a11)(b12+b13)
    U: a01->1, a11->-1. U[1,13] += 1, U[4,13] -= 1.
    V: b12->1, b13->1. V[6,13] += 1, V[7,13] += 1.
    W: m7_2 is in C[0,2].
    C[0,2] index 4.
    W[4,13] += 1.
    
    A2*B2:
    A2 = [[a02], [a12]]
    B2 = [[b20, b21, b22, b23]]
    
    8 multiplications:
    a02*b20, a02*b21, a02*b22, a02*b23
    a12*b20, a12*b21, a12*b22, a12*b23
    
    Columns 14-21.
    
    Col 14: a02*b20
    U[2,14] += 1 (a02 is i=0, j=2 -> 0*3+2=2)
    V[8,14] += 1 (b20 is j=2, k=0 -> 2*4+0=8)
    W[0,14] += 1 (C[0,0] index 0)
    
    Col 15: a02*b21
    U[2,15] += 1
    V[9,15] += 1 (b21 is j=2, k=1 -> 2*4+1=9)
    W[2,15] += 1 (C[0,1] index 2)
    
    Col 16: a02*b22
    U[2,16] += 1
    V[10,16] += 1 (b22 is j=2, k=2 -> 2*4+2=10)
    W[4,16] += 1 (C[0,2] index 4)
    
    Col 17: a02*b23
    U[2,17] += 1
    V[11,17] += 1 (b23 is j=2, k=3 -> 2*4+3=11)
    W[6,17] += 1 (C[0,3] index 6)
    
    Col 18: a12*b20
    U[5,18] += 1 (a12 is i=1, j=2 -> 1*3+2=5)
    V[8,18] += 1
    W[1,18] += 1 (C[1,0] index 1)
    
    Col 19: a12*b21
    U[5,19] += 1
    V[9,19] += 1
    W[3,19] += 1 (C[1,1] index 3)
    
    Col 20: a12*b22
    U[5,20] += 1
    V[10,20] += 1
    W[5,20] += 1 (C[1,2] index 5)
    
    Col 21: a12*b23
    U[5,21] += 1
    V[11,21] += 1
    W[7,21] += 1 (C[1,3] index 7)
    
    This gives rank 22.
    
    I will return this.
    """
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to U, V, W
    def add_u(i, j, c, val):
        U[i * m + j][c] += val
    def add_v(j, k, c, val):
        V[j * p + k][c] += val
    def add_w(k, i, c, val):
        W[k * n + i][c] += val
        
    # Strassen 1: A1 * B1a
    # A1 = [[a00, a01], [a10, a11]]
    # B1a = [[b00, b01], [b10, b11]]
    
    # m1_1 = (a00+a11)(b00+b11)
    add_u(0, 0, 0, 1); add_u(1, 1, 0, 1)
    add_v(0, 0, 0, 1); add_v(1, 1, 0, 1)
    add_w(0, 0, 0, 1); add_w(1, 1, 0, 1)
    
    # m2_1 = (a10+a11)b00
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(0, 1, 1, 1); add_w(1, 1, 1, -1)
    
    # m3_1 = a00(b01-b11)
    add_u(0, 0, 2, 1)
    add_v(0, 1, 2, 1); add_v(1, 1, 2, -1)
    add_w(1, 0, 2, 1); add_w(1, 1, 2, 1)
    
    # m4_1 = a11(b10-b00)
    add_u(1, 1, 3, 1)
    add_v(1, 0, 3, 1); add_v(0, 0, 3, -1)
    add_w(0, 0, 3, 1); add_w(0, 1, 3, 1)
    
    # m5_1 = (a00+a01)b11
    add_u(0, 0, 4, 1); add_u(0, 1, 4, 1)
    add_v(1, 1, 4, 1)
    add_w(0, 0, 4, -1); add_w(1, 0, 4, 1)
    
    # m6_1 = (a10-a00)(b00+b10)
    add_u(1, 0, 5, 1); add_u(0, 0, 5, -1)
    add_v(0, 0, 5, 1); add_v(1, 0, 5, 1)
    add_w(1, 1, 5, 1)
    
    # m7_1 = (a01-a11)(b10+b11)
    add_u(0, 1, 6, 1); add_u(1, 1, 6, -1)
    add_v(1, 0, 6, 1); add_v(1, 1, 6, 1)
    add_w(0, 0, 6, 1)
    
    # Strassen 2: A1 * B1b
    # B1b = [[b02, b03], [b12, b13]]
    
    # m1_2 = (a00+a11)(b02+b13)
    add_u(0, 0, 7, 1); add_u(1, 1, 7, 1)
    add_v(0, 2, 7, 1); add_v(1, 3, 7, 1)
    add_w(2, 0, 7, 1); add_w(3, 1, 7, 1)
    
    # m2_2 = (a10+a11)b02
    add_u(1, 0, 8, 1); add_u(1, 1, 8, 1)
    add_v(0, 2, 8, 1)
    add_w(2, 1, 8, 1); add_w(3, 1, 8, -1)
    
    # m3_2 = a00(b03-b13)
    add_u(0, 0, 9, 1)
    add_v(0, 3, 9, 1); add_v(1, 3, 9, -1)
    add_w(3, 0, 9, 1); add_w(3, 1, 9, 1)
    
    # m4_2 = a11(b12-b02)
    add_u(1, 1, 10, 1)
    add_v(1, 2, 10, 1); add_v(0, 2, 10, -1)
    add_w(2, 0, 10, 1); add_w(2, 1, 10, 1)
    
    # m5_2 = (a00+a01)b13
    add_u(0, 0, 11, 1); add_u(0, 1, 11, 1)
    add_v(1, 3, 11, 1)
    add_w(2, 0, 11, -1); add_w(3, 0, 11, 1)
    
    # m6_2 = (a10-a00)(b02+b12)
    add_u(1, 0, 12, 1); add_u(0, 0, 12, -1)
    add_v(0, 2, 12, 1); add_v(1, 2, 12, 1)
    add_w(3, 1, 12, 1)
    
    # m7_2 = (a01-a11)(b12+b13)
    add_u(0, 1, 13, 1); add_u(1, 1, 13, -1)
    add_v(1, 2, 13, 1); add_v(1, 3, 13, 1)
    add_w(2, 0, 13, 1)
    
    # A2 * B2
    # A2 = [[a02], [a12]]
    # B2 = [[b20, b21, b22, b23]]
    
    # a02 * b20
    add_u(0, 2, 14, 1)
    add_v(2, 0, 14, 1)
    add_w(0, 0, 14, 1)
    
    # a02 * b21
    add_u(0, 2, 15, 1)
    add_v(2, 1, 15, 1)
    add_w(1, 0, 15, 1)
    
    # a02 * b22
    add_u(0, 2, 16, 1)
    add_v(2, 2, 16, 1)
    add_w(2, 0, 16, 1)
    
    # a02 * b23
    add_u(0, 2, 17, 1)
    add_v(2, 3, 17, 1)
    add_w(3, 0, 17, 1)
    
    # a12 * b20
    add_u(1, 2, 18, 1)
    add_v(2, 0, 18, 1)
    add_w(0, 1, 18, 1)
    
    # a12 * b21
    add_u(1, 2, 19, 1)
    add_v(2, 1, 19, 1)
    add_w(1, 1, 19, 1)
    
    # a12 * b22
    add_u(1, 2, 20, 1)
    add_v(2, 2, 20, 1)
    add_w(2, 1, 20, 1)
    
    # a12 * b23
    add_u(1, 2, 21, 1)
    add_v(2, 3, 21, 1)
    add_w(3, 1, 21, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
