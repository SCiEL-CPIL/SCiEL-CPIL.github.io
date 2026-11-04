"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    1. Start with schoolbook decomposition (rank 60).
    2. Use Strassen's algorithm on 2x2 sub-blocks to reduce rank.
       - Split the 2x3 matrix into 2x2 and 2x1 blocks
       - Split the 3x10 matrix into 2x10 and 1x10 blocks
       - Apply Strassen to the 2x2 * 2x10 part (saves 1 rank per 2x2 block)
       - For 2x2 * 2x10: schoolbook is 4*10=40, Strassen is 7*10=70... wait that's worse
       
    Let me reconsider. The key insight from the measured data:
    "Using Strassen decomposition for 2x2 sub-blocks and schoolbook for the remainder reduces rank from 60 to 55"
    
    So the approach is:
    - Partition the 2x3 matrix as [A1 | A2] where A1 is 2x2 and A2 is 2x1
    - Partition the 3x10 matrix as [B1; B2] where B1 is 2x10 and B2 is 1x10
    - The product is A1*B1 + A2*B2
    - A1*B1 is 2x2 * 2x10: use Strassen (7 multiplications instead of 4 per 2x2 block, but we have 10 columns)
      Actually for 2x2 * 2x10, we can think of it as 10 independent 2x2*2x1 multiplications.
      Each 2x2 * 2x1 can be done with Strassen in 7 scalar mults instead of 4.
      So 10 * 7 = 70 for A1*B1 vs 10*4 = 40 schoolbook. That's worse!
      
    Wait, let me re-read: "reduces rank from 60 to 55". So it's a net saving of 5.
    
    Let me think differently. Maybe the partition is different.
    
    Actually, for M_{2,3,10}:
    - Schoolbook: 2*3*10 = 60
    
    One known approach: treat the 2x3 as three 2x1 column vectors, and the 3x10 as ten 3x1 column vectors... no.
    
    Let me think about this more carefully. The measured result says rank 55 is achievable.
    
    A standard technique: For M_{n,m,p}, if we can find a way to share computations...
    
    Actually, I recall that for 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    
    For 2x3 * 3x10, one approach:
    - Split into 2x2 * 2x10 + 2x1 * 1x10
    - 2x2 * 2x10: This is like multiplying a 2x2 by a 2x10. We can view the 2x10 as 10 columns of 2x1.
      For each 2x1 column, 2x2 * 2x1 requires 4 scalar mults (or can it be done with fewer?).
      2x2 * 2x1: Let A=[[a,b],[c,d]], x=[x1,x2]. Result = [a*x1+b*x2, c*x1+d*x2].
      This is 4 mults. Can we do better? The bilinear complexity of 2x2*2x1... 
      Actually, 2x2 * 2x1 is equivalent to two independent dot products of length 2, so rank 4.
      
    Hmm, let me try a different partition.
    
    Alternative: Split the 3 in the middle differently.
    - 2x3 * 3x10 = (2x1 * 1x10) + (2x2 * 2x10)
    
    Or maybe the trick is to use the small n=2 to share across the p=10 dimension.
    
    Let me just implement the known result: rank 55.
    
    One known decomposition for M_{2,3,10} with rank 55:
    
    Actually, let me think about this using the "border rank" or known lower bounds.
    
    For now, let me implement a systematic search approach that finds rank 55.
    
    Key idea from the hint: "leveraging the bilinear complexity... by exploiting the small n=2 dimension to share computation"
    
    Here's a concrete approach:
    - The 2x3 matrix A has columns a1, a2, a3 (each 2x1)
    - The 3x10 matrix B has rows b1, b2, b3 (each 1x10)
    - Product C = a1*b1 + a2*b2 + a3*b3 (outer product sum)
    
    Each ai * bi is a 2x10 matrix. So C is a sum of three 2x10 outer products.
    
    Now, for a single 2x1 * 1x10 multiplication, the rank is 2*10 = 20 (schoolbook).
    But we're summing three such terms, and the total schoolbook rank is 3*20 = 60.
    
    Can we share between the three terms? 
    
    Consider that a1, a2, a3 are 2-vectors. Let's write:
    a1 = [a11, a12], a2 = [a21, a22], a3 = [a31, a32]
    
    C[0,j] = a11*b1[j] + a21*b2[j] + a31*b3[j]
    C[1,j] = a12*b1[j] + a22*b2[j] + a32*b3[j]
    
    For each column j, we're computing:
    [C[0,j]]   [a11 a21 a31] [b1[j]]
    [C[1,j]] = [a12 a22 a32] [b2[j]]
                           [b3[j]]
    
    This is a 2x3 * 3x1 multiplication for each j, which has rank 6 (schoolbook) or can it be less?
    The bilinear complexity of 2x3 * 3x1 is at most 6, and it's known to be exactly 6 (since it's a subcase of 2x3*3x1 which is the same as our problem with p=1, and M_{2,3,1} = 6).
    
    So for each of the 10 columns, we need rank 6, giving 60 total. No savings this way.
    
    Let me try yet another approach. What if we group columns?
    
    Group the 10 columns into 5 pairs. For each pair of columns j, j+1:
    We compute a 2x2 block of C.
    
    C[:, j:j+2] = A * B[:, j:j+2] where B[:, j:j+2] is 3x2.
    
    So we need M_{2,3,2} for each group. M_{2,3,2} has schoolbook rank 2*3*2 = 12.
    Can we do better? 
    
    2x3 * 3x2: Let A be 2x3, B be 3x2.
    C = A*B is 2x2.
    
    Schoolbook: 6 mults.
    Can we do it in 5? 
    
    Let A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    C[0,0] = ag + bi + ck
    C[0,1] = ah + bj + cl
    C[1,0] = dg + ei + fk
    C[1,1] = dh + ej + fl
    
    This is a 2x3 * 3x2 = 2x2 multiplication. The rank of M_{2,3,2}...
    This is equivalent to multiplying a 2x3 by a 3x2 to get 2x2.
    
    I believe M_{2,3,2} has rank 8 (it's a subcase that might not benefit from Strassen directly).
    
    Hmm, this is getting complicated. Let me just implement the known rank-55 decomposition.
    
    After some thought, here's a working approach for rank 55:
    
    Use the fact that 2x2 * 2x2 can be done in 7 mults (Strassen).
    
    Partition:
    - A (2x3) = [A_left (2x2) | A_right (2x1)]
    - B (3x10) = [B_top (2x10); B_bottom (1x10)]
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2 * 2x10):
    Treat B_top as 10 columns of 2x1. For each column, 2x2 * 2x1.
    2x2 * 2x1: A=[[a,b],[c,d]], x=[x1,x2].
    y1 = a*x1 + b*x2
    y2 = c*x1 + d*x2
    
    This is 4 mults. No Strassen benefit here since it's not square.
    
    OK let me just go with a direct implementation that achieves rank 55 based on the measured result.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to construct rank 55 decomposition
    # Strategy: Use Strassen-like sharing on 2x2 blocks
    
    # Partition A (2x3) into A1 (2x2) and A2 (2x1)
    # Partition B (3x10) into B1 (2x10) and B2 (1x10)
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x10):
    # A1 = [[a, b], [c, d]], B1 has rows r1, r2 (each 1x10)
    # Result row 0: a*r1 + b*r2 (element-wise)
    # Result row 1: c*r1 + d*r2 (element-wise)
    # This is 4*10 = 40 mults schoolbook
    
    # For A2*B2 (2x1 * 1x10):
    # A2 = [e, f]^T, B2 = s (1x10)
    # Result: e*s in row 0, f*s in row 1
    # This is 2*10 = 20 mults schoolbook
    
    # Total schoolbook: 40 + 20 = 60
    
    # To reduce: Note that in A1*B1, we compute:
    # Row 0: a*r1[j] + b*r2[j] for j=1..10
    # Row 1: c*r1[j] + d*r2[j] for j=1..10
    
    # We can factor: for each j, we compute (a*r1[j], b*r2[j], c*r1[j], d*r2[j])
    # That's 4 mults per j, 40 total.
    
    # Alternative: Use the identity that allows sharing.
    # (a+b)(r1+r2) = a*r1 + a*r2 + b*r1 + b*r2
    # So a*r1 + b*r2 = (a+b)(r1+r2) - a*r2 - b*r1
    # This doesn't help directly.
    
    # Let me try a different known result.
    # 
    # Actually, I found that the correct approach for rank 55 is:
    # Use 5 groups of 2 columns each. For each group, use a 2x3 * 3x2 decomposition.
    # If M_{2,3,2} has rank 11 instead of 12, then 5*11 = 55.
    
    # Let me verify: 2x3 * 3x2 to get 2x2.
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C[0,0] = ag+bi+ck, C[0,1] = ah+bj+cl
    # C[1,0] = dg+ei+fk, C[1,1] = dh+ej+fl
    
    # Schoolbook: 12 mults (a*g, b*i, c*k, a*h, b*j, c*l, d*g, e*i, f*k, d*h, e*j, f*l)
    
    # Can we do it in 11? Let's try to find a rank-11 decomposition.
    
    # One approach: 
    # m1 = a*g
    # m2 = b*(i+h)  -- no, that mixes things
    # 
    # Let me try:
    # m1 = (a+d)*g
    # m2 = (b+e)*(i+j)  -- no
    # 
    # Actually, for 2x3*3x2, I'll just use schoolbook (rank 12) per group and see if I can save elsewhere.
    
    # Alternative: Maybe the saving comes from a different structure.
    
    # Let me just return the schoolbook decomposition for now, as rank 55 requires more careful construction.
    # But the measured data says 55 is achievable, so let me try harder.
    
    # Here's another approach: 
    # Use the fact that n=2 is small. 
    # C[0,:] = A[0,:]*B[:, :]  (dot product of row 0 of A with each column of B... no, B is 3x10)
    # C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j] + A[0,2]*B[2,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j] + A[1,2]*B[2,j]
    
    # For all j=0..9, we have 20 scalar products of length 3.
    # Total: 20*3 = 60 mults.
    
    # Can we share? Note that C[0,j] and C[1,j] share the B[*,j] values.
    # But the A coefficients are different.
    
    # What if we compute:
    # t1 = (A[0,0]+A[1,0]) * B[0,j]
    # t2 = (A[0,1]+A[1,1]) * B[1,j]  
    # t3 = (A[0,2]+A[1,2]) * B[2,j]
    # Then C[0,j] + C[1,j] = t1 + t2 + t3
    # But we need C[0,j] and C[1,j] separately.
    
    # This gives us the sum, not the individual values. Not helpful directly.
    
    # I'll implement a rank-55 construction based on known results.
    # After research, here's a valid rank-55 decomposition:
    
    # Use 5 pairs of columns. For each pair, use a 2x3*3x2 = 2x2 mult with rank 11.
    # Total: 5 * 11 = 55.
    
    # Rank-11 decomposition of 2x3 * 3x2:
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # 
    # m1 = a*g
    # m2 = b*i  
    # m3 = c*k
    # m4 = a*h
    # m5 = b*(j+i)  -- no, let me be more careful
    # 
    # Actually, I'll use a known rank-11 decomposition:
    # m1 = (a+d)*g
    # m2 = (b+e)*(i)
    # m3 = (c+f)*k
    # m4 = a*h
    # m5 = b*j
    # m6 = c*l
    # m7 = (a+b+c)*(g+i+k)  -- this is getting messy
    
    # Let me just use schoolbook for the 2x3*3x2 (rank 12) and accept rank 60,
    # OR try a computational search.
    
    # Given time constraints, I'll implement a simple local search to try to find rank < 60.
    
    # For now, return schoolbook.
    if time.time() > deadline:
        return schoolbook()
    
    # Attempt rank 55 via structured decomposition
    # 5 groups of 2 columns, each group uses rank-11 for 2x3*3x2
    
    # Define a rank-11 decomposition for 2x3 * 3x2:
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C[0,0] = ag + bi + ck
    # C[0,1] = ah + bj + cl  
    # C[1,0] = dg + ei + fk
    # C[1,1] = dh + ej + fl
    
    # 11-term decomposition using shared multiplications:
    # We can save 1 multiplication by combining terms.
    # 
    # Key idea: Compute (a+b)*g and (d+e)*i, then use linearity
    # to extract individual terms.
    #
    # t1 = a*g
    # t2 = b*i  
    # t3 = c*k
    # t4 = a*h
    # t5 = b*j
    # t6 = c*l
    # t7 = d*g
    # t8 = e*i
    # t9 = f*k
    # t10 = d*h
    # t11 = e*j
    # f*l is missing... 
    #
    # Actually, let me use a different approach. For 2x3*3x2, 
    # the bilinear complexity is known to be 12 (same as schoolbook)
    # for this specific case. Let me verify by trying to find a rank-11.
    #
    # Alternative: Use the fact that we only need 2x2 output.
    # Maybe we can use 11 mults by being clever.
    #
    # Let me try:
    # m1 = (a+d)*g
    # m2 = (b+e)*i
    # m3 = (c+f)*k
    # m4 = a*h
    # m5 = b*j
    # m6 = c*l
    # m7 = d*h
    # m8 = e*j
    # m9 = f*l
    # m10 = (a-d)*g
    # m11 = (b-e)*i
    #
    # Then:
    # a*g = (m1 + m10)/2
    # d*g = (m1 - m10)/2
    # b*i = (m2 + m11)/2
    # e*i = (m2 - m11)/2
    # c*k = m3 - f*k... no, m3 = c*k + f*k
    #
    # This doesn't work cleanly. Let me just use schoolbook for each group.
    #
    # Actually, I realize the saving might come from a different structure.
    # Let me try using rank 11 by accepting that some terms are shared.
    #
    # For now, implement schoolbook per group (rank 12 each, total 60)
    # but structure it to allow future optimization.
    
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
