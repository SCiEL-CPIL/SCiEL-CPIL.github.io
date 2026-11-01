import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # We aim for a rank-24 decomposition.
    # Strategy:
    # 1. Split the middle dimension m=3 into j=0, j=1, j=2.
    # 2. For j=0, use schoolbook for all k (2x1x5 -> rank 2*5=10).
    #    Wait, 2x1x5 rank is 10.
    #    For j=1,2, we have a 2x2x5 block.
    #    We can split the 2x2x5 block into:
    #       - A 2x2x2 block (k=0,1) and a 2x2x3 block (k=2,3,4).
    #       - Or other splits.
    
    # Let's try a different split to minimize rank.
    # Known rank bounds:
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,3) <= 12 (Known: R(2,2,3)=11? No, R(2,2,3) is 11 or 12. Actually R(2,2,3)=11 is conjectured but 12 is proven. Wait, R(2,2,3) = 11 is not proven. Best known is 11? No, 12. Let's assume 12 for safety or check.)
    # Actually, R(2,2,3) is known to be 11? No. 
    # Let's look up R(2,2,p).
    # R(2,2,1)=4, R(2,2,2)=7, R(2,2,3)=11? No, R(2,2,3)=11 is a lower bound? 
    # Actually, R(2,2,3) = 11 is not known. The best upper bound is 12.
    # If we use 12 for R(2,2,3):
    # Block j=0: 2x1x5. Rank = 10 (schoolbook: sum_{i,k} A_{i0} B_{0k}).
    # Block j=1,2: 2x2x5.
    #   Split 2x2x5 into 2x2x2 (k=0,1) and 2x2x3 (k=2,3,4).
    #   R(2,2,2) = 7.
    #   R(2,2,3) = 12 (using known decomposition).
    #   Total for j=1,2: 7 + 12 = 19.
    # Total rank = 10 + 19 = 29. This matches the current program.
    
    # Can we do better?
    # What if we group j=0,1,2 differently?
    # Or use a mixed decomposition.
    
    # Let's try to decompose 2x3x5 directly.
    # R(2,3,5) >= ?
    # Lower bound: R(2,3,5) >= max(R(2,3,1), R(2,1,5), R(1,3,5)) ...
    # R(2,3,1) = 6 (2x3 matrix times 3x1 vector -> rank 6? No, rank is number of terms. C_i = sum_j A_ij B_j. This is rank 3? No.
    # Tensor T_{ijk} = delta_{...}.
    # For p=1, C_i = A_i . b. This is just a matrix-vector product.
    # The tensor is T_{ij0} = delta_{...} no, T_{ij0} is the coefficient of A_ij B_{00} in C_{i0}.
    # C_{i0} = sum_j A_{ij} B_{j0}.
    # So T_{ij0} = 1 for all i,j.
    # This is a rank-1 tensor? No.
    # T_{ij0} = u_i v_j w_0? No, T_{ij0} depends on i and j.
    # It is the tensor of the matrix A itself.
    # The rank of the "multiplication tensor" for p=1 is the rank of the matrix of coefficients?
    # No, the tensor is T_{ijk} where C_k = sum_{ij} T_{ijk} A_{ij} B_{jk}.
    # For p=1, C_0 = sum_{ij} A_{ij} B_{j0}.
    # The tensor entries are T_{ij0} = 1 if j is connected? No.
    # The term is A_{ij} B_{j0}. So T_{ij0} = 1 for all i,j.
    # This tensor is of the form sum_k (a^{(k)} \otimes b^{(k)} \otimes c^{(k)}).
    # For p=1, the rank is the number of pairs (i,j) that contribute?
    # No, we can factor.
    # C_i = sum_j A_{ij} B_{j0}.
    # This is rank 1? No.
    # Let's check R(2,1,1).
    # C_0 = A_0 B_0 + A_1 B_1. Rank 2.
    # C_1 = A_2 B_0 + A_3 B_1. Rank 2.
    # Total rank 4?
    # Can we do it in 2?
    # M1 = (A0+A2)B0 -> C0 += M1, C1 += M1
    # M2 = (A1+A3)B1 -> C0 += M2, C1 += M2
    # C0 = M1+M2 = A0B0 + A2B0 + A1B1 + A3B1. 
    # But we need C0 = A0B0 + A1B1.
    # So M1 must be A0B0 + A1B1? No, M1 is a product of linear forms.
    # So R(2,1,1) = 2.
    
    # Back to 2x3x5.
    # Current score 1.0345 corresponds to rank 29.
    # Seed rank is likely 30 (schoolbook).
    # 30/29 = 1.034.
    # If we find rank 28, score = 30/28 = 1.071.
    # If rank 27, score = 30/27 = 1.111.
    
    # Is rank 28 possible?
    # R(2,3,5) >= ?
    # Using the "slice rank" or other bounds.
    # Let's try to improve the 2x2x3 block.
    # R(2,2,3) = 11?
    # If R(2,2,3) = 11, then total rank = 10 (for j=0) + 7 (for k=0,1) + 11 (for k=2,3,4) = 28.
    
    # Does a rank-11 decomposition for 2x2x3 exist?
    # Yes, it is known that R(2,2,3) <= 11.
    # Reference: "The rank of the tensor for 2x2x3 matrix multiplication is 11" (Landsberg, etc.)
    # Actually, I recall R(2,2,3)=11 is a conjecture or a result?
    # It is known that R(2,2,3) = 11.
    
    # Let's construct the rank-11 decomposition for 2x2x3.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1,2}.
    # We need 11 columns.
    
    # A known decomposition for 2x2x3 (from literature):
    # It can be constructed by extending Strassen (7 terms for 2x2x2) and adding 4 terms for the 3rd slice?
    # Or using a specific algorithm.
    
    # Let's try to find or derive it.
    # One approach: Use the fact that R(2,2,3) <= R(2,2,2) + R(2,2,1) + ...?
    # No, subadditivity: R(A,B,C) <= R(A,B,C1) + R(A,B,C2).
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # So we can just take the Strassen decomposition for the first 2 k's, and schoolbook for the 3rd k.
    # Wait, if we do that, the U, V, W matrices for the 2nd part must be zero on the first 2 k's.
    # Yes, that is valid.
    # So:
    # Part 1: 2x2x2 (k=0,1). Rank 7.
    # Part 2: 2x2x1 (k=2). Rank 4 (schoolbook).
    # Total rank 11.
    
    # So the total rank for 2x3x5 would be:
    # j=0: 2x1x5. Rank 10.
    # j=1,2:
    #   k=0,1: 2x2x2. Rank 7.
    #   k=2,3,4: 2x2x3.
    #     Split k=2,3,4 into k=2 (2x2x1, rank 4) and k=3,4 (2x2x2, rank 7).
    #     So 2x2x3 rank <= 11.
    #   Total for j=1,2: 7 + 11 = 18.
    # Total rank = 10 + 18 = 28.
    
    # Let's verify this construction.
    # Block 1: j=0.
    #   i=0,1; k=0..4.
    #   Terms: A_{i0} B_{0k}.
    #   2 * 5 = 10 terms.
    #   U[i*3+0] = 1, V[0*5+k] = 1, W[k*2+i] = 1.
    
    # Block 2: j=1,2.
    #   Sub-block 2a: k=0,1. (2x2x2)
    #   Use Strassen on i=0,1; j=1,2; k=0,1.
    #   Map j' = j-1. k' = k.
    #   7 terms.
    
    #   Sub-block 2b: k=2. (2x2x1)
    #   Terms: A_{i,j} B_{j,2} for i=0,1; j=1,2.
    #   4 terms.
    #   U[i*3+j] = 1, V[j*5+2] = 1, W[2*2+i] = 1.
    
    #   Sub-block 2c: k=3,4. (2x2x2)
    #   Use Strassen on i=0,1; j=1,2; k=3,4.
    #   Map j' = j-1. k' = k-3.
    #   7 terms.
    
    # Total rank = 10 + 7 + 4 + 7 = 28.
    
    # Let's implement this.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, k=0..4
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j=1,2, k=0,1 (Strassen 2x2x2)
    # Map: i->i, j'->j-1, k'->k
    # Strassen factors for 2x2x2
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, -C11
    # M3: A00(B01-B11) -> C01, C11
    # M4: A11(B10-B00) -> C00, C10
    # M5: (A00+A01)B11 -> -C00, C01
    # M6: (A10-A00)(B00+B01) -> C11
    # M7: (A01-A11)(B10+B11) -> C00
    
    # Indices in local 2x2x2:
    # u: i*2 + j'
    # v: j'*2 + k'
    # w: k'*2 + i
    
    # M1
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 0][col] = 1
    W[0 * n + 1][col] = 1; W[1 * n + 1][col] = -1
    col += 1
    
    # M3
    U[0 * m + 1][col] = 1
    V[1 * p + 1][col] = 1; V[2 * p + 1][col] = -1
    W[1 * n + 0][col] = 1; W[1 * n + 1][col] = 1
    col += 1
    
    # M4
    U[1 * m + 2][col] = 1
    V[2 * p + 0][col] = 1; V[1 * p + 0][col] = -1
    W[0 * n + 0][col] = 1; W[0 * n + 1][col] = 1
    col += 1
    
    # M5
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = -1; W[1 * n + 0][col] = 1
    col += 1
    
    # M6
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 0][col] = 1; V[1 * p + 1][col] = 1
    W[1 * n + 1][col] = 1
    col += 1
    
    # M7
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 0][col] = 1; V[2 * p + 1][col] = 1
    W[0 * n + 0][col] = 1
    col += 1
    
    # Block 2b: j=1,2, k=2 (Schoolbook 2x2x1)
    # Terms: A_{i,j} B_{j,2}
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j=1,2, k=3,4 (Strassen 2x2x2)
    # Map: i->i, j'->j-1, k'->k-3
    # M1
    U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M2
    U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
    V[1 * p + 3][col] = 1
    W[3 * n + 1][col] = 1; W[4 * n + 1][col] = -1
    col += 1
    
    # M3
    U[0 * m + 1][col] = 1
    V[1 * p + 4][col] = 1; V[2 * p + 4][col] = -1
    W[4 * n + 0][col] = 1; W[4 * n + 1][col] = 1
    col += 1
    
    # M4
    U[1 * m + 2][col] = 1
    V[2 * p + 3][col] = 1; V[1 * p + 3][col] = -1
    W[3 * n + 0][col] = 1; W[3 * n + 1][col] = 1
    col += 1
    
    # M5
    U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
    V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = -1; W[4 * n + 0][col] = 1
    col += 1
    
    # M6
    U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
    V[1 * p + 3][col] = 1; V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # M7
    U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
    V[2 * p + 3][col] = 1; V[2 * p + 4][col] = 1
    W[3 * n + 0][col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}
