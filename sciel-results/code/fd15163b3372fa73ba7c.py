"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    We use a Kronecker product structure. Note that 2x4 * 4x6 can be seen as:
    (2x2) x (2x2) * (2x2) x (2x2) where the 4x4 block is treated as 2x2 blocks.
    
    Actually, let's think more carefully. We have A (2x4) * B (4x6) = C (2x6).
    
    We can decompose this using the fact that we can multiply 2x2 blocks.
    Let A be split into two 2x2 blocks: A = [A1 A2] where A1, A2 are 2x2.
    Let B be split into three 2x2 blocks: B = [B1; B2; B3] where B1, B2, B3 are 2x2.
    
    Then C = A1*B1 + A2*B2 + A2*B3... wait, that's not right.
    
    Let me reconsider. A is 2x4, B is 4x6.
    A = [a11 a12 a13 a14]
        [a21 a22 a23 a24]
    
    B = [b11 b12 b13 b14 b15 b16]
        [b21 b22 b23 b24 b25 b26]
        [b31 b32 b33 b34 b35 b36]
        [b41 b42 b43 b44 b45 b46]
    
    C = A * B is 2x6.
    C[0,0] = a11*b11 + a12*b21 + a13*b31 + a14*b41
    
    This is just standard matrix multiplication. The rank of the tensor is at most 2*4*6 = 48.
    
    Let me try a different approach using Kronecker products.
    
    Note that 2x4 * 4x6 = (2x1) x (1x4) * (4x1) x (1x6) ... no that doesn't work.
    
    Actually, we can write this as a composition of smaller multiplications.
    
    Let's use the fact that we can multiply 2x2 matrices with rank 7 (Strassen) or rank 7.
    
    Wait, let me think about this differently. We have dimensions 2, 4, 6.
    
    We can decompose 4 = 2*2 and 6 = 3*2 or 2*3.
    
    Let's try: A (2x4) = A (2x2*2), B (4x6) = B (2*2 x 2*3)
    
    So A = [A1 A2] where A1, A2 are 2x2.
    B = [[B11 B12], [B21 B22]] where each Bij is 2x3.
    
    Then C = A1*B11 + A2*B21 (left 2x3) and A1*B12 + A2*B22 (right 2x3).
    
    So C = [C_left C_right] where C_left = A1*B11 + A2*B21 and C_right = A1*B12 + A2*B22.
    
    Each of these is a sum of two 2x2 * 2x3 products.
    
    For 2x2 * 2x3, the rank is at most 2*2*3 = 12.
    
    So total rank would be at most 2*12 + 2*12 = 48. Same as before.
    
    Let me try to be smarter. 
    
    Actually, let's use the Strassen algorithm for 2x2 matrix multiplication.
    
    We have 2x2 * 2x2 with rank 7.
    
    For 2x2 * 2x3: we can pad to 2x2 * 2x4 (by adding a zero column) or split the 3 into 2+1.
    
    Let me try: 2x2 * 2x3. Split the 3 into 2+1.
    A (2x2) * [B1 B2] where B1 is 2x2, B2 is 2x1.
    = A*B1 + A*B2
    
    A*B1 is 2x2*2x2, rank 7 (Strassen).
    A*B2 is 2x2*2x1, which is just a vector-matrix product. Rank is at most 2*2*1 = 4.
    
    So 2x2*2x3 has rank at most 7+4 = 11.
    
    Then for our problem:
    C_left = A1*B11 + A2*B21 where each term is 2x2 * 2x3.
    We can compute this as:
    Let X = A1 + A2, Y = A1 - A2, Z = B11 + B21, W = B11 - B21.
    
    Actually, this is getting complicated. Let me just implement a direct approach with rank 48 first, then try to optimize.
    
    Actually, let me try a different decomposition.
    
    We can use the fact that matrix multiplication is bilinear.
    
    For 2x4 * 4x6, we can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor has 2*4*6 = 48 components.
    
    Let me try to use a Kronecker product structure.
    
    2 = 2, 4 = 2*2, 6 = 3*2
    
    So we can write this as (2x2) x (1x2) * (2x2) x (3x1) ... no.
    
    Let me just implement the straightforward rank-48 solution and see if I can find a better one.
    
    Actually, I recall that for m x n * n x p, the rank is at most m*n*p, and there are known bounds.
    
    For 2x4 * 4x6, let me try to use the fact that we can do 2x2 multiplications.
    
    Split A into two 2x2 matrices: A = [A1 A2]
    Split B into four 2x2 matrices: B = [[B11 B12], [B21 B22]]
    
    Then C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each term like A1*B11 is 2x2 * 2x2. Using Strassen, this has rank 7.
    
    So C_left = A1*B11 + A2*B21 and C_right = A1*B12 + A2*B22.
    
    We need to compute two sums of 2x2*2x2 products.
    
    For C_left: we need A1*B11 + A2*B21.
    We can use the identity:
    (A1+A2)*(B11+B21) = A1*B11 + A1*B21 + A2*B11 + A2*B21
    
    So A1*B11 + A2*B21 = (A1+A2)*(B11+B21) - A1*B21 - A2*B11
    
    That's 3 products of 2x2*2x2, each of rank 7, so total rank 21 for C_left.
    
    Similarly for C_right: A1*B12 + A2*B22 = (A1+A2)*(B12+B22) - A1*B22 - A2*B12, rank 21.
    
    Total rank: 21 + 21 = 42.
    
    That's better than 48!
    
    Let me verify this. We have:
    C_left = (A1+A2)*(B11+B21) - A1*B21 - A2*B11
    
    Let's check:
    (A1+A2)*(B11+B21) = A1*B11 + A1*B21 + A2*B11 + A2*B21
    Subtract A1*B21 + A2*B11:
    = A1*B11 + A2*B21. Correct!
    
    So the rank is 3*7 + 3*7 = 42.
    
    Now, can we do better? Let me think...
    
    Actually, for 2x2*2x2, the optimal rank is 7 (Strassen).
    
    So with this approach, we get rank 42.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We'll use the Strassen algorithm for 2x2 matrix multiplication.
    # Strassen computes C = A*B for 2x2 matrices using 7 multiplications.
    
    # The 7 products in Strassen are:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    
    # Then:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Now for our problem:
    # A is 2x4, split into A1, A2 (each 2x2)
    # B is 4x6, split into B11, B12, B21, B22 (each 2x3)
    
    # Wait, B is 4x6, so splitting into 2x2 blocks gives 2x3 blocks.
    # B = [[B11 B12], [B21 B22]] where each is 2x3.
    
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    # Each of these is a sum of two 2x2 * 2x3 products.
    
    # For 2x2 * 2x3, we can use Strassen-like approach.
    # Split the 2x3 into 2x2 + 2x1.
    
    # Actually, let me just use the direct approach for 2x2*2x3.
    # 2x2 * 2x3 has rank at most 12.
    
    # But we can do better. Let's use the fact that:
    # A * [B1 B2] = [A*B1 A*B2]
    
    # So we need to compute A*B1 (2x2*2x2) and A*B2 (2x2*2x1).
    
    # A*B1 has rank 7 (Strassen).
    # A*B2: this is just a linear combination of columns of A.
    # A*B2 = [a11*b11 + a12*b21, a11*b12 + a12*b22; a21*b11 + a22*b21, a21*b12 + a22*b22]
    # This has rank at most 4 (each entry is a product of two variables).
    
    # So 2x2*2x3 has rank at most 7+4 = 11.
    
    # For C_left = A1*B11 + A2*B21:
    # We can write this as (A1+A2)*(B11+B21) - A1*B21 - A2*B11
    # Each term is 2x2*2x3, rank 11.
    # Total rank for C_left: 3*11 = 33.
    
    # Similarly for C_right: 33.
    
    # Total: 66. That's worse than 48!
    
    # Let me reconsider.
    
    # Actually, I think I made an error. Let me re-examine.
    
    # For 2x2*2x3, the rank is at most 2*2*3 = 12.
    # Using Strassen for the 2x2 part and direct for the 2x1 part:
    # A*B1 (2x2*2x2): rank 7
    # A*B2 (2x2*2x1): rank 4
    # Total: 11.
    
    # But wait, when we compute (A1+A2)*(B11+B21), we're multiplying 2x2 * 2x3.
    # So each such product has rank 11.
    
    # So C_left has rank 3*11 = 33.
    # C_right has rank 33.
    # Total: 66.
    
    # That's worse than 48. So the Strassen approach doesn't help here.
    
    # Let me just use the direct rank-48 approach.
    
    rank = n * m * p  # 48
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][idx] = 1.0
                V[v_idx][idx] = 1.0
                W[w_idx][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
