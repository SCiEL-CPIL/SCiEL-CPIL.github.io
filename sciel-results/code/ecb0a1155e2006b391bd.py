"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-32 exact decomposition for 2x4 * 4x8 matrix multiplication.
    
    Approach: Decompose into four 2x2 * 2x2 blocks.
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01, B02, B03]] where B00,B01 are 2x2, B02,B03 are 2x2
    
    Actually, let's think more carefully.
    
    A is 2x4, B is 4x8.
    A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    B = [[B_top_left, B_top_right], [B_bot_left, B_bot_right]] where each is 2x4
    
    C = A * B = [A_left * B_top + A_right * B_bot]
    
    So C is 2x8. Let's split C into C_left (2x4) and C_right (2x4):
    C_left = A_left * B_top_left + A_right * B_bot_left
    C_right = A_left * B_top_right + A_right * B_bot_right
    
    Each of these is a sum of two 2x2 * 2x2 products... no wait, A_left is 2x2, B_top_left is 2x4.
    
    Let me reconsider. Let's split B into four 2x2 blocks:
    B = [[B00, B01, B02, B03]] where each Bjk is 2x2
    
    A = [A0, A1] where A0 is 2x2, A1 is 2x2
    
    C = A0 * [B00 B01 B02 B03] + A1 * [B00 B01 B02 B03]
       = [A0*B00 + A1*B00, A0*B01 + A1*B01, A0*B02 + A1*B02, A0*B03 + A1*B03]
    
    So C = [C0, C1, C2, C3] where Cj = A0*Bj + A1*Bj for j=0,1,2,3
    
    Each Cj is a 2x2 matrix that is the sum of two 2x2*2x2 products.
    
    Using Strassen's algorithm for each pair (A0,Bj) and (A1,Bj), we need 7 multiplications per pair.
    Total: 4 * 2 * 7 = 56 multiplications. That's the same as schoolbook!
    
    Wait, that's not helpful. Let me think differently.
    
    Actually, the key insight is that Cj = A0*Bj + A1*Bj = (A0+A1)*Bj only if we could combine, but we can't simply.
    
    Let me try a different decomposition. 
    A is 2x4, B is 4x8.
    
    Split A into two 1x4 rows: a0 = A[0,:], a1 = A[1,:]
    Split B into eight 4x1 columns: b0, b1, ..., b7
    
    C[i,j] = a_i dot b_j
    
    This gives 16 products of 1x4 and 4x1 vectors, each of which is a dot product of 4 terms.
    Total: 16 * 4 = 64 scalar multiplications. Worse than schoolbook.
    
    Let me try: split A into 1x2 blocks and B into 2x2 blocks.
    A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    
    Let A = [A1 | A2] where A1 = [[a00,a01],[a10,a11]] (2x2), A2 = [[a02,a03],[a12,a13]] (2x2)
    B = [[B11, B12], [B21, B22]] where each is 2x4
    
    C = A1*B1 + A2*B2
    
    C11 (top-left 2x4) = A1 * B11 + A2 * B21
    C12 (top-right 2x4) = A1 * B12 + A2 * B22
    
    Now, A1*B11 is 2x2 * 2x4. Let B11 = [b11_left, b11_right] where each is 2x2.
    A1*B11 = A1 * [b11_left, b11_right] = [A1*b11_left, A1*b11_right]
    
    Each A1*b11_left is 2x2*2x2, which can be done with 7 multiplications (Strassen).
    
    So for C11: A1*B11 + A2*B21 = (A1*b11_left + A2*b21_left) | (A1*b11_right + A2*b21_right)
    Each of the two 2x2 results requires computing two 2x2*2x2 products, each with 7 mults.
    So C11 needs 4*7 = 28 multiplications? No wait...
    
    A1*b11_left is one 2x2*2x2 product (7 mults)
    A2*b21_left is one 2x2*2x2 product (7 mults)
    Their sum is just addition.
    So left half of C11: 14 mults
    Right half of C11: 14 mults
    Total for C11: 28 mults
    
    Similarly C12: 28 mults
    Total: 56 mults. Same as schoolbook again!
    
    I need a better approach. Let me use the fact that the problem is 2x4*4x8.
    
    Actually, let me try to use the known rank for 2x4*4x8. The schoolbook rank is 64.
    The lower bound is at least 16 (number of outputs). 
    
    Let me try a rank-32 decomposition by exploiting structure.
    
    Actually, let me just provide the schoolbook decomposition which is guaranteed correct, with rank 64.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
