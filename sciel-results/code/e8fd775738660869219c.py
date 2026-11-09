"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 6x6 * 6x6.
    
    The schoolbook rank is 216.
    We can improve this by using block structure and known algorithms.
    
    However, finding the true rank-minimal decomposition for 6x6x6 is a hard open problem.
    The best known upper bounds for 6x6 matrix multiplication are often derived from 
    tensor product constructions or specific algorithms.
    
    One simple improvement is to use the Strassen-like recursive structure or 
    known low-rank decompositions for smaller blocks and combine them.
    
    But for a "from scratch" approach without heavy pre-computed lookup tables, 
    we might try a simple block decomposition.
    
    Let's try a 3x3 block structure where each block is 2x2.
    The rank of 2x2 matrix multiplication is 7 (Strassen).
    The rank of 3x3 matrix multiplication is 21 (or 23, depending on the algorithm).
    The total rank would be (rank of 3x3 mult) * (rank of 2x2 mult) = 21 * 7 = 147.
    
    This is a valid decomposition! 
    The tensor for 6x6 multiplication can be viewed as a 3x3 block matrix of 2x2 blocks.
    Multiplying two 6x6 matrices is equivalent to multiplying two 3x3 block matrices 
    where each entry is a 2x2 matrix.
    The multiplication of the blocks follows the rules of 3x3 matrix multiplication,
    and the internal block multiplications follow the rules of 2x2 matrix multiplication.
    
    The tensor T_{6,6,6} can be factored as T_{3,3,3} ⊗ T_{2,2,2}.
    The rank is at most rank(T_{3,3,3}) * rank(T_{2,2,2}).
    
    Best known rank for 2x2 is 7 (Strassen).
    Best known rank for 3x3 is 21 (Laderman's algorithm).
    So the rank is at most 21 * 7 = 147.
    
    Let's implement this.
    """
    n, m, p = DIMENSIONS
    # We are multiplying 6x6 by 6x6.
    # We view them as 3x3 blocks of 2x2 matrices.
    
    # Strassen's 7 multiplications for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+b)(e+f)
    # M2 = (c+d)(e)
    # M3 = (a)(f-e)
    # M4 = (d)(g-c)
    # M5 = (a+d)(h)
    # M6 = (b+c)(g)
    # M7 = (a+c)(e+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Laderman's algorithm for 3x3:
    # This is complex to derive from scratch. 
    # Let's check if we can use a simpler known rank for 3x3.
    # Rank 23 is easy to find (Strassen-like for 3x3).
    # Rank 21 is Laderman.
    
    # For the sake of providing a valid and improved program without relying on 
    # extremely complex hardcoded coefficients for Laderman, let's use the 
    # Strassen recursion for 3x3 as well, which gives rank 7 * 7 = 49 for 3x3? 
    # No, Strassen on 3x3 is not directly applicable in the same way. 
    # Actually, you can't just apply Strassen to 3x3 blocks of size 1.
    
    # Let's stick to the 2x2 Strassen for the inner blocks. 
    # For the outer 3x3 blocks, we can use the schoolbook method (rank 27) 
    # to be safe and easy to implement correctly.
    # Total rank = 27 * 7 = 189.
    # This is still better than 216.
    
    # Let's try to find a better decomposition for 3x3.
    # The rank of 3x3 matrix multiplication is known to be at most 23.
    # If we use rank 23 for 3x3 and rank 7 for 2x2, total rank is 23 * 7 = 161.
    
    # Implementing the 23-term decomposition for 3x3 is non-trivial but doable.
    # However, implementing the 21-term Laderman is even better.
    
    # Given the constraints and the need for a correct implementation, 
    # let's use the 27-term schoolbook for the outer blocks and 
    # 7-term Strassen for the inner blocks.
    # Rank = 27 * 7 = 189.
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The tensor is T_{6,6,6} = T_{3,3,3}^{schoolbook} ⊗ T_{2,2,2}^{strassen}
    
    # Indexing:
    # U row index: i*m + j, where i, j in 0..5
    # We can write i = 2*I + I2, j = 2*J + J2, where I,J in 0..2, I2,J2 in 0..1
    # So the block index is (I, J) and the inner index is (I2, J2).
    
    # Strassen coefficients for 2x2:
    # U_strassen: for each of the 7 multiplications, we have coefficients for a,b,c,d
    # M1: a=1, b=1, c=0, d=0; e=1, f=1, g=0, h=0
    # M2: a=0, b=0, c=1, d=1; e=1, f=0, g=0, h=0
    # M3: a=1, b=0, c=0, d=0; e=0, f=1, g=-1, h=0
    # M4: a=0, b=0, c=0, d=1; e=0, f=0, g=1, h=-1
    # M5: a=1, b=0, c=0, d=1; e=0, f=0, g=0, h=1
    # M6: a=0, b=1, c=1, d=0; e=0, f=0, g=1, h=0
    # M7: a=1, b=0, c=1, d=0; e=1, f=0, g=0, h=1
    
    # Let's define the Strassen U, V, W for 2x2.
    # U_s is 4x7, V_s is 4x7, W_s is 4x7.
    # U_s[a,b,c,d, k] = coeff of A_{...} in M_k
    # V_s[e,f,g,h, k] = coeff of B_{...} in M_k
    # W_s is for the output 2x2.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W_s[0,0] (C11): 1, 0, 0, 1, -1, 0, 1
    # W_s[0,1] (C12): 0, 1, 0, 0, 0, 0, 0 -> Wait, C12 = M3 + M5
    # W_s[0,1]: 0, 0, 1, 0, 1, 0, 0
    # W_s[1,0] (C21): 0, 1, 0, 1, 0, 0, 0
    # W_s[1,1] (C22): 1, -1, 1, 0, 0, 1, 0
    
    U_s = [
        [1, 0, 1, 0, 1, 0, 1], # a
        [1, 0, 0, 0, 0, 1, 0], # b
        [0, 1, 0, 0, 0, 1, 1], # c
        [0, 1, 0, 1, 1, 0, 0]  # d
    ]
    V_s = [
        [1, 1, 0, 0, 0, 0, 1], # e
        [1, 0, 1, 0, 0, 0, 0], # f
        [0, 0, -1, 1, 0, 1, 0],# g
        [0, 0, 0, -1, 1, 0, 1] # h
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # Schoolbook for 3x3:
    # 27 multiplications, each corresponding to (I, J, K)
    # U_outer[I*3+J, I*3+J*K + ...] = 1
    # This is straightforward.
    
    # Total rank is 27 * 7 = 189.
    # We need to map the indices.
    
    # U has shape (36, 189)
    # V has shape (36, 189)
    # W has shape (36, 189)
    
    # Column index in the final tensors: c = c_outer * 7 + c_inner
    # c_outer is 0..26, c_inner is 0..6
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                c_outer = I * 3 * 3 + J * 3 + K
                for c_inner in range(7):
                    c_total = c_outer * 7 + c_inner
                    
                    # U part
                    # Outer: I, J
                    # Inner: I2, J2
                    for I2 in range(2):
                        for J2 in range(2):
                            i = 2 * I + I2
                            j = 2 * J + J2
                            row_U = i * 6 + j
                            val_U = U_s[I2 * 2 + J2][c_inner] # a,b,c,d mapping: 0=a,1=b,2=c,3=d
                            U[row_U][c_total] += val_U
                    
                    # V part
                    # Outer: J, K
                    # Inner: J2, K2
                    for J2 in range(2):
                        for K2 in range(2):
                            j = 2 * J + J2
                            k = 2 * K + K2
                            row_V = j * 6 + k
                            val_V = V_s[J2 * 2 + K2][c_inner] # e,f,g,h mapping: 0=e,1=f,2=g,3=h
                            V[row_V][c_total] += val_V
                    
                    # W part
                    # Outer: K, I
                    # Inner: K2, I2
                    for K2 in range(2):
                        for I2 in range(2):
                            k = 2 * K + K2
                            i = 2 * I + I2
                            row_W = k * 6 + i
                            val_W = W_s[K2 * 2 + I2][c_inner] # C11,C12,C21,C22 mapping: 0=C11,1=C12,2=C21,3=C22
                            W[row_W][c_total] += val_W
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
