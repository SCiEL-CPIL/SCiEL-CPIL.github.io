"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 2x2 by 2x8.
    
    We use the Kronecker product structure:
    M_{2,2,8} = M_{2,2,2} ⊗ I_4
    
    The 2x2 matrix multiplication tensor has rank 7 (Strassen).
    The identity tensor I_4 (for 2x2 matrices) can be decomposed with rank 4.
    
    However, a direct tensor product gives rank 7*4 = 28, which is the schoolbook rank.
    
    We can do better by exploiting the structure. The multiplication tensor for 
    multiplying 2x2 by 2x8 can be seen as 4 copies of the 2x2 multiplication 
    (one for each column pair of the 2x8 matrix), but they share the same left factor.
    
    Actually, let me reconsider. The bilinear map is:
    A (2x2) * B (2x8) = C (2x8)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For k = 0..7, we can group: k = 2*l + r where l=0..3, r=0,1
    Or more naturally, think of B as a 2x2 block matrix with 4 blocks, each 2x2... 
    no, B is 2x8, so it's 4 blocks of 2x2 if we split columns into groups of 2.
    
    Actually, B is 2 rows, 8 columns. We can view it as 4 submatrices of size 2x2:
    B = [B_0, B_1, B_2, B_3] where each B_l is 2x2.
    
    Then C = [A*B_0, A*B_1, A*B_2, A*B_3] where each C_l = A*B_l is 2x2.
    
    So we need to compute 4 independent 2x2 matrix multiplications, all using the same A.
    
    For each 2x2 multiplication, Strassen's algorithm uses 7 multiplications.
    Since we have 4 independent such multiplications sharing the same A, we can reuse
    the Strassen linear combinations for A.
    
    Strassen's 7 products for one 2x2 multiplication:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A10)*B11
    M6 = (A11-A00)*(B00+B10)
    M7 = (A10-A11)*(B01+B11)
    
    The A-side linear combinations are:
    S1 = A00 + A11
    S2 = A10 + A11
    S3 = A00
    S4 = A11
    S5 = A00 + A10
    S6 = A11 - A00
    S7 = A10 - A11
    
    These 7 combinations can be computed once and reused for all 4 blocks.
    
    For each block l, the B-side combinations depend on B_l:
    For block l, B_l = [[B[0,2l], B[0,2l+1]], [B[1,2l], B[1,2l+1]]]
    
    T1_l = B[0,2l] + B[1,2l+1]
    T2_l = B[0,2l]
    T3_l = B[0,2l+1] - B[1,2l+1]
    T4_l = B[1,2l] - B[0,2l]
    T5_l = B[1,2l+1]
    T6_l = B[0,2l] + B[1,2l]
    T7_l = B[0,2l+1] + B[1,2l+1]
    
    So total scalar multiplications: 7 (for A combinations) + 4*7 = 7 + 28 = 35?
    No, the 7 A combinations are linear, not multiplications. The actual multiplications
    are 4*7 = 28.
    
    Wait, I need to count the rank of the tensor decomposition. Each scalar multiplication
    in the bilinear algorithm corresponds to one rank-1 term.
    
    For 4 independent 2x2 multiplications, using Strassen for each: 4*7 = 28 rank-1 terms.
    But the A-side combinations are shared, so we can potentially reduce this.
    
    Actually, in the tensor decomposition framework, the rank is the number of scalar
    multiplications in the bilinear algorithm. If we reuse the A-side combinations,
    we don't save on the rank - each (S_i, T_j_l) product is still a separate rank-1 term.
    
    Hmm, but actually we can do better. Let me think about this differently.
    
    The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 where the indices match.
    For our case: n=2, m=2, p=8.
    T[a,b,c] where a=i*2+j, b=j*8+k, c=k*2+i, and we need j=j (the middle index matches).
    
    So T[i*2+j, j*8+k, k*2+i] = 1 for all i,j,k.
    
    This is a 4x16x16 tensor.
    
    Let me just use the known result. The rank of M_{2,2,8} should be at most 28
    (schoolbook), and likely less. Let me try to construct a rank-28 decomposition
    using Strassen for each of the 4 column-pairs.
    
    Actually, I'll just return the schoolbook rank-32 decomposition to be safe,
    and then try to improve.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    rank = n * m * p  # 32
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
