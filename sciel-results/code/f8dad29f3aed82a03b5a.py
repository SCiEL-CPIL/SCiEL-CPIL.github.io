"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    Strategy: Use the dissection method exploiting n=2.
    
    Key insight: Split the 3x3 "core" dimension into 2+1 parts.
    Let A be 2x3, B be 3x7, C = A*B be 2x7.
    
    Write A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    Write B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 1x7
    
    Then C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2 * 2x7 = 2x7): use Strassen-like approach
    For A_right * B_bottom (2x1 * 1x7 = 2x7): this is outer product, rank 2
    
    Actually, let me use a cleaner dissection:
    
    Method 1: Direct Strassen extension
    Use the fact that 2x3 * 3x7 can be seen as computing 7 columns of 2x3 * 3x1.
    
    Better approach: Use the tensor product structure.
    
    The 2x3x7 multiplication tensor can be decomposed using:
    1. The 2x2x2 Strassen algorithm (rank 7)
    2. Combined with 1x1x7 and 1x3x1 factors
    
    Here's a concrete construction:
    
    Split m=3 into 2+1. Let:
    A = [[a11, a12, a13], [a21, a22, a23]]
    B = [[b11, b12, ..., b17], [b21, b22, ..., b27], [b31, b32, ..., b37]]
    
    C[i,k] = a_i1*b_1k + a_i2*b_2k + a_i3*b_3k
    
    Using Strassen for the 2x2 part (columns 1,2) and direct for column 3:
    
    For the 2x2 * 2x7 part, use Strassen:
    P1 = a11 * (b11+b12+...+b17)  -- wait, this doesn't work directly.
    
    Let me use a different, more systematic approach.
    
    Approach: Use the known rank-5 algorithm for 2x3x3 and extend to 2x3x7.
    
    For 2x3x3, one rank-5 algorithm is:
    Let A = [[a1,a2,a3],[a4,a5,a6]], B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    
    P1 = a1 * b11
    P2 = a2 * (b21+b22+b23)  
    P3 = (a1+a4) * (b11+b12+b13)
    P4 = a5 * b22
    P5 = (a3+a6) * (b31+b32+b33)
    
    Hmm, this is getting complicated. Let me try a simpler but still improved approach.
    
    Simple improvement: Group the computation by rows of A.
    
    C[0,k] = a00*b0k + a01*b1k + a02*b2k
    C[1,k] = a10*b0k + a11*b1k + a12*b2k
    
    For each k, we need 2 dot products of length 3.
    
    Using the identity: (x+y)(u+v) = xu + xv + yu + yv
    We can compute some terms with shared products.
    
    Actually, here's a clean approach using the fact that n=2:
    
    Define:
    s1 = a00 + a10  (sum of column 0 of A)
    s2 = a01 + a11  (sum of column 1 of A)  
    s3 = a02 + a12  (sum of column 2 of A)
    
    Then:
    C[0,k] + C[1,k] = s1*b0k + s2*b1k + s3*b2k
    
    And:
    C[0,k] - C[1,k] = (a00-a10)*b0k + (a01-a11)*b1k + (a02-a12)*b2k
    
    Let d1 = a00-a10, d2 = a01-a11, d3 = a02-a12
    
    So we need to compute:
    Sum_k = s1*b0k + s2*b1k + s3*b2k  (for each k)
    Diff_k = d1*b0k + d2*b1k + d3*b2k  (for each k)
    
    Then C[0,k] = (Sum_k + Diff_k) / 2
    C[1,k] = (Sum_k - Diff_k) / 2
    
    Now, Sum_k is a dot product of 3-vectors (s1,s2,s3) and (b0k,b1k,b2k)
    Diff_k is a dot product of 3-vectors (d1,d2,d3) and (b0k,b1k,b2k)
    
    For each k, computing a dot product of 3-vectors requires 3 multiplications (schoolbook).
    So total = 7 * 3 (for Sum) + 7 * 3 (for Diff) = 42. No improvement!
    
    But wait - we can share the B terms! The vectors (b0k,b1k,b2k) are the same for both Sum and Diff.
    
    Actually, that doesn't help directly because we still need separate multiplications.
    
    Let me try yet another approach. Use the fact that we can precompute combinations of B.
    
    Define for each k:
    B0k = b0k, B1k = b1k, B2k = b2k
    
    We need:
    C[0,k] = a00*B0k + a01*B1k + a02*B2k
    C[1,k] = a10*B0k + a11*B1k + a12*B2k
    
    This is a 2x3 * 3x1 multiplication for each k, repeated 7 times.
    
    The key is to find a rank-r decomposition of M_{2,3,1} and then "tile" it 7 times.
    
    For M_{2,3,1}: Can we do better than rank 6?
    
    C0 = a00*b0 + a01*b1 + a02*b2
    C1 = a10*b0 + a11*b1 + a12*b2
    
    This is a bilinear map from (a00,a01,a02,a10,a11,a12) x (b0,b1,b2) -> (C0,C1)
    
    The tensor is:
    T = a00*b0*C0 + a01*b1*C0 + a02*b2*C0 + a10*b0*C1 + a11*b1*C1 + a12*b2*C1
    
    This has 6 terms, and I believe the rank is 6 (no known lower rank).
    
    So tiling gives 7*6 = 42. Back to schoolbook.
    
    Let me try the Strassen dissection more carefully.
    
    Split m=3 into 2+1:
    A = [A2x2 | A2x1], B = [B2x7; B1x7]
    
    C = A2x2 * B2x7 + A2x1 * B1x7
    
    For A2x1 * B1x7: This is outer product. (a10,a11)^T * (b20,b21,...,b26)
    C_outer[i,k] = a1i * b2k
    This requires 2*7 = 14 multiplications if done naively, but actually it's rank 2:
    C_outer = a10 * [b20,b21,...,b26] + a11 * [b20,b21,...,b26]
    Wait, that's not right. C_outer[i,k] = a_i * b_k where i=0,1 and k=0..6.
    So C_outer[0,k] = a10 * b2k for all k
    C_outer[1,k] = a11 * b2k for all k
    
    This is: C_outer = (a10, a11)^T ⊗ (b20, b21, ..., b26)
    As a tensor: sum over i,k of a_i * b_k * C[i,k]
    This has rank 2 (since it's a separable tensor in the a and b variables, with 2x7 output).
    
    Actually, the tensor for outer product of 2-vector and 7-vector is:
    T_outer = sum_{i=0}^{1} sum_{k=0}^{6} a_i * b_k * C_{i,k}
    
    This can be written as:
    = a_0 * (sum_k b_k * C_{0,k}) + a_1 * (sum_k b_k * C_{1,k})
    
    But that's not a sum of rank-1 tensors in the standard sense.
    
    In the standard tensor decomposition, each rank-1 term is of the form:
    u_a * v_b * w_c where u is in U-space, v in V-space, w in W-space.
    
    For the outer product part:
    Term (i,k): u has 1 at position i (in 2-dim A-space), v has 1 at position k (in 7-dim B-space), w has 1 at position (i,k) (in 14-dim C-space).
    
    So the outer product part has rank 2*7 = 14.
    
    For the A2x2 * B2x7 part:
    This is 2x2 * 2x7 = 2x7.
    We can use Strassen for 2x2 * 2x1 and tile it 7 times: 7 * 7 = 49. Worse!
    
    Or we can use a different approach for 2x2 * 2x7.
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Using Strassen:
    P1 = A[0,0] * (B[0,k] + B[1,k])  -- but this is per k, so 7 terms
    P2 = A[1,1] * (B[0,k] + B[1,k])  -- 7 terms
    P3 = (A[0,0] + A[1,1]) * B[0,k]  -- 7 terms
    P4 = (A[0,1] + A[1,0]) * B[1,k]  -- 7 terms
    
    Total: 28 terms for the 2x2*2x7 part, plus 14 for the outer product = 42. Still schoolbook!
    
    I'm stuck. Let me just implement a correct schoolbook decomposition and accept rank 42.
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
# EVOLVE-BLOCK-END
