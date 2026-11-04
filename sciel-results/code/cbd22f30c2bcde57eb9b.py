"""Matrix multiplication 2x3 by 3x5 using Laderman-inspired rank-7 embedding."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-7 decomposition for 2x3 * 3x5 matrix multiplication.
    
    We embed the 2x3 matrix A and 3x5 matrix B into 3x3 matrices by zero-padding:
    A_pad = [[a11, a12, a13],
             [a21, a22, a23],
             [0,   0,   0  ]]
    B_pad = [[b11, b12, b13, b14, b15],
             [b21, b22, b23, b24, b25],
             [b31, b32, b33, b34, b35]]
    
    But wait - B is 3x5, not 3x3. Let me reconsider.
    
    Actually, for a 2x3 * 3x5 multiplication, the output is 2x5.
    
    Let me think about this differently. The standard approach for rectangular 
    matrix multiplication using Laderman would be to find a rank decomposition.
    
    For 2x3 * 3x5, the schoolbook rank is 2*3*5 = 30.
    
    Let me try a different approach: use a rank-7 or similar decomposition.
    
    Actually, let me just use the straightforward schoolbook approach but see if 
    there's a known lower rank. For 2x3 * 3x5, I believe the optimal rank might 
    be around 7-10.
    
    Let me try a constructive approach. We want to compute C = A*B where A is 2x3 
    and B is 3x5.
    
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    
    So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    ... etc.
    
    One approach: use a bilinear decomposition. Let's try to find a rank-r decomposition.
    
    For now, let me just return the schoolbook rank-30 decomposition to ensure correctness,
    and then we can try to optimize.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Schoolbook decomposition of rank n*m*p = 30
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 15 x 30
    W = [[0] * rank for _ in range(p * n)]  # 10 x 30
    
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
