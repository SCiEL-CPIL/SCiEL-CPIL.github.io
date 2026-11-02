"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    This is the theoretical minimum rank for 2x2 matrix multiplication.
    
    Interface:
    U: shape (n*m, rank) = (4, 7). Row index i*m + j corresponds to A[i, j].
    V: shape (m*p, rank) = (4, 7). Row index j*p + k corresponds to B[j, k].
    W: shape (p*n, rank) = (4, 7). Row index k*n + i corresponds to C[i, k] 
       (Note: The task specifies W[k*n+i, r] for the output factor, which 
       corresponds to C[i, k] if we view the output tensor T[i, j, k] 
       where C[i, k] = sum_j A[i, j]B[j, k]. 
       Wait, standard tensor definition for matrix mult is T[i, j, k] = delta_{i,j}... no.
       The tensor T such that C = A B is T[i, j, k] where C[i, k] = sum_j A[i, j] T[j, k]? No.
       C[i, k] = sum_j A[i, j] B[j, k].
       So the tensor is T[i, j, k] = 1 if the indices align for the product?
       Actually, the decomposition is:
       C[i, k] = sum_{r=1}^R U[i, r] V[k, r]? No, that's not right either.
       
       Let's stick to the standard Strassen formulation provided in the current code which is known to be correct.
       
       U[i*m + j, r] is the coefficient of A[i, j] in the r-th intermediate product.
       V[j*p + k, r] is the coefficient of B[j, k] in the r-th intermediate product.
       W[k*n + i, r] is the coefficient of the r-th intermediate product in C[i, k].
       
       Let's verify the indices for W.
       The task says: "output factor follows the official trace convention W[k*n+i,r]".
       Usually, C is indexed C[i, k].
       If W is a matrix of shape (p*n, rank), and row index is k*n + i, 
       then W[k*n + i, r] contributes to C[i, k].
       This matches the standard convention where the output tensor is flattened 
       in the order of the second index of B (k) then the first index of A (i)?
       Or is it C[i, k] flattened as i*n + k?
       The prompt says W[k*n+i, r]. 
       If n=2, k=0, i=0 -> index 0. C[0,0].
       k=0, i=1 -> index 1. C[1,0].
       k=1, i=0 -> index 2. C[0,1].
       k=1, i=1 -> index 3. C[1,1].
       
       So W[row, r] adds to C[i, k] where row = k*n + i.
       
       Let's re-verify the Strassen terms with this W indexing.
       
       m0 = (a00 + a11)(b00 + b11)
       U[0,0]=1, U[3,0]=1 (a00, a11)
       V[0,0]=1, V[3,0]=1 (b00, b11)
       Result m0 contributes to C[0,0] and C[1,1].
       C[0,0]: i=0, k=0 -> W[0*2+0, 0] = W[0,0] should be 1.
       C[1,1]: i=1, k=1 -> W[1*2+1, 0] = W[3,0] should be 1.
       
       Current code:
       W[0][0] = 1
       W[3][0] = 1
       This matches.
       
       m1 = (a10 + a11)b00
       U[2,1]=1, U[3,1]=1 (a10, a11)
       V[0,1]=1 (b00)
       Result m1 contributes to C[1,0] and C[1,1].
       C[1,0]: i=1, k=0 -> W[0*2+1, 1] = W[1,1] should be 1.
       C[1,1]: i=1, k=1 -> W[1*2+1, 1] = W[3,1] should be -1.
       
       Current code:
       W[1][1] = 1
       W[3][1] = -1
       This matches.
       
       m2 = a00(b01 - b11)
       U[0,2]=1 (a00)
       V[1,2]=1, V[3,2]=-1 (b01, b11)
       Result m2 contributes to C[0,1] and C[1,1].
       C[0,1]: i=0, k=1 -> W[1*2+0, 2] = W[2,2] should be 1.
       C[1,1]: i=1, k=1 -> W[1*2+1, 2] = W[3,2] should be 1.
       
       Current code:
       W[2][2] = 1
       W[3][2] = 1
       This matches.
       
       m3 = a11(b10 - b00)
       U[3,3]=1 (a11)
       V[2,3]=1, V[0,3]=-1 (b10, b00)
       Result m3 contributes to C[0,0] and C[1,0].
       C[0,0]: i=0, k=0 -> W[0,3] should be 1.
       C[1,0]: i=1, k=0 -> W[1,3] should be 1.
       
       Current code:
       W[0][3] = 1
       W[1][3] = 1
       This matches.
       
       m4 = (a00 + a01)b11
       U[0,4]=1, U[1,4]=1 (a00, a01)
       V[3,4]=1 (b11)
       Result m4 contributes to C[0,0] and C[0,1].
       C[0,0]: i=0, k=0 -> W[0,4] should be -1.
       C[0,1]: i=0, k=1 -> W[2,4] should be 1.
       
       Current code:
       W[0][4] = -1
       W[2][4] = 1
       This matches.
       
       m5 = (a10 - a00)(b00 + b01)
       U[2,5]=1, U[0,5]=-1 (a10, a00)
       V[0,5]=1, V[1,5]=1 (b00, b01)
       Result m5 contributes to C[1,1].
       C[1,1]: i=1, k=1 -> W[3,5] should be 1.
       
       Current code:
       W[3][5] = 1
       This matches.
       
       m6 = (a01 - a11)(b10 + b11)
       U[1,6]=1, U[3,6]=-1 (a01, a11)
       V[2,6]=1, V[3,6]=1 (b10, b11)
       Result m6 contributes to C[0,0].
       C[0,0]: i=0, k=0 -> W[0,6] should be 1.
       
       Current code:
       W[0][6] = 1
       This matches.
    """
    n, m, p = DIMENSIONS
    rank = 7

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's seven products:
    #
    # Let A = [[a00, a01],
    #          [a10, a11]]
    #
    #     B = [[b00, b01],
    #          [b10, b11]]
    #
    # C = A * B
    #
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5

    # --- Product 0: m0 = (a00 + a11)(b00 + b11) ---
    U[0][0] = 1   # a00
    U[3][0] = 1   # a11
    V[0][0] = 1   # b00
    V[3][0] = 1   # b11
    W[0][0] = 1   # c00  (k=0, i=0 -> 0*2+0=0)
    W[3][0] = 1   # c11  (k=1, i=1 -> 1*2+1=3)

    # --- Product 1: m1 = (a10 + a11)b00 ---
    U[2][1] = 1   # a10
    U[3][1] = 1   # a11
    V[0][1] = 1   # b00
    W[1][1] = 1   # c10  (k=0, i=1 -> 0*2+1=1)
    W[3][1] = -1  # c11

    # --- Product 2: m2 = a00(b01 - b11) ---
    U[0][2] = 1   # a00
    V[1][2] = 1   # b01
    V[3][2] = -1  # b11
    W[2][2] = 1   # c01  (k=1, i=0 -> 1*2+0=2)
    W[3][2] = 1   # c11

    # --- Product 3: m3 = a11(b10 - b00) ---
    U[3][3] = 1   # a11
    V[2][3] = 1   # b10
    V[0][3] = -1  # b00
    W[0][3] = 1   # c00
    W[1][3] = 1   # c10

    # --- Product 4: m4 = (a00 + a01)b11 ---
    U[0][4] = 1   # a00
    U[1][4] = 1   # a01
    V[3][4] = 1   # b11
    W[0][4] = -1  # c00
    W[2][4] = 1   # c01

    # --- Product 5: m5 = (a10 - a00)(b00 + b01) ---
    U[2][5] = 1   # a10
    U[0][5] = -1  # a00
    V[0][5] = 1   # b00
    V[1][5] = 1   # b01
    W[3][5] = 1   # c11

    # --- Product 6: m6 = (a01 - a11)(b10 + b11) ---
    U[1][6] = 1   # a01
    U[3][6] = -1  # a11
    V[2][6] = 1   # b10
    V[3][6] = 1   # b11
    W[0][6] = 1   # c00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
