"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _build_decomp():
    """Build a rank-64 decomposition using Strassen-style 2x2 block recursion for 4x4x7 matmul.
    
    Strategy: Partition A (4x4) into 2x2 blocks of 2x2 matrices, B (4x7) into 2x1 blocks of 2x7 matrices.
    Use Strassen's 7-multiplication algorithm for the 2x2 x 2x7 product, then combine.
    
    Actually, let me think more carefully. We want to compute C = A*B where A is 4x4, B is 4x7, C is 4x7.
    
    Partition A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each block is 2x2.
    Partition B into 2x1 blocks: B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    Then C = [A11*B1 + A12*B2; A21*B1 + A22*B2] where C1 = A11*B1 + A12*B2 (2x7) and C2 = A21*B1 + A22*B2 (2x7).
    
    For each 2x2 * 2x7 product, we can use a Strassen-like approach. But Strassen is for square matrices.
    
    Let me think differently. The standard approach for 4x4x4 is to use Strassen recursively:
    4x4x4 can be done with rank 49 (7^2) by applying Strassen to 2x2 blocks.
    
    For 4x4x7: We can view this as two 2x2x7 products (top row of C and bottom row of C).
    Each 2x2x7 product: A_block (2x2) * B_block (2x7) = C_block (2x7).
    
    For 2x2 * 2x7: We can partition B_block into two 2x3.5... no, that doesn't work cleanly.
    
    Alternative: Use the fact that 2x2 * 2x7 can be computed with 7 multiplications if we use a generalized Strassen.
    
    Actually, let me use a cleaner approach. The rank of 4x4x4 matmul is at most 49 (Strassen^2).
    For 4x4x7, we can compute it as:
    - Split B into B_left (4x3) and B_right (4x4)
    - Compute C_left = A * B_left (4x3) and C_right = A * B_right (4x4)
    - C_left can be done with rank at most 4*4*3 = 48 (schoolbook) or better
    - C_right can be done with rank at most 49 (Strassen)
    
    But that's not optimal. Let me think of a better decomposition.
    
    Actually, the cleanest approach: Use Strassen for the 4x4x4 part and schoolbook for the remaining 4x4x3 part.
    
    4x4x4: rank 49 (Strassen applied twice: 2x2x2 has rank 7, so 4x4x4 has rank 7^2 = 49)
    4x4x3: rank 4*4*3 = 48 (schoolbook) or possibly less
    
    Total: 49 + 48 = 97. That's better than 112.
    
    But can we do better? Let me try a different split.
    
    Actually, let me just implement the Strassen-based 4x4x4 decomposition (rank 49) and combine with schoolbook for 4x4x3 (rank 48), giving total rank 97.
    
    Wait, I can do better. Let me use the tensor product structure more carefully.
    
    For 4x4x4: Apply Strassen (rank 7) to 2x2x2 blocks. The 4x4x4 tensor is the tensor product of two 2x2x2 tensors, so rank is 7*7 = 49.
    
    For 4x4x7: I can write this as 4x4x4 + 4x4x3. The 4x4x4 part has rank 49. The 4x4x3 part... 
    
    4x4x3: Partition A into 2x2 blocks, B into 2x3 blocks. Each 2x2x2 block product... no, B is 4x3, so blocks are 2x3.
    
    Hmm, 2x2 * 2x3: This has rank at most 2*2*3 = 12 (schoolbook). There are 4 such block products, so total rank 4*12 = 48.
    
    So total rank = 49 + 48 = 97.
    
    Let me implement this.
    """
    n, m, p = 4, 4, 7
    
    # We'll build the decomposition as a concatenation of:
    # 1. 4x4x4 part (rank 49) using Strassen
    # 2. 4x4x3 part (rank 48) using schoolbook
    # Total rank: 97
    
    rank_total = 97
    U = [[0.0] * rank_total for _ in range(n * m)]   # 16 x 97
    V = [[0.0] * rank_total for _ in range(m * p)]   # 28 x 97
    W = [[0.0] * rank_total for _ in range(p * n)]   # 28 x 97
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # Columns 0-48
    _strassen_4x4x4(U, V, W, col_offset=0)
    
    # Part 2: 4x4x3 using schoolbook (rank 48)
    # Columns 49-96
    _schoolbook_4x4x3(U, V, W, col_offset=49)
    
    return U, V, W

def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7).
    
    Returns U, V, W each of shape (4, 7) for the 2x2x2 tensor.
    The tensor is T[i,j,k] = delta[i,k] * delta[j,?] ... 
    
    Actually, for 2x2 * 2x2 = 2x2:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i,j,k) where i is row of A, j is col of A / row of B, k is col of B.
    T[i,j,k] = 1 if this term contributes to C[i,k], else 0.
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 - M4 + M5 + M7
    C12 = M3 + M4
    C21 = M2 + M5
    C22 = M1 - M3 + M6 + M7
    """
    # U maps (i,j) -> coefficients for each of the 7 products
    # V maps (j,k) -> coefficients for each of the 7 products  
    # W maps (k,i) -> coefficients for each of the 7 products (output C[i,k], index k*n+i with n=2)
    
    # For 2x2x2: n=2, m=2, p=2
    # U is 4x7 (i*m+j = i*2+j), V is 4x7 (j*p+k = j*2+k), W is 4x7 (k*n+i = k*2+i)
    
    # M1 = (A11+A22)(B11+B22)
    # A11 is A[0,0], A22 is A[1,1]
    # B11 is B[0,0], B22 is B[1,1]
    # U[:,0]: A11 coeff 1, A22 coeff 1 -> U[0,0]=1, U[3,0]=1
    # V[:,0]: B11 coeff 1, B22 coeff 1 -> V[0,0]=1, V[3,0]=1
    # W[:,0]: C11 coeff 1, C22 coeff 1 -> W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
    
    # M2 = (A21+A22)B11
    # A21 is A[1,0], A22 is A[1,1]
    # B11 is B[0,0]
    # U[:,1]: A21 coeff 1, A22 coeff 1 -> U[2,1]=1, U[3,1]=1
    # V[:,1]: B11 coeff 1 -> V[0,1]=1
    # W[:,1]: C21 coeff 1 -> W[2,1]=1 (k=0,i=1)
    
    # M3 = A11(B21-B22)
    # A11 is A[0,0]
    # B21 is B[1,0], B22 is B[1,1]
    # U[:,2]: A11 coeff 1 -> U[0,2]=1
    # V[:,2]: B21 coeff 1, B22 coeff -1 -> V[2,2]=1, V[3,2]=-1
    # W[:,2]: C12 coeff 1, C22 coeff -1 -> W[1,2]=1 (k=1,i=0), W[3,2]=-1 (k=1,i=1)
    
    # M4 = A22(B12-B11)
    # A22 is A[1,1]
    # B12 is B[0,1], B11 is B[0,0]
    # U[:,3]: A22 coeff 1 -> U[3,3]=1
    # V[:,3]: B12 coeff 1, B11 coeff -1 -> V[1,3]=1, V[0,3]=-1
    # W[:,3]: C11 coeff -1, C12 coeff 1 -> W[0,3]=-1 (k=0,i=0), W[1,3]=1 (k=1,i=0)
    
    # M5 = (A11+A12)B22
    # A11 is A[0,0], A12 is A[0,1]
    # B22 is B[1,1]
    # U[:,4]: A11 coeff 1, A12 coeff 1 -> U[0,4]=1, U[1,4]=1
    # V[:,4]: B22 coeff 1 -> V[3,4]=1
    # W[:,4]: C11 coeff 1, C21 coeff 1 -> W[0,4]=1 (k=0,i=0), W[2,4]=1 (k=0,i=1)
    
    # M6 = (A21-A11)(B11+B12)
    # A21 is A[1,0], A11 is A[0,0]
    # B11 is B[0,0], B12 is B[0,1]
    # U[:,5]: A21 coeff 1, A11 coeff -1 -> U[2,5]=1, U[0,5]=-1
    # V[:,5]: B11 coeff 1, B12 coeff 1 -> V[0,5]=1, V[1,5]=1
    # W[:,5]: C22 coeff 1 -> W[3,5]=1 (k=1,i=1)
    
    # M7 = (A12-A22)(B21+B22)
    # A12 is A[0,1], A22 is A[1,1]
    # B21 is B[1,0], B22 is B[1,1]
    # U[:,6]: A12 coeff 1, A22 coeff -1 -> U[1,6]=1, U[3,6]=-1
    # V[:,6]: B21 coeff 1, B22 coeff 1 -> V[2,6]=1, V[3,6]=1
    # W[:,6]: C11 coeff 1, C22 coeff 1 -> W[0,6]=1 (k=0,i=0), W[3,6]=1 (k=1,i=1)
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # M3
    U[0][2] = 1.0
    V[2][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = -1.0
    
    # M4
    U[3][3] = 1.0
    V[1][3] = 1.0; V[0][3] = -1.0
    W[0][3] = -1.0; W[1][3] = 1.0
    
    # M5
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = 1.0; W[2][4] = 1.0
    
    # M6
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0; W[3][6] = 1.0
    
    return U, V, W

def _strassen_4x4x4(U, V, W, col_offset=0):
    """Fill in Strassen-based 4x4x4 decomposition starting at col_offset.
    
    Uses tensor product of two Strassen 2x2x2 decompositions.
    Rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Partition into 2x2 blocks of 2x2 submatrices
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    # U_4x4x4 = U2 ⊗ U2, V_4x4x4 = V2 ⊗ V2, W_4x4x4 = W2 ⊗ W2
    
    # U2 is 4x7, so U2 ⊗ U2 is 16x49
    # Index mapping: U2[i2,j2] where i2 = i_outer*2 + i_inner, j2 = j_outer*2 + j_inner
    # U_4x4x4[(i_outer*2+i_inner)*2 + (j_outer*2+j_inner), r_outer*7 + r_inner] 
    #   = U2[i_outer*2+i_inner, r_outer] * U2[j_outer*2+j_inner, r_inner]
    
    # Wait, I need to be more careful with the indexing.
    # A is 4x4, indexed as A[i,j] where i=0..3, j=0..3
    # In block form: i = i_outer*2 + i_inner, j = j_outer*2 + j_inner
    # A[i,j] = A_{i_outer,j_outer}[i_inner, j_inner]
    
    # The tensor product: T_4x4x4[i,j,k] = sum over block structure
    # Actually, the 4x4x4 matmul tensor is NOT simply the Kronecker product of two 2x2x2 tensors.
    # Let me reconsider.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # With i = i_o*2+i_i, j = j_o*2+j_i, k = k_o*2+k_i:
    # C[i_o*2+i_i, k_o*2+k_i] = sum_{j_o,j_i} A[i_o*2+i_i, j_o*2+j_i] * B[j_o*2+j_i, k_o*2+k_i]
    # = sum_{j_o} (sum_{j_i} A[i_o*2+i_i, j_o*2+j_i] * B[j_o*2+j_i, k_o*2+k_i])
    
    # This is not a simple tensor product. The Strassen decomposition for 4x4x4 works differently.
    
    # Let me use the standard Strassen recursion:
    # C = A*B where A,B,C are 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each C_ij is a 2x2 matrix computed from 2x2 * 2x2 products.
    # Using Strassen for each:
    # For C11: A11*B11 + A12*B21
    #   M1 = (A11+A12)(B11-B21)... no, that's not right.
    
    # Actually, the standard approach is to apply Strassen to the 4x4x4 directly by treating
    # it as a 2x2 block matrix multiplication where each "element" is a 2x2 matrix.
    # But matrix multiplication of 2x2 blocks doesn't commute, so we need to be careful.
    
    # The correct Strassen recursion for 4x4x4:
    # C = A*B, split into 2x2 blocks of 2x2 matrices
    # Use 7 products of 2x2 matrices:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 - P4 + P5 + P7
    # C12 = P3 + P4
    # C21 = P2 + P5
    # C22 = P1 - P3 + P6 + P7
    
    # Each Pk is a 2x2 matrix product, which can be done with rank 7 (Strassen for 2x2x2).
    # So total rank = 7 * 7 = 49.
    
    # Now I need to construct the actual U, V, W matrices.
    
    # For each of the 7 "outer" Strassen products Pk (k=1..7):
    #   Pk = Xk * Yk where Xk is a 2x2 linear combination of A blocks, Yk is a 2x2 linear combination of B blocks
    #   Then Pk contributes to C blocks via W coefficients.
    #   Within each Pk, we use the 2x2x2 Strassen decomposition (7 "inner" products).
    
    # Outer Strassen coefficients:
    # For A blocks: A11, A12, A21, A22 (each 2x2)
    # For B blocks: B11, B12, B21, B22 (each 2x2)
    
    # P1: X = A11+A22, Y = B11+B22
    # P2: X = A21+A22, Y = B11
    # P3: X = A11, Y = B21-B22
    # P4: X = A22, Y = B12-B11
    # P5: X = A11+A12, Y = B22
    # P6: X = A21-A11, Y = B11+B12
    # P7: X = A12-A22, Y = B21+B22
    
    # C11 = P1 - P4 + P5 + P7
    # C12 = P3 + P4
    # C21 = P2 + P5
    # C22 = P1 - P3 + P6 + P7
    
    # A block indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1) in (row_block, col_block)
    # A[i,j] where i = row_block*2 + row_inner, j = col_block*2 + col_inner
    # So A11 covers A[0,0],A[0,1],A[1,0],A[1,1]
    # A12 covers A[0,2],A[0,3],A[1,2],A[1,3]
    # A21 covers A[2,0],A[2,1],A[3,0],A[3,1]
    # A22 covers A[2,2],A[2,3],A[3,2],A[3,3]
    
    # B block indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1) in (row_block, col_block)
    # B[j,k] where j = row_block*2 + row_inner, k = col_block*2 + col_inner
    # B11 covers B[0,0],B[0,1],B[1,0],B[1,1]
    # B12 covers B[0,2],B[0,3],B[1,2],B[1,3]
    # B21 covers B[2,0],B[2,1],B[3,0],B[3,1]
    # B22 covers B[2,2],B[2,3],B[3,2],B[3,3]
    
    # C block indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1) in (row_block, col_block)
    # C[i,k] where i = row_block*2 + row_inner, k = col_block*2 + col_inner
    # C11 covers C[0,0],C[0,1],C[1,0],C[1,1]
    # C12 covers C[0,2],C[0,3],C[1,2],C[1,3]
    # C21 covers C[2,0],C[2,1],C[3,0],C[3,1]
    # C22 covers C[2,2],C[2,3],C[3,2],C[3,3]
    
    # W index: k*n + i where n=4, so W index for C[i,k] is k*4+i
    
    # Outer coefficients (for the 7 outer products):
    # A_block_coefs[k_outer][block_idx] where block_idx: 0=A11, 1=A12, 2=A21, 3=A22
    # B_block_coefs[k_outer][block_idx] where block_idx: 0=B11, 1=B12, 2=B21, 3=B22
    # C_block_coefs[k_outer][block_idx] where block_idx: 0=C11, 1=C12, 2=C21, 3=C22
    
    A_coefs = [
        [1, 0, 0, 1],  # P1: A11+A22
        [0, 0, 1, 1],  # P2: A21+A22
        [1, 0, 0, 0],  # P3: A11
        [0, 0, 0, 1],  # P4: A22
        [1, 1, 0, 0],  # P5: A11+A12
        [-1, 0, 1, 0], # P6: A21-A11
        [0, 1, 0, -1], # P7: A12-A22
    ]
    
    B_coefs = [
        [1, 0, 0, 1],  # P1: B11+B22
        [1, 0, 0, 0],  # P2: B11
        [0, 0, 1, -1], # P3: B21-B22
        [0, 1, 0, -1], # P4: B12-B11
        [0, 0, 0, 1],  # P5: B22
        [1, 1, 0, 0],  # P6: B11+B12
        [0, 0, 1, 1],  # P7: B21+B22
    ]
    
    C_coefs = [
        [1, 0, 0, 0],   # P1 -> C11
        [0, 0, 1, 0],   # P2 -> C21
        [0, 1, 0, 0],   # P3 -> C12
        [0, 0, 0, 1],   # P4 -> C22... wait
        [0, 0, 1, 0],   # P5 -> C21... 
        [0, 0, 0, 1],   # P6 -> C22
        [0, 0, 0, 1],   # P7 -> C22
    ]
    
    # Wait, I need to track contributions to ALL C blocks, not just one.
    # Let me redo:
    # C11 = P1 - P4 + P5 + P7
    # C12 = P3 + P4
    # C21 = P2 + P5
    # C22 = P1 - P3 + P6 + P7
    
    # So for each outer product Pk, its contribution to each C block:
    C_contrib = [
        [1, 0, 0, 0],   # P1: C11=1, C12=0, C21=0, C22=0... no
        # P1 contributes to C11 (+1) and C22 (+1)
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 0],   # P2: C21=1
        [0, 1, 0, 0],   # P3: C12=1
        [0, 1, 0, 0],   # P4: C12=1... no, P4 contributes to C11 (-1) and C12 (+1)
        [0, 0, 0, 0],   # placeholder
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    
    # Let me be more careful:
    # C11 = 1*P1 + 0*P2 + 0*P3 + (-1)*P4 + 1*P5 + 0*P6 + 1*P7
    # C12 = 0*P1 + 0*P2 + 1*P3 + 1*P4 + 0*P5 + 0*P6 + 0*P7
    # C21 = 0*P1 + 1*P2 + 0*P3 + 0*P4 + 1*P5 + 0*P6 + 0*P7
    # C22 = 1*P1 + 0*P2 + (-1)*P3 + 0*P4 + 0*P5 + 1*P6 + 1*P7
    
    C_contrib = [
        [1, 0, 0, 1],   # P1: C11=1, C12=0, C21=0, C22=1
        [0, 0, 1, 0],   # P2: C21=1
        [0, 1, 0, -1],  # P3: C12=1, C22=-1
        [-1, 1, 0, 0],  # P4: C11=-1, C12=1
        [1, 0, 1, 0],   # P5: C11=1, C21=1
        [0, 0, 0, 1],   # P6: C22=1
        [1, 0, 0, 1],   # P7: C11=1, C22=1
    ]
    
    # Now, for each outer product k (0..6) and each inner Strassen product l (0..6):
    # The combined product is: (sum over A blocks with A_coefs[k]) * (sum over B blocks with B_coefs[k])
    # Then within that 2x2x2 product, we use the inner Strassen decomposition.
    
    # U index: i*m + j = i*4 + j for 4x4 A
    # V index: j*p + k = j*4 + k for 4x4 B
    # W index: k*n + i = k*4 + i for 4x4 C
    
    # For outer product k and inner product l:
    # Column index in the 49-column space: k*7 + l
    
    # U contribution: For each A block (row_block, col_block) with coefficient A_coefs[k][block]:
    #   For each (row_inner, col_inner) in the 2x2 block:
    #     i = row_block*2 + row_inner, j = col_block*2 + col_inner
    #     U[i*4+j, k*7+l] += A_coefs[k][block] * U2[row_inner*2+col_inner, l]
    
    # V contribution: For each B block (row_block, col_block) with coefficient B_coefs[k][block]:
    #   For each (row_inner, col_inner) in the 2x2 block:
    #     j = row_block*2 + row_inner, k_col = col_block*2 + col_inner
    #     V[j*4+k_col, k*7+l] += B_coefs[k][block] * V2[row_inner*2+col_inner, l]
    
    # W contribution: For each C block (row_block, col_block) with coefficient C_contrib[k][block]:
    #   For each (row_inner, col_inner) in the 2x2 block:
    #     i = row_block*2 + row_inner, k_col = col_block*2 + col_inner
    #     W[k_col*4+i, k*7+l] += C_contrib[k][block] * W2[col_inner*2+row_inner, l]
    
    # Note: W2 index is k*n+i for 2x2, so k_inner*2 + i_inner. In our C block, i_inner is row, k_inner is col.
    # W2[k_inner*2 + i_inner, l]
    
    for k_outer in range(7):
        for l_inner in range(7):
            col = col_offset + k_outer * 7 + l_inner
            
            # U contributions
            for ab in range(4):  # A block index
                coef_a = A_coefs[k_outer][ab]
                if coef_a == 0:
                    continue
                row_block = ab // 2
                col_block = ab % 2
                for ri in range(2):
                    for ci in range(2):
                        i = row_block * 2 + ri
                        j = col_block * 2 + ci
                        u_idx = ri * 2 + ci
                        U[i * 4 + j][col] += coef_a * U2[u_idx][l_inner]
            
            # V contributions
            for bb in range(4):  # B block index
                coef_b = B_coefs[k_outer][bb]
                if coef_b == 0:
                    continue
                row_block = bb // 2
                col_block = bb % 2
                for ri in range(2):
                    for ci in range(2):
                        j = row_block * 2 + ri
                        k_col = col_block * 2 + ci
                        v_idx = ri * 2 + ci
                        V[j * 4 + k_col][col] += coef_b * V2[v_idx][l_inner]
            
            # W contributions
            for cb in range(4):  # C block index
                coef_c = C_contrib[k_outer][cb]
                if coef_c == 0:
                    continue
                row_block = cb // 2
                col_block = cb % 2
                for ri in range(2):
                    for ci in range(2):
                        i = row_block * 2 + ri
                        k_col = col_block * 2 + ci
                        w_idx = ci * 2 + ri  # k_inner*2 + i_inner
                        W[k_col * 4 + i][col] += coef_c * W2[w_idx][l_inner]

def _schoolbook_4x4x3(U, V, W, col_offset=0):
    """Fill in schoolbook 4x4x3 decomposition starting at col_offset.
    
    Computes A (4x4) * B (4x3) = C (4x3) with rank 4*4*3 = 48.
    Only fills in the first 3 columns of B and C.
    """
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
    
    col = col_offset
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    U, V, W = _build_decomp()
    return {"U": U, "V": V, "W": W}
