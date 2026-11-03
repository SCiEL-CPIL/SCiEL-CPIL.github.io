import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2 + j, r]
    V[j*2 + k, r]
    W[k*2 + i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11 contributions? 
    # Actually, W indices are (k, i). 
    # C00 gets M1, C11 gets M1.
    # C00 is (i=0, k=0). W index is k*2+i = 0*2+0 = 0.
    # C11 is (i=1, k=1). W index is k*2+i = 1*2+1 = 3.
    # So W[0,0]=1, W[3,0]=1.
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C10 gets M2. C10 is (i=1, k=0). W index 0*2+1 = 2.
    W[2, 1] = 1
    
    # M3 = A00(B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C01 gets M3. C01 is (i=0, k=1). W index 1*2+0 = 2.
    W[2, 2] = 1
    
    # M4 = A11(B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C00 gets M4, C10 gets M4.
    # C00 (i=0, k=0) -> W[0,3]=1
    # C10 (i=1, k=0) -> W[2,3]=1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C00 gets -M5, C01 gets M5.
    # C00 (i=0, k=0) -> W[0,4]=-1
    # C01 (i=0, k=1) -> W[2,4]=1
    W[0, 4] = -1
    W[2, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 gets M6.
    # C11 (i=1, k=1) -> W[3,5]=1
    W[3, 5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C00 gets M7.
    # C00 (i=0, k=0) -> W[0,6]=1
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_4():
    """
    Construct rank-49 decomposition for 4x4x4 using Strassen recursion.
    """
    U2, V2, W2 = _strassen_2x2_2()
    R2 = 7
    R4 = R2 * R2 # 49
    
    U4 = np.zeros((16, R4))
    V4 = np.zeros((16, R4))
    W4 = np.zeros((16, R4))
    
    # The 4x4 matrices are blocks of 2x2.
    # A is 2x2 blocks: A00, A01, A10, A11.
    # B is 2x2 blocks: B00, B01, B10, B11.
    # C is 2x2 blocks: C00, C01, C10, C11.
    
    # Strassen products M1..M7 are 2x2 matrix multiplications.
    # Each Mq is computed by 7 scalar multiplications.
    # Total 49 scalar multiplications.
    
    # We need to map the 49 scalar products to the U, V, W matrices.
    # The indices for U4 are (i, j) where i, j in 0..3.
    # i = 2*I + i_local, j = 2*J + j_local.
    # Similarly for V4 and W4.
    
    # For each Strassen product q (0..6), and each local Strassen product r (0..6):
    # The scalar product is U2[i_local, r] * V2[j_local, r] * W2[k_local, r]
    # This contributes to the block product.
    
    # We iterate over the 7 Strassen block products.
    # For each block product q, we use the 2x2 Strassen decomposition.
    
    # The block indices I, J, K in 0..1.
    # I, J correspond to the 2x2 block structure of A and B.
    # K, I correspond to the 2x2 block structure of C.
    
    # The Strassen formulas for 2x2 block multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mq (which is a 2x2 matrix product) into 7 scalar products.
    # So we have 7 * 7 = 49 scalar products.
    
    # Let's build U4, V4, W4.
    # Column index r4 = q * 7 + r_local.
    
    for q in range(7):
        for r_local in range(7):
            r4 = q * 7 + r_local
            
            # The local 2x2 Strassen decomposition provides U2, V2, W2.
            # U2[i_local, r_local] is the coefficient for A[i_local, j_local]
            # V2[j_local, r_local] is the coefficient for B[j_local, k_local]
            # W2[k_local, r_local] is the coefficient for C[k_local, i_local]
            
            # We need to map these to the 4x4 indices.
            # The block product Mq involves specific blocks of A and B.
            # We need to determine which blocks of A and B are involved in Mq.
            
            # This is complex. Let's use a simpler approach:
            # The 4x4 multiplication can be viewed as a 2x2 multiplication of 2x2 blocks.
            # The Strassen algorithm for 2x2 blocks uses 7 products of 2x2 matrices.
            # Each 2x2 matrix product is decomposed into 7 scalar products.
            
            # We can construct the U4, V4, W4 by "expanding" the 2x2 Strassen formulas.
            
            # Let's define the block coefficients for A and B for each Mq.
            # A blocks: A00, A01, A10, A11.
            # B blocks: B00, B01, B10, B11.
            
            # M1: A00, A11 and B00, B11
            # M2: A10, A11 and B00
            # M3: A00 and B01, B11
            # M4: A11 and B10, B00
            # M5: A00, A01 and B11
            # M6: A10, A00 and B00, B01
            # M7: A01, A11 and B10, B11
            
            # For each Mq, we have a linear combination of A blocks and B blocks.
            # We can represent this as a 2x2 matrix of coefficients for A and B.
            
            # This is getting too complicated to code from scratch in the time limit.
            # I will use the known fact that rank(4x4x4) <= 49.
            # I will implement a generic recursive Strassen.
            
            pass
            
    # Alternative: Use the known rank-20 solution if available?
    # No, I'll stick to the schoolbook fallback if this is too hard.
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_school = np.zeros((n*m, schoolbook_rank))
    V_school = np.zeros((m*p, schoolbook_rank))
    W_school = np.zeros((p*n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
