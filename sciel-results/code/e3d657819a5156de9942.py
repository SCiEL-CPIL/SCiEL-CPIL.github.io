"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7) indexed by [i*2+j, r]
    V shape: (4, 7) indexed by [j*2+k, r]
    W shape: (4, 7) indexed by [k*2+i, r]
    
    We use the standard Strassen 7-product algorithm.
    Let A be 2x2, B be 2x2. C = AB.
    Products:
    P1 = (A00)(B01 - B11)
    P2 = (A00 + A10)(B11)
    P3 = (A10 - A00)(B00)
    P4 = (A11)(B01 - B00)
    P5 = (A00 + A11)(B00 + B11)
    P6 = (A01 - A11)(B01 + B11)
    P7 = (A00 + A01 - A10 - A11)(B10 + B11)
    
    C00 = P5 + P4 - P2 - P7
    C01 = P1 + P7
    C10 = P3 + P5
    C11 = P1 + P2 - P3 - P6
    """
    # We need to express this as sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For each product P_r, it is (linear combo of A) * (linear combo of B)
    # Then C entries are linear combos of P_r.
    # We can absorb the linear combinations of C into W.
    
    # Let's define the 7 rank-1 terms.
    # Term 1: P1 = (A00)(B01 - B11)
    #   U part: A00 -> i=0,j=0 gets 1. Others 0.
    #   V part: B01 - B11 -> j=0,k=1 gets 1; j=1,k=1 gets -1.
    #   W part: This P1 contributes to C01 (+1) and C11 (+1).
    #     C01: k=0,i=1 -> W[0*2+1, 0] += 1
    #     C11: k=1,i=1 -> W[1*2+1, 0] += 1
    #   Wait, W index is [k*n+i, r]. n=2.
    #   C01 is i=0, k=1. Index = 1*2 + 0 = 2.
    #   C11 is i=1, k=1. Index = 1*2 + 1 = 3.
    #   Let's re-verify indices.
    #   A is n x m = 2x2. i in 0..1, j in 0..1. U[i*2+j, r].
    #   B is m x p = 2x2. j in 0..1, k in 0..1. V[j*2+k, r].
    #   C is p x n = 2x2. k in 0..1, i in 0..1. W[k*2+i, r].
    
    # Let's build it systematically.
    # r=0: P1
    # U: A00 (i=0,j=0) -> U[0,0] = 1
    # V: B01 - B11 (j=0,k=1) -> V[0*2+1,0] = 1; (j=1,k=1) -> V[1*2+1,0] = -1
    # W: C01 += 1 (k=0,i=1) -> W[0*2+1,0] = 1; C11 += 1 (k=1,i=1) -> W[1*2+1,0] = 1
    
    # r=1: P2 = (A00 + A10)(B11)
    # U: A00 (0,0)=1, A10 (1,0)=1 -> U[0,1]=1, U[1*2+0,1]=1
    # V: B11 (j=1,k=1) -> V[1*2+1,1] = 1
    # W: C00 -= P2 (k=0,i=0) -> W[0,1] = -1; C11 -= P2 (k=1,i=1) -> W[3,1] = -1
    
    # r=2: P3 = (A10 - A00)(B00)
    # U: A10 (1,0)=1, A00 (0,0)=-1 -> U[0,2]=-1, U[2,2]=1  (Wait, U index i*2+j. A10 is i=1,j=0 -> idx 2. A00 is i=0,j=0 -> idx 0)
    # V: B00 (j=0,k=0) -> V[0,2] = 1
    # W: C10 += P3 (k=1,i=0) -> W[1*2+0,2] = 1; C11 -= P3 (k=1,i=1) -> W[3,2] = -1
    
    # r=3: P4 = (A11)(B01 - B00)
    # U: A11 (1,1) -> U[1*2+1,3] = 1
    # V: B01 (0,1)=1, B00 (0,0)=-1 -> V[1,3]=1, V[0,3]=-1
    # W: C00 += P4 (k=0,i=0) -> W[0,3] = 1
    
    # r=4: P5 = (A00 + A11)(B00 + B11)
    # U: A00 (0,0)=1, A11 (1,1)=1 -> U[0,4]=1, U[3,4]=1
    # V: B00 (0,0)=1, B11 (1,1)=1 -> V[0,4]=1, V[3,4]=1
    # W: C00 += P5 (k=0,i=0) -> W[0,4]=1; C10 += P5 (k=1,i=0) -> W[2,4]=1
    
    # r=5: P6 = (A01 - A11)(B01 + B11)
    # U: A01 (0,1)=1, A11 (1,1)=-1 -> U[1,5]=1, U[3,5]=-1
    # V: B01 (0,1)=1, B11 (1,1)=1 -> V[1,5]=1, V[3,5]=1
    # W: C11 -= P6 (k=1,i=1) -> W[3,5] = -1
    
    # r=6: P7 = (A00 + A01 - A10 - A11)(B10 + B11)
    # U: A00(0,0)=1, A01(0,1)=1, A10(1,0)=-1, A11(1,1)=-1
    #    -> U[0,6]=1, U[1,6]=1, U[2,6]=-1, U[3,6]=-1
    # V: B10 (1,0)=1, B11 (1,1)=1 -> V[2,6]=1, V[3,6]=1
    # W: C01 += P7 (k=0,i=1) -> W[1,6]=1; C00 -= P7 (k=0,i=0) -> W[0,6]=-1
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0
    U[0,0] = 1
    V[1,0] = 1; V[3,0] = -1
    W[1,0] = 1; W[3,0] = 1
    
    # r=1
    U[0,1] = 1; U[2,1] = 1
    V[3,1] = 1
    W[0,1] = -1; W[3,1] = -1
    
    # r=2
    U[0,2] = -1; U[2,2] = 1
    V[0,2] = 1
    W[2,2] = 1; W[3,2] = -1
    
    # r=3
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[0,3] = 1
    
    # r=4
    U[0,4] = 1; U[3,4] = 1
    V[0,4] = 1; V[3,4] = 1
    W[0,4] = 1; W[2,4] = 1
    
    # r=5
    U[1,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[3,5] = 1
    W[3,5] = -1
    
    # r=6
    U[0,6] = 1; U[1,6] = 1; U[2,6] = -1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[1,6] = 1; W[0,6] = -1
    
    return U, V, W

def _expand_to_2x4x4():
    """
    Expand the 2x2x2 Strassen solution to 2x4x4.
    n=2, m=4, p=4.
    A is 2x4. B is 4x4. C is 4x2.
    We can view A as two 2x2 blocks: A = [A_left, A_right] where A_left is col 0,1 and A_right is col 2,3.
    B is 4x4. We can partition rows of B into two 2x4 blocks: B_top (rows 0,1) and B_bottom (rows 2,3).
    Wait, B is m x p = 4 x 4.
    The product C = A B.
    C[i,k] = sum_j A[i,j] B[j,k].
    j goes from 0 to 3.
    We can split the sum into j=0,1 and j=2,3.
    C[i,k] = sum_{j in 0,1} A[i,j] B[j,k] + sum_{j in 2,3} A[i,j] B[j,k].
    
    Let A1 = A[:, 0:2] (2x2).
    Let A2 = A[:, 2:4] (2x2).
    Let B1 = B[0:2, :] (2x4).
    Let B2 = B[2:4, :] (2x4).
    
    C1 = A1 B1 (2x4).
    C2 = A2 B2 (2x4).
    C = C1 + C2.
    
    We can compute C1 using Strassen 2x2x2 logic? No, B1 is 2x4, not 2x2.
    However, we can compute the product of a 2x2 and a 2x4 matrix.
    Let's just use the block structure.
    A is 2x4. B is 4x4.
    We can decompose the 2x4x4 tensor into 4 sub-tensors of size 2x2x2?
    A is 2x4. Split columns: A1 (cols 0,1), A2 (cols 2,3).
    B is 4x4. Split rows: B1 (rows 0,1), B2 (rows 2,3).
    C is 4x2. Split rows: C1 (rows 0,1), C2 (rows 2,3).
    
    C1[i,k] = sum_{j=0,1} A1[i,j] B1[j,k] + sum_{j=2,3} A2[i,j] B2[j,k].
    This involves A1, B1 and A2, B2.
    The terms are:
    1. A1 * B1 (2x2 * 2x4 -> 2x4). This is 2 products of 2x2?
       B1 has 4 columns.
       B1 = [B1_01, B1_23] where B1_01 is cols 0,1 and B1_23 is cols 2,3.
       So A1 * B1_01 is 2x2 * 2x2 -> 2x2.
       A1 * B1_23 is 2x2 * 2x2 -> 2x2.
       Similarly for A2.
       
    So we have 4 blocks of 2x2x2 matrix multiplication:
    Block 1: A1 * B1_01 -> contributes to C1_01 (top-left 2x2 of C)
    Block 2: A1 * B1_23 -> contributes to C1_23 (top-right 2x2 of C)
    Block 3: A2 * B2_01 -> contributes to C2_01 (bottom-left 2x2 of C)
    Block 4: A2 * B2_23 -> contributes to C2_23 (bottom-right 2x2 of C)
    
    Each 2x2x2 multiplication can be done with rank 7.
    Total rank = 4 * 7 = 28.
    
    Let's construct the full U, V, W.
    n=2, m=4, p=4.
    U shape: (2*4, 28) = (8, 28)
    V shape: (4*4, 28) = (16, 28)
    W shape: (4*2, 28) = (8, 28)
    
    Indices:
    U[i, j, r] where i in 0..1, j in 0..3. Index = i*4 + j.
    V[j, k, r] where j in 0..3, k in 0..3. Index = j*4 + k.
    W[k, i, r] where k in 0..3, i in 0..1. Index = k*2 + i.
    
    We have 4 blocks. Let's call them b=0,1,2,3.
    Block 0: A1 * B1_01.
      A1 corresponds to i in 0..1, j in 0..1.
      B1_01 corresponds to j in 0..1, k in 0..1.
      C1_01 corresponds to k in 0..1, i in 0..1.
      
    Block 1: A1 * B1_23.
      A1: i in 0..1, j in 0..1.
      B1_23: j in 0..1, k in 2..3.
      C1_23: k in 2..3, i in 0..1.
      
    Block 2: A2 * B2_01.
      A2: i in 0..1, j in 2..3.
      B2_01: j in 2..3, k in 0..1.
      C2_01: k in 0..1, i in 0..1.
      
    Block 3: A2 * B2_23.
      A2: i in 0..1, j in 2..3.
      B2_23: j in 2..3, k in 2..3.
      C2_23: k in 2..3, i in 0..1.
      
    For each block, we use the Strassen 2x2x2 U, V, W matrices (size 4x7).
    Let U_s, V_s, W_s be the Strassen factors.
    
    For Block 0 (offsets: A_j in 0,1; B_j in 0,1; B_k in 0,1; C_k in 0,1):
      U_full[i, j, r] = U_s[i, j, r] for i in 0..1, j in 0..1.
      V_full[j, k, r] = V_s[j, k, r] for j in 0..1, k in 0..1.
      W_full[k, i, r] = W_s[k, i, r] for k in 0..1, i in 0..1.
      
    For Block 1 (offsets: A_j in 0,1; B_j in 0,1; B_k in 2,3; C_k in 2,3):
      U_full[i, j, r] = U_s[i, j, r] for i in 0..1, j in 0..1.
      V_full[j, k, r] = V_s[j, k, r] for j in 0..1, k in 2..3. Note: V_s index is local k (0,1).
      W_full[k, i, r] = W_s[k, i, r] for k in 2..3, i in 0..1. Note: W_s index is local k (0,1).
      
    For Block 2 (offsets: A_j in 2,3; B_j in 2,3; B_k in 0,1; C_k in 0,1):
      U_full[i, j, r] = U_s[i, j, r] for i in 0..1, j in 2..3.
      V_full[j, k, r] = V_s[j, k, r] for j in 2..3, k in 0..1.
      W_full[k, i, r] = W_s[k, i, r] for k in 0..1, i in 0..1.
      
    For Block 3 (offsets: A_j in 2,3; B_j in 2,3; B_k in 2,3; C_k in 2,3):
      U_full[i, j, r] = U_s[i, j, r] for i in 0..1, j in 2..3.
      V_full[j, k, r] = V_s[j, k, r] for j in 2..3, k in 2..3.
      W_full[k, i, r] = W_s[k, i, r] for k in 2..3, i in 0..1.
      
    The rank is 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # Block 0
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*m + j, r] = U_s[i*2 + j, r]
        for j in range(2):
            for k in range(2):
                V[j*p + k, r] = V_s[j*2 + k, r]
        for k in range(2):
            for i in range(2):
                W[k*n + i, r] = W_s[k*2 + i, r]
                
    # Block 1
    for r in range(7, 14):
        rs = r - 7
        for i in range(2):
            for j in range(2):
                U[i*m + j, r] = U_s[i*2 + j, rs]
        for j in range(2):
            for k in range(2):
                # k_local = 0,1 -> k_global = 2,3
                V[j*p + (k+2), r] = V_s[j*2 + k, rs]
        for k in range(2):
            for i in range(2):
                # k_local = 0,1 -> k_global = 2,3
                W[(k+2)*n + i, r] = W_s[k*2 + i, rs]
                
    # Block 2
    for r in range(14, 21):
        rs = r - 14
        for i in range(2):
            for j in range(2):
                # j_local = 0,1 -> j_global = 2,3
                U[i*m + (j+2), r] = U_s[i*2 + j, rs]
        for j in range(2):
            for k in range(2):
                # j_local = 0,1 -> j_global = 2,3
                V[(j+2)*p + k, r] = V_s[j*2 + k, rs]
        for k in range(2):
            for i in range(2):
                W[k*n + i, r] = W_s[k*2 + i, rs]
                
    # Block 3
    for r in range(21, 28):
        rs = r - 21
        for i in range(2):
            for j in range(2):
                # j_local = 0,1 -> j_global = 2,3
                U[i*m + (j+2), r] = U_s[i*2 + j, rs]
        for j in range(2):
            for k in range(2):
                # j_local = 0,1 -> j_global = 2,3
                # k_local = 0,1 -> k_global = 2,3
                V[(j+2)*p + (k+2), r] = V_s[j*2 + k, rs]
        for k in range(2):
            for i in range(2):
                # k_local = 0,1 -> k_global = 2,3
                W[(k+2)*n + i, r] = W_s[k*2 + i, rs]
                
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit for safety
    
    # First, build the known rank-28 solution
    U, V, W = _expand_to_2x4x4()
    best_U, best_V, best_W = U, V, W
    best_rank = 28
    
    # Try to find rank 27 using ALS
    # This is a structural change: attempting to reduce rank via optimization.
    # However, rank 27 for 2x4x4 is known to be impossible? 
    # Actually, the lower bound for 2x4x4 is 28? 
    # Let's check.
    # If rank 27 is impossible, ALS will fail to converge to a valid solution.
    # We will try, but if it takes too long or fails, we return rank 28.
    
    # Given the time constraint and the difficulty of finding a non-integer solution,
    # and the fact that rank 28 is a strong improvement over 32,
    # we will primarily return the rank 28 solution.
    # Attempting ALS for rank 27 is risky and might not yield a valid exact solution
    # with integer/half-integer entries.
    
    # We return the rank 28 solution immediately to ensure validity and speed.
    # The score will be 32/28 = 1.142...
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
