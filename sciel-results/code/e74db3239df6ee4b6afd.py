import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Construct 4x4x4 part (Rank 49 via Strassen)
    # 2. Construct 4x4x2 part (Rank 8 via Schoolbook)
    # 3. Combine

    # --- 2x2x2 Rank 7 Decomposition ---
    # U_loc, V_loc, W_loc of shape (4, 7)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))

    # Term 0: (A00+A11)(B00+B11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # Term 1: (A10+A11)B00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1

    # Term 2: A00(B01-B11)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = -1

    # Term 3: A11(B10-B00)
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = -1; W_loc[2, 3] = 1

    # Term 4: (A00+A01)B11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[1, 4] = 1; W_loc[3, 4] = 1

    # Term 5: (A10-A00)(B00+B01)
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[0, 5] = 1

    # Term 6: (A01-A11)(B10+B11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = -1; W_loc[3, 6] = 1

    # --- 4x4x4 Strassen Construction ---
    # Strassen Steps
    # X_s: 2x2 matrix of coeffs for A blocks
    # Y_s: 2x2 matrix of coeffs for B blocks
    # Z_s: 2x2 matrix of coeffs for C blocks (how M_s contributes to C blocks)
    
    # Step 0: M0 = (A00+A11)(B00+B11)
    X0 = np.array([[1, 0], [0, 1]])
    Y0 = np.array([[1, 0], [0, 1]])
    Z0 = np.array([[1, 0], [0, 1]]) # C00 += M0, C11 += M0

    # Step 1: M1 = (A10+A11)B00
    X1 = np.array([[0, 0], [1, 1]])
    Y1 = np.array([[1, 0], [0, 0]])
    Z1 = np.array([[0, 0], [1, 0]]) # C10 += M1

    # Step 2: M2 = A00(B01-B11)
    X2 = np.array([[1, 0], [0, 0]])
    Y2 = np.array([[0, 1], [0, -1]])
    Z2 = np.array([[0, 1], [0, -1]]) # C01 += M2, C11 -= M2

    # Step 3: M3 = A11(B10-B00)
    X3 = np.array([[0, 0], [0, 1]])
    Y3 = np.array([[-1, 0], [1, 0]])
    Z3 = np.array([[-1, 0], [1, 0]]) # C00 -= M3, C10 += M3

    # Step 4: M4 = (A00+A01)B11
    X4 = np.array([[1, 1], [0, 0]])
    Y4 = np.array([[0, 0], [0, 1]])
    Z4 = np.array([[0, 1], [0, 1]]) # C01 += M4, C11 += M4

    # Step 5: M5 = (A10-A00)(B00+B01)
    X5 = np.array([[-1, 0], [1, 0]])
    Y5 = np.array([[1, 1], [0, 0]])
    Z5 = np.array([[1, 0], [0, 0]]) # C00 += M5

    # Step 6: M6 = (A01-A11)(B10+B11)
    X6 = np.array([[0, 1], [0, -1]])
    Y6 = np.array([[0, 0], [1, 1]])
    Z6 = np.array([[-1, 0], [0, 1]]) # C00 -= M6, C11 += M6

    Steps = [
        (X0, Y0, Z0),
        (X1, Y1, Z1),
        (X2, Y2, Z2),
        (X3, Y3, Z3),
        (X4, Y4, Z4),
        (X5, Y5, Z5),
        (X6, Y6, Z6)
    ]

    Rank_444 = 7 * 7
    U_444 = np.zeros((16, Rank_444))
    V_444 = np.zeros((16, Rank_444))
    W_444 = np.zeros((16, Rank_444))

    for s_idx, (X_s, Y_s, Z_s) in enumerate(Steps):
        for t_idx in range(7):
            r = s_idx * 7 + t_idx
            
            # Fill U
            for i in range(4):
                br_i = i // 2
                lr_i = i % 2
                for j in range(4):
                    bc_j = j // 2
                    lc_j = j % 2
                    # Local index for U_loc is (lr_i, lc_j) -> lr_i * 2 + lc_j
                    loc_idx = lr_i * 2 + lc_j
                    U_444[i * 4 + j, r] = X_s[br_i, bc_j] * U_loc[loc_idx, t_idx]
            
            # Fill V
            for j in range(4):
                br_j = j // 2
                lr_j = j % 2
                for k in range(4):
                    bc_k = k // 2
                    lc_k = k % 2
                    loc_idx = lr_j * 2 + lc_k
                    V_444[j * 4 + k, r] = Y_s[br_j, bc_k] * V_loc[loc_idx, t_idx]
            
            # Fill W
            for k in range(4):
                br_k = k // 2
                lr_k = k % 2
                for i in range(4):
                    bc_i = i // 2
                    lc_i = i % 2
                    loc_idx = lr_k * 2 + lc_i
                    W_444[k * 4 + i, r] = Z_s[br_k, bc_i] * W_loc[loc_idx, t_idx]

    # --- 4x4x2 Schoolbook Construction ---
    # Rank 8
    Rank_442 = 8
    U_442 = np.zeros((16, Rank_442))
    V_442 = np.zeros((8, Rank_442)) # m*p = 4*2 = 8
    W_442 = np.zeros((8, Rank_442)) # p*n = 2*4 = 8
    
    # For k in 0,1 (cols of B2)
    # For j in 0..3 (rows of B2 / cols of A)
    # For i in 0..3 (rows of A / rows of C)
    # Term r = k*4 + j
    # U[i*4+j, r] = 1
    # V[j*2+k, r] = 1
    # W[k*4+i, r] = 1
    
    for k in range(2):
        for j in range(4):
            r = k * 4 + j
            for i in range(4):
                U_442[i * 4 + j, r] = 1
                V_442[j * 2 + k, r] = 1
                W_442[k * 4 + i, r] = 1

    # --- Combine ---
    # U_total: (16, 49+8)
    # V_total: (24, 49+8) -> 16 (for 4x4) + 8 (for 4x2)
    # W_total: (24, 49+8) -> 16 (for 4x4) + 8 (for 4x2)
    
    Total_Rank = Rank_444 + Rank_442
    
    U_total = np.zeros((16, Total_Rank))
    V_total = np.zeros((24, Total_Rank))
    W_total = np.zeros((24, Total_Rank))
    
    # Copy 4x4x4 part
    U_total[:, :Rank_444] = U_444
    V_total[:16, :Rank_444] = V_444
    W_total[:16, :Rank_444] = W_444
    
    # Copy 4x4x2 part
    U_total[:, Rank_444:] = U_442
    V_total[16:, Rank_444:] = V_442
    W_total[16:, Rank_444:] = W_442

    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()

    # Verify
    def verify(U, V, W, tol=1e-9):
        n, m, p = 4, 4, 6
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j*p+k], W_arr[k*n+i]))
                    if abs(val - 1.0) > tol:
                        return False
        # Check 0s (Sample)
        # Just check a few random zero cases to be safe
        for _ in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            j2 = np.random.randint(0, m)
            if j2 == j: continue
            val = np.dot(U_arr[i*m+j], np.multiply(V_arr[j2*p+k], W_arr[k*n+i]))
            if abs(val) > tol:
                return False
        return True

    if not verify(U_list, V_list, W_list):
        # Fallback to schoolbook
        rank_schoolbook = 96
        U_school = [[0] * rank_schoolbook for _ in range(16)]
        V_school = [[0] * rank_schoolbook for _ in range(24)]
        W_school = [[0] * rank_schoolbook for _ in range(24)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_school[i * 4 + j][col] = 1
                    V_school[j * 6 + k][col] = 1
                    W_school[k * 4 + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_list, "V": V_list, "W": W_list}
