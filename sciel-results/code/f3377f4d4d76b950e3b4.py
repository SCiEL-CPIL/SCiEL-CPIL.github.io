import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Strassen decomposition for 4x4x4 matrix multiplication
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our case: A is 4x5, B is 5x5
    # Split: A = [A1(4x4), A2(4x1)], B = [B1(4x5); B2(1x5)]
    # C = A1*B1 + A2*B1 + A1*B2 + A2*B2
    
    # We'll use:
    # 1. Strassen for A1 * B1[:4] (4x4 * 4x4 part of B1)
    # 2. Direct for A1 * B1[4:] (4x4 * 4x1)
    # 3. Direct for A2 * B1 (4x1 * 4x5)
    # 4. Direct for A1 * B2 (4x4 * 1x5)
    # 5. Direct for A2 * B2 (4x1 * 1x5)
    
    # Total rank = 7 (Strassen) + 4 + 20 + 4 + 4 = 39? 
    # Wait, let me recount:
    # A1*B1[:4]: 7 (Strassen)
    # A1*B1[4:]: 4 (each of 4 rows of A1 times the 4x1 column)
    # A2*B1: 4 (each of 4 scalars in A2 times 4x5 matrix B1)
    # A1*B2: 4 (each of 4 columns of A1 times the 1x5 row B2)
    # A2*B2: 4 (outer product of 4x1 and 1x5)
    # Total: 7 + 4 + 4 + 4 + 4 = 23
    
    # Let me build the factor matrices
    rank = 23
    U = np.zeros((n * m, rank))  # 20 x 23
    V = np.zeros((m * p, rank))  # 25 x 23
    W = np.zeros((p * n, rank))  # 20 x 23
    
    # Indexing:
    # U: row index = i*m + j (i in 0..3, j in 0..4)
    # V: row index = j*p + k (j in 0..4, k in 0..4)
    # W: row index = k*n + i (k in 0..4, i in 0..3)
    
    # Strassen part: A1 (4x4) * B1_first4 (4x4)
    # A1[i,j] for i,j in 0..3
    # B1_first4[j,k] for j,k in 0..3
    # Result C_strassen[i,k] for i,k in 0..3
    
    # Strassen factors for 4x4x4:
    # M1 = (A11 + A22)(B11 + B22)
    # A11 = A1[0:2,0:2], A22 = A1[2:4,2:4]
    # B11 = B1[0:2,0:2], B22 = B1[2:4,2:4]
    
    # For Strassen, we need to express each multiplication as sum of products
    # M1: (A11 + A22) * (B11 + B22)
    # This contributes to C11 and C22
    
    # Let me define the Strassen multiplications explicitly:
    # Each Mq is a 4x4 matrix product computed with rank 7
    
    # Actually, let me use a simpler approach: use the known rank-7 decomposition for 4x4x4
    
    # The 4x4x4 tensor can be decomposed with rank 7 using Strassen's algorithm
    # The factors are:
    
    # For Strassen 2x2x2, the rank is 7. For 4x4x4, we can use 7 multiplications
    # where each "multiplication" is a 2x2 block multiplication
    
    # Let me construct it step by step:
    
    # Strassen multiplication indices: 0-6
    # M0 = (A11+A22)*(B11+B22)
    # M1 = (A21+A22)*B11
    # M2 = A11*(B12-B22)
    # M3 = A22*(B21-B11)
    # M4 = (A11+A12)*B22
    # M5 = (A21-A11)*(B11+B12)
    # M6 = (A12-A22)*(B21+B22)
    
    # C11 = M0 + M3 - M4 + M6
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M1 + M2 + M5
    
    # Now for our 4x4x4 case (A1 is 4x4, B1_first4 is 4x4):
    # A1 = [[A11, A12], [A21, A22]] where each block is 2x2
    # B1f = [[B11, B12], [B21, B22]] where each block is 2x2
    
    # The output C_strassen = [[C11, C12], [C21, C22]]
    
    # Let me assign column indices for Strassen: 0-6
    col = 0
    
    # M0 = (A11+A22)*(B11+B22)
    # A11+A22: elements at positions (0,0),(0,1),(1,0),(1,1) and (2,2),(2,3),(3,2),(3,3)
    # For U: we need A1[i,j] for the relevant (i,j)
    # For V: we need B1f[j,k] for the relevant (j,k)
    # For W: we need C[i,k] for the relevant (i,k)
    
    # This is getting complex. Let me use a different approach:
    # I'll directly construct the tensor and use a known decomposition.
    
    # Actually, let me just use the schoolbook for now and verify it works,
    # then try to optimize.
    
    # Schoolbook decomposition:
    rank_sb = n * m * p  # 100
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Now let me try the Strassen-based decomposition with rank 23
    # But first, let me verify the schoolbook works
    
    # Actually, let me implement the Strassen decomposition properly.
    
    # For 4x4x4 using Strassen (rank 7):
    # The factor matrices for the 4x4x4 tensor T444 where T444[i,j,k] = 1 if j is the middle index
    
    # Let me define the Strassen factors for 4x4x4:
    # U444: 16x7, V444: 16x7, W444: 16x7
    
    # Using the standard Strassen decomposition:
    U444 = np.zeros((16, 7))
    V444 = np.zeros((16, 7))
    W444 = np.zeros((16, 7))
    
    # Block indices:
    # A blocks: A11=(0,1)x(0,1), A12=(0,1)x(2,3), A21=(2,3)x(0,1), A22=(2,3)x(2,3)
    # B blocks: B11=(0,1)x(0,1), B12=(0,1)x(2,3), B21=(2,3)x(0,1), B22=(2,3)x(2,3)
    # C blocks: C11=(0,1)x(0,1), C12=(0,1)x(2,3), C21=(2,3)x(0,1), C22=(2,3)x(2,3)
    
    # M0 = (A11+A22)*(B11+B22)
    # A11+A22 has 1: at (0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)
    # B11+B22 has 1: at (0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)
    # C contribution: C11 gets +1, C22 gets +1
    for i in [0,1,2,3]:
        for j in [0,1,2,3]:
            if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                U444[i*4+j, 0] = 1.0
                V444[j*4+i, 0] = 1.0  # Wait, V index is j*p+k, but for 4x4x4, p=4
                # Actually for 4x4x4: U index = i*4+j, V index = j*4+k, W index = k*4+i
                # So for M0:
                # U444[i*4+j, 0] = 1 for i,j in A11+A22
                # V444[j*4+k, 0] = 1 for j,k in B11+B22
                # W444[k*4+i, 0] = 1 for k,i in C11+C22
    # This is getting messy. Let me use a cleaner approach.
    
    # I'll use the fact that the 4x4x4 multiplication tensor has rank 7 (Strassen)
    # and construct the factors explicitly.
    
    # Actually, let me just return the schoolbook for now to ensure correctness,
    # and note that a better decomposition exists.
    
    # But the task asks to find rank-minimal. Let me try rank 23.
    
    # I'll construct the rank-23 decomposition:
    
    # Part 1: Strassen for A1 * B1[:4] (4x4 * 4x4) -> rank 7
    # Part 2: A1 * B1[4:] (4x4 * 4x1) -> rank 4
    # Part 3: A2 * B1 (4x1 * 4x5) -> rank 4
    # Part 4: A1 * B2 (4x4 * 1x5) -> rank 4
    # Part 5: A2 * B2 (4x1 * 1x5) -> rank 4
    
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for 4x4x4 (A1 * B1_first4)
    # A1[i,j] for i,j in 0..3
    # B1_first4[j,k] for j,k in 0..3
    # C1[i,k] for i,k in 0..3
    
    # Strassen factors for 4x4x4:
    # I'll use the standard decomposition
    
    # Let me define the 7 Strassen multiplications:
    # For each multiplication q in 0..6:
    #   U_part1[i*4+j, q] = coefficient for A1[i,j]
    #   V_part1[j*4+k, q] = coefficient for B1_first4[j,k]  
    #   W_part1[k*4+i, q] = coefficient for C1[i,k]
    
    # Standard Strassen for 2x2x2 extended to 4x4x4:
    # Actually, Strassen's algorithm works on 2x2 blocks, so for 4x4 we have
    # 2x2 blocks of 2x2 matrices.
    
    # Let me use a known rank-7 decomposition for 4x4x4:
    
    # The 4x4x4 tensor T444[i,j,k] = delta_{i,k} * delta_{j,j} ... no.
    # T444[i,j,k] = 1 if the multiplication (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # For Strassen, the factors are:
    
    # M0: (A11+A22)(B11+B22)
    # A11+A22: rows 0,1,2,3; cols 0,1,2,3 with A11 at (0:2,0:2) and A22 at (2:4,2:4)
    # So U444[i*4+j, 0] = 1 if (i<2 and j<2) or (i>=2 and j>=2)
    # B11+B22: V444[j*4+k, 0] = 1 if (j<2 and k<2) or (j>=2 and k>=2)
    # C11+C22: W444[k*4+i, 0] = 1 if (k<2 and i<2) or (k>=2 and i>=2)
    
    # Let me construct all 7:
    
    # M0: (A11+A22)(B11+B22)
    for i in range(4):
        for j in range(4):
            if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                U[i * m + j, 0] = 1.0  # A1[i,j]
    for j in range(4):
        for k in range(4):
            if (j < 2 and k < 2) or (j >= 2 and k >= 2):
                V[j * p + k, 0] = 1.0  # B1[j,k]
    for k in range(4):
        for i in range(4):
            if (k < 2 and i < 2) or (k >= 2 and i >= 2):
                W[k * n + i, 0] = 1.0  # C[i,k]
    
    # M1: (A21+A22)B11
    for i in range(4):
        for j in range(4):
            if i >= 2 and (j < 2 or j >= 2):  # A21+A22: rows 2,3
                U[i * m + j, 1] = 1.0
    for j in range(4):
        for k in range(4):
            if j < 2 and k < 2:  # B11
                V[j * p + k, 1] = 1.0
    for k in range(4):
        for i in range(4):
            if k < 2 and i >= 2:  # C21
                W[k * n + i, 1] = 1.0
    
    # M2: A11(B12-B22)
    for i in range(4):
        for j in range(4):
            if i < 2 and j < 2:  # A11
                U[i * m + j, 2] = 1.0
    for j in range(4):
        for k in range(4):
            if (j < 2 and k >= 2):  # B12
                V[j * p + k, 2] = 1.0
            elif (j >= 2 and k >= 2):  # -B22
                V[j * p + k, 2] = -1.0
    for k in range(4):
        for i in range(4):
            if k >= 2 and i < 2:  # C12
                W[k * n + i, 2] = 1.0
    
    # M3: A22(B21-B11)
    for i in range(4):
        for j in range(4):
            if i >= 2 and j >= 2:  # A22
                U[i * m + j, 3] = 1.0
    for j in range(4):
        for k in range(4):
            if (j >= 2 and k < 2):  # B21
                V[j * p + k, 3] = 1.0
            elif (j < 2 and k < 2):  # -B11
                V[j * p + k, 3] = -1.0
    for k in range(4):
        for i in range(4):
            if (k < 2 and i >= 2):  # C21
                W[k * n + i, 3] = 1.0
            if (k >= 2 and i >= 2):  # C22
                W[k * n + i, 3] = 1.0
    
    # Wait, I'm making errors. Let me restart with a cleaner Strassen decomposition.
    
    # Actually, let me just use the schoolbook decomposition to ensure correctness.
    # The rank is 100, which is valid.
    
    # But I want to try for a lower rank. Let me use a known result:
    # For 4x5x5, the minimum rank is at most 4*5*5 = 100 (schoolbook)
    # Using Strassen for the 4x4x4 part gives at most 7 + 4*5 + 4*5 + 4 = 7+20+20+4 = 51?
    
    # Let me try a different decomposition:
    # C = A * B
    # Split B into B = [B1(5x4), B2(5x1)]
    # C = A*B1 + A*B2
    # A*B1: 4x5 * 5x4 = 4x4, use Strassen-like? No, Strassen is for square.
    
    # Actually, I think the best I can do quickly is return the schoolbook.
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
