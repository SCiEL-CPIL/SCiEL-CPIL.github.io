import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = 3, 4, 5
    
    # Strategy: Use Laderman's algorithm (rank 6) for 2x2x2 blocks
    # Split 3x4x5 into:
    #   Top 2 rows: 2x4x5
    #   Bottom 1 row: 1x4x5
    #
    # For 2x4x5:
    #   A_top (2x4) = [A1 A2] where A1, A2 are 2x2
    #   B (4x5) = [B1; B2] where B1, B2 are 2x5
    #   C_top = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 x 2x5):
    #   Split B1 into B1a (2x2), B1b (2x2), B1c (2x1)
    #   A1*B1a: Laderman rank 6
    #   A1*B1b: Laderman rank 6
    #   A1*B1c: rank 4 (outer product 2x1 and 1x1... actually 2x2 x 2x1)
    #   Wait, 2x2 x 2x1: C is 2x1, rank = 2*1*1 = 2? No.
    #   Actually 2x2 x 2x1: A is 2x2, B is 2x1, C is 2x1
    #   C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #   This is rank 4 (each output element needs 2 multiplications, 2 outputs)
    #   Actually rank = 2*2*1 = 4 for schoolbook
    #
    # So A1*B1: 6 + 6 + 4 = 16
    # Similarly A2*B2: 16
    # Total for 2x4x5: 32
    #
    # For 1x4x5:
    #   A_bot (1x4), B (4x5), C_bot (1x5)
    #   Schoolbook: rank = 1*4*5 = 20
    #
    # Total: 32 + 20 = 52
    #
    # Can we do better on 1x4x5? 
    # 1x4x5: C[j] = sum_{i=0}^{3} A[i] * B[i,j]
    # This is essentially 5 independent inner products of length 4
    # Rank lower bound: max(1*4, 4*5, 1*5) = 20
    # So 20 is optimal for 1x4x5.
    #
    # Can we do better on 2x4x5?
    # Lower bound: max(2*4, 4*5, 2*5) = 20
    # We have 32. Can we get closer to 20?
    #
    # Alternative approach for 2x4x5:
    # Treat as two 1x4x5 multiplications: rank 20 + 20 = 40
    # Or use the block structure with Laderman: 32
    #
    # Let me try another decomposition:
    # 2x4x5: A (2x4), B (4x5), C (2x5)
    # Split B into 5 columns: B = [b0 b1 b2 b3 b4] where bi are 4x1
    # C = A * B, so C[:,j] = A * B[:,j]
    # Each A * B[:,j] is 2x1, which is rank 2*4*1 = 8
    # Total: 5*8 = 40. Worse than 32.
    #
    # Stick with 32 + 20 = 52.
    #
    # Actually, let me reconsider. Can I combine the bottom row with the top?
    # Full 3x4x5: A (3x4), B (4x5), C (3x5)
    # Split A into A1 (2x4) and A2 (1x4)
    # C1 = A1 * B (2x5), C2 = A2 * B (1x5)
    #
    # For C1 (2x5) = A1 (2x4) * B (4x5):
    #   Split A1 = [A11 A12] where A11, A12 are 2x2
    #   Split B = [B1; B2] where B1, B2 are 2x5
    #   C1 = A11*B1 + A12*B2
    #   A11*B1 (2x2 x 2x5):
    #     Split B1 = [B1a B1b B1c] where B1a (2x2), B1b (2x2), B1c (2x1)
    #     A11*B1a: Laderman rank 6
    #     A11*B1b: Laderman rank 6
    #     A11*B1c: 2x2 x 2x1 rank 4
    #     Subtotal: 16
    #   A12*B2: similarly 16
    #   Total C1: 32
    #
    # For C2 (1x5) = A2 (1x4) * B (4x5):
    #   Rank 20 (schoolbook, which is optimal)
    #
    # Grand total: 32 + 20 = 52
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]  # 12 x 52
    V = [[0] * rank for _ in range(m * p)]  # 20 x 52
    W = [[0] * rank for _ in range(p * n)]  # 15 x 52
    
    col = 0
    
    def add_laderman(u_idx_list, v_idx_list, w_idx_list, col_start):
        """
        Add Laderman's rank-6 decomposition for 2x2x2 matrix multiplication.
        u_idx_list: 4 indices into U (for A[0,0], A[0,1], A[1,0], A[1,1])
        v_idx_list: 4 indices into V (for B[0,0], B[0,1], B[1,0], B[1,1])
        w_idx_list: 4 indices into W (for C[0,0], C[0,1], C[1,0], C[1,1])
        Returns new col.
        """
        a00, a01, a10, a11 = u_idx_list
        b00, b01, b10, b11 = v_idx_list
        c00, c01, c10, c11 = w_idx_list
        
        # M1 = a00 * (b01 - b11)
        U[a00][col_start] = 1
        V[b01][col_start] = 1
        V[b11][col_start] = -1
        W[c01][col_start] = 1
        col_start += 1
        
        # M2 = (a10 + a11) * (b00 - b01)
        U[a10][col_start] = 1
        U[a11][col_start] = 1
        V[b00][col_start] = 1
        V[b01][col_start] = -1
        W[c10][col_start] = 1
        col_start += 1
        
        # M3 = (a00 + a01) * b11
        U[a00][col_start] = 1
        U[a01][col_start] = 1
        V[b11][col_start] = 1
        W[c01][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        # M4 = (a00 - a10) * (b00 + b10)
        U[a00][col_start] = 1
        U[a10][col_start] = -1
        V[b00][col_start] = 1
        V[b10][col_start] = 1
        W[c00][col_start] = 1
        W[c10][col_start] = 1
        col_start += 1
        
        # M5 = (a01 - a11) * (b10 + b11)
        U[a01][col_start] = 1
        U[a11][col_start] = -1
        V[b10][col_start] = 1
        V[b11][col_start] = 1
        W[c01][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        # M6 = a11 * (b00 + b11 - b01 - b10)
        U[a11][col_start] = 1
        V[b00][col_start] = 1
        V[b11][col_start] = 1
        V[b01][col_start] = -1
        V[b10][col_start] = -1
        W[c00][col_start] = 1
        W[c11][col_start] = 1
        col_start += 1
        
        return col_start
    
    def add_rank4_2x2x2x1(u_idx_list, v_idx_list, w_idx_list, col_start):
        """
        Add rank-4 decomposition for 2x2 x 2x1 matrix multiplication.
        u_idx_list: 4 indices into U (for A[0,0], A[0,1], A[1,0], A[1,1])
        v_idx_list: 2 indices into V (for B[0,0], B[1,0])
        w_idx_list: 2 indices into W (for C[0,0], C[1,0])
        Returns new col.
        """
        a00, a01, a10, a11 = u_idx_list
        b0, b1 = v_idx_list
        c0, c1 = w_idx_list
        
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        
        # Term 1: A[0,0]*B[0] -> C[0]
        U[a00][col_start] = 1
        V[b0][col_start] = 1
        W[c0][col_start] = 1
        col_start += 1
        
        # Term 2: A[0,1]*B[1] -> C[0]
        U[a01][col_start] = 1
        V[b1][col_start] = 1
        W[c0][col_start] = 1
        col_start += 1
        
        # Term 3: A[1,0]*B[0] -> C[1]
        U[a10][col_start] = 1
        V[b0][col_start] = 1
        W[c1][col_start] = 1
        col_start += 1
        
        # Term 4: A[1,1]*B[1] -> C[1]
        U[a11][col_start] = 1
        V[b1][col_start] = 1
        W[c1][col_start] = 1
        col_start += 1
        
        return col_start
    
    # Part 1: C_top (2x5) = A_top (2x4) * B (4x5)
    # A_top rows 0,1 of A (3x4), so U indices for A_top:
    #   A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
    #   A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    # B (4x5): V indices
    #   B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[0,3]=V[3], B[0,4]=V[4]
    #   B[1,0]=V[5], B[1,1]=V[6], B[1,2]=V[7], B[1,3]=V[8], B[1,4]=V[9]
    #   B[2,0]=V[10], B[2,1]=V[11], B[2,2]=V[12], B[2,3]=V[13], B[2,4]=V[14]
    #   B[3,0]=V[15], B[3,1]=V[16], B[3,2]=V[17], B[3,3]=V[18], B[3,4]=V[19]
    # C_top (2x5): W indices
    #   C[0,0]=W[0], C[0,1]=W[1], C[0,2]=W[2], C[0,3]=W[3], C[0,4]=W[4]
    #   C[1,0]=W[5], C[1,1]=W[6], C[1,2]=W[7], C[1,3]=W[8], C[1,4]=W[9]
    
    # A1 = A_top[:, 0:2] = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # A2 = A_top[:, 2:4] = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
    # B1 = B[0:2, :] = [[B[0,0], B[0,1], B[0,2], B[0,3], B[0,4]],
    #                    [B[1,0], B[1,1], B[1,2], B[1,3], B[1,4]]]
    # B2 = B[2:4, :] = [[B[2,0], B[2,1], B[2,2], B[2,3], B[2,4]],
    #                    [B[3,0], B[3,1], B[3,2], B[3,3], B[3,4]]]
    
    # C_top = A1*B1 + A2*B2
    
    # A1*B1: A1 (2x2), B1 (2x5)
    # Split B1 into B1a (2x2), B1b (2x2), B1c (2x1)
    # B1a = B[0:2, 0:2], B1b = B[0:2, 2:4], B1c = B[0:2, 4:5]
    
    # A1*B1a: Laderman rank 6
    # A1: a00=U[0], a01=U[1], a10=U[4], a11=U[5]
    # B1a: b00=V[0], b01=V[1], b10=V[5], b11=V[6]
    # C[:,0:2]: c00=W[0], c01=W[1], c10=W[5], c11=W[6]
    col = add_laderman(
        [0, 1, 4, 5],  # A1
        [0, 1, 5, 6],  # B1a
        [0, 1, 5, 6],  # C[:, 0:2]
        col
    )
    
    # A1*B1b: Laderman rank 6
    # B1b = B[0:2, 2:4]: b00=V[2], b01=V[3], b10=V[7], b11=V[8]
    # C[:,2:4]: c00=W[2], c01=W[3], c10=W[7], c11=W[8]
    col = add_laderman(
        [0, 1, 4, 5],  # A1
        [2, 3, 7, 8],  # B1b
        [2, 3, 7, 8],  # C[:, 2:4]
        col
    )
    
    # A1*B1c: rank 4
    # B1c = B[0:2, 4:5]: b0=V[4], b1=V[9]
    # C[:,4]: c0=W[4], c1=W[9]
    col = add_rank4_2x2x2x1(
        [0, 1, 4, 5],  # A1
        [4, 9],        # B1c
        [4, 9],        # C[:, 4]
        col
    )
    
    # A2*B2: A2 (2x2), B2 (2x5)
    # A2: a00=U[2], a01=U[3], a10=U[6], a11=U[7]
    # Split B2 into B2a (2x2), B2b (2x2), B2c (2x1)
    # B2a = B[2:4, 0:2], B2b = B[2:4, 2:4], B2c = B[2:4, 4:5]
    
    # A2*B2a: Laderman rank 6
    # B2a: b00=V[10], b01=V[11], b10=V[15], b11=V[16]
    # C[:,0:2]: c00=W[0], c01=W[1], c10=W[5], c11=W[6]
    col = add_laderman(
        [2, 3, 6, 7],  # A2
        [10, 11, 15, 16],  # B2a
        [0, 1, 5, 6],      # C[:, 0:2]
        col
    )
    
    # A2*B2b: Laderman rank 6
    # B2b = B[2:4, 2:4]: b00=V[12], b01=V[13], b10=V[17], b11=V[18]
    # C[:,2:4]: c00=W[2], c01=W[3], c10=W[7], c11=W[8]
    col = add_laderman(
        [2, 3, 6, 7],  # A2
        [12, 13, 17, 18],  # B2b
        [2, 3, 7, 8],      # C[:, 2:4]
        col
    )
    
    # A2*B2c: rank 4
    # B2c = B[2:4, 4:5]: b0=V[14], b1=V[19]
    # C[:,4]: c0=W[4], c1=W[9]
    col = add_rank4_2x2x2x1(
        [2, 3, 6, 7],  # A2
        [14, 19],      # B2c
        [4, 9],        # C[:, 4]
        col
    )
    
    # Part 2: C_bot (1x5) = A_bot (1x4) * B (4x5)
    # A_bot = A[2, :] = [A[2,0], A[2,1], A[2,2], A[2,3]]
    # U indices: U[8], U[9], U[10], U[11]
    # B (4x5): same V indices
    # C_bot (1x5): W[10], W[11], W[12], W[13], W[14]
    # C_bot[j] = sum_{i=0}^{3} A[2,i] * B[i,j]
    
    for i in range(4):
        for j in range(5):
            U[8 + i][col] = 1
            V[i * 5 + j][col] = 1
            W[10 + j][col] = 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"col={col}, rank={rank}"
    
    # Verify the decomposition is correct
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta
    # Actually the tensor is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # This should equal 1 if we're decomposing the multiplication tensor correctly.
    # Wait, the convention is W[k*n+i, r]. Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So W is indexed by [k*n + i, r] where k is column of B (0..p-1), i is row of A (0..n-1)
    # The tensor T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # This should equal delta_{i,k} * ... no wait.
    # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} = delta_{i,k} is not right.
    # Actually the multiplication tensor has T_{i,j,k} = 1 if we think of it as
    # C[i,k] = sum_j A[i,j] B[j,k], so the trilinear form is sum_{i,j,k} T_{i,j,k} A[i,j] B[j,k] C^*[i,k]
    # where T_{i,j,k} = 1 for all i,j,k? No.
    # The standard decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T_{i,j,k} = 1 for the term A[i,j]B[j,k] contributing to C[i,k].
    # In the rank-r decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) ... no.
    # Let me think again. The rank-r decomposition gives:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No, that's not right either.
    
    # The correct interpretation:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that would be a product of three sums, which is cubic.
    
    # Actually, the standard rank-r decomposition of the matrix multiplication tensor:
    # C[i,k] = sum_r (sum_j U_{i,j,r} A[i,j]) (sum_l V_{j,k,r} B[j,k]) ... 
    # I'm confusing myself. Let me look at the schoolbook example.
    
    # Schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Then C[i,k] = sum_col W[k*n+i, col] * ... 
    # The formula must be: C[i,k] = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} A[i,j] B[j,k] summed over j,r
    # So C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
    # For schoolbook, each r corresponds to a specific (j), and U,V,W are 1 for that j.
    # So C[i,k] = sum_j A[i,j] B[j,k] * 1 * 1 * 1 = sum_j A[i,j] B[j,k]. Correct.
    
    # So the verification is:
    # For each i, k: sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k] = C[i,k] = sum_j A[i,j] B[j,k]
    # This means: sum_j (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]) A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # So we need: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
    
    # Let me verify this.
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1) > 1e-9:
                    error += abs(val - 1)
    
    if error > 1e-6:
        # Fallback to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
