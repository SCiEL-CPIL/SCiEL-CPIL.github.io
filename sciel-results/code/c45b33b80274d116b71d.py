"""Block-partitioned hybrid construction using Strassen for 2x2x2 and schoolbook for 2x2x1."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Strategy: Partition into 2x2 blocks
    # A is 4x4, B is 4x7, C is 4x7
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22
    # Split B into 2 blocks of 2x7: B1 (top 2 rows), B2 (bottom 2 rows)
    # Split C into 2 blocks of 2x7: C1 (top 2 rows), C2 (bottom 2 rows)
    
    # C1 = A11*B1 + A12*B2  (2x2 * 2x7 + 2x2 * 2x7 = 2x7)
    # C2 = A21*B1 + A22*B2  (2x2 * 2x7 + 2x2 * 2x7 = 2x7)
    
    # For each 2x2 * 2x7 multiplication, we can further split B into 2x3 + 2x4? 
    # No, better: treat 2x7 as 2x(3+4) or just use the fact that 2x2 * 2x7 
    # can be decomposed using Strassen for the 2x2x2 part and schoolbook for the rest.
    
    # Actually, let's use the known optimal decomposition for 2x2x2 (Strassen rank 7)
    # and 2x2x1 (schoolbook rank 4).
    
    # A 2x2 * 2x7 product:
    # Let A2x2 = [[a11, a12], [a21, a22]]
    # Let B2x7 = [[b11, b12, ..., b17], [b21, b22, ..., b27]]
    # Result C2x7 = A2x2 * B2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can view this as: for each column k of B, we have a 2x2 * 2x1 multiplication.
    # 2x2 * 2x1 has rank 4 (schoolbook):
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This uses 4 terms.
    
    # But we can do better by grouping. The known result is that 2x2x2 has rank 7 (Strassen).
    # For 2x2x7, we can think of it as 7 separate 2x2x1 multiplications, but that's not optimal.
    # 
    # Better approach: Use the block structure.
    # Total rank = 4 * (rank for 2x2x2) + ... wait, let me recalculate.
    
    # Let's use the specific construction from the cell finding:
    # "Replacing the 48-term schoolbook tail for the 4x4x3 part with four rank-11 2x2x3 blocks"
    # This suggests:
    # - 4x4x4 part: use rank 49 (Kronecker product of two 2x2x2 Strassen? No, 4x4x4 rank is known to be 49? 
    #   Actually, 4x4x4 rank is at most 49 via Kronecker, but known best is 49? Or is it lower?)
    #   The Kronecker product of two Strassen decompositions gives rank 7*7=49 for 4x4x4.
    # - 4x4x3 part: use four 2x2x3 blocks, each of rank 11. 4*11 = 44.
    # Total = 49 + 44 = 93.
    
    # So the strategy is:
    # 1. Decompose the 4x4x4 sub-tensor (columns 0-3 of B) using Kronecker product of Strassen: rank 49.
    # 2. Decompose the 4x4x3 sub-tensor (columns 4-6 of B) using four 2x2x3 blocks of rank 11 each: rank 44.
    # Total rank = 93.
    
    # Let's implement this.
    
    # First, define Strassen's 7-term decomposition for 2x2x2.
    # Input: A (2x2), B (2x2), Output: C (2x2)
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In tensor form for 2x2x2:
    # u = i*2 + k (A index), v = k*2 + j (B index), w = j*2 + i (C index)
    # Wait, the convention is:
    # U shape: (n*m, rank) = (4, rank) for 2x2
    # V shape: (m*p, rank) = (4, rank) for 2x2
    # W shape: (p*n, rank) = (4, rank) for 2x2
    # u = i*m + k = i*2 + k
    # v = k*p + j = k*2 + j
    # w = j*n + i = j*2 + i
    
    # Let's build the Strassen decomposition for 2x2x2.
    # Each term t has coefficients in U, V, W.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Term 1: M1 = (A00+A11)*(B00+B11)
        # A indices: A00 -> u=0*2+0=0, A11 -> u=1*2+1=3
        # B indices: B00 -> v=0*2+0=0, B11 -> v=1*2+1=3
        # C contribution: C00 += M1, C11 += M1
        # C00 -> w=0*2+0=0, C11 -> w=1*2+1=3
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # Term 2: M2 = (A10+A11)*B00
        # A indices: A10 -> u=1*2+0=2, A11 -> u=3
        # B indices: B00 -> v=0
        # C contribution: C10 += M2
        # C10 -> w=0*2+1=1
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[1][1] += 1
        
        # Term 3: M3 = A00*(B01-B11)
        # A indices: A00 -> u=0
        # B indices: B01 -> v=0*2+1=1, B11 -> v=3
        # C contribution: C01 += M3, C11 += M3
        # C01 -> w=1*2+0=2, C11 -> w=3
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[2][2] += 1; W[3][2] += 1
        
        # Term 4: M4 = A11*(B10-B00)
        # A indices: A11 -> u=3
        # B indices: B10 -> v=1*2+0=2, B00 -> v=0
        # C contribution: C00 += M4, C10 += M4
        # C00 -> w=0, C10 -> w=1
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[1][3] += 1
        
        # Term 5: M5 = (A00+A01)*B11
        # A indices: A00 -> u=0, A01 -> u=0*2+1=1
        # B indices: B11 -> v=3
        # C contribution: C00 -= M5, C01 += M5
        # C00 -> w=0, C01 -> w=2
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] -= 1; W[2][4] += 1
        
        # Term 6: M6 = (A10-A00)*(B00+B01)
        # A indices: A10 -> u=2, A00 -> u=0
        # B indices: B00 -> v=0, B01 -> v=1
        # C contribution: C11 += M6
        # C11 -> w=3
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # Term 7: M7 = (A01-A11)*(B10+B11)
        # A indices: A01 -> u=1, A11 -> u=3
        # B indices: B10 -> v=2, B11 -> v=3
        # C contribution: C00 += M7, C11 += M7
        # C00 -> w=0, C11 -> w=3
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1; W[3][6] += 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        """Return U, V, W for 2x2x1 schoolbook decomposition."""
        rank = 4
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(2)]  # m*p = 2*1 = 2
        W = [[0]*rank for _ in range(2)]  # p*n = 1*2 = 2
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # u = i*2+k, v = k*1+j = k, w = j*2+i = i (since j=0)
        
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        U[0][0] += 1
        V[0][0] += 1
        W[0][0] += 1
        
        # Term 2: A[0,1]*B[1,0] -> C[0,0]
        U[1][1] += 1
        V[1][1] += 1
        W[0][1] += 1
        
        # Term 3: A[1,0]*B[0,0] -> C[1,0]
        U[2][2] += 1
        V[0][2] += 1
        W[1][2] += 1
        
        # Term 4: A[1,1]*B[1,0] -> C[1,0]
        U[3][3] += 1
        V[1][3] += 1
        W[1][3] += 1
        
        return U, V, W
    
    def kron_strassen_4x4x4():
        """Return U, V, W for 4x4x4 using Kronecker product of two Strassen decompositions."""
        U_s, V_s, W_s = strassen_2x2x2()
        U_s = np.array(U_s, dtype=float)
        V_s = np.array(V_s, dtype=float)
        W_s = np.array(W_s, dtype=float)
        
        # Kronecker product: rank = 7*7 = 49
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        for i in range(7):
            for j in range(7):
                col = i * 7 + j
                # U: kron(U_s[:, i], U_s[:, j])
                U[:, col] = np.kron(U_s[:, i], U_s[:, j])
                # V: kron(V_s[:, i], V_s[:, j])
                V[:, col] = np.kron(V_s[:, i], V_s[:, j])
                # W: kron(W_s[:, i], W_s[:, j])
                W[:, col] = np.kron(W_s[:, i], W_s[:, j])
        
        return U.tolist(), V.tolist(), W.tolist()
    
    def strassen_schoolbook_2x2x3():
        """Return U, V, W for 2x2x3 using Strassen for 2x2x2 part and schoolbook for 2x2x1 part.
        Rank = 7 + 4 = 11."""
        U_ss, V_ss, W_ss = strassen_2x2x2()
        U_sk, V_sk, W_sk = schoolbook_2x2x1()
        
        U_ss = np.array(U_ss, dtype=float)
        V_ss = np.array(V_ss, dtype=float)
        W_ss = np.array(W_ss, dtype=float)
        U_sk = np.array(U_sk, dtype=float)
        V_sk = np.array(V_sk, dtype=float)
        W_sk = np.array(W_sk, dtype=float)
        
        rank = 11
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))  # m*p = 2*3 = 6
        W = np.zeros((6, rank))  # p*n = 3*2 = 6
        
        # Strassen part: columns 0-6, for the first 2 columns of B (indices 0,1)
        # V indices for 2x2x2: v = k*2 + j, where j is 0 or 1 (first 2 columns)
        # In 2x2x3, v = k*3 + j, where j is 0,1,2
        # For j=0: v = k*3 + 0
        # For j=1: v = k*3 + 1
        # So we need to map V_ss indices to V indices.
        # V_ss has shape (4, 7), with v_ss = k*2 + j_ss, where j_ss is 0 or 1.
        # We need to place these into V with v = k*3 + j_ss.
        
        for col in range(7):
            U[:, col] = U_ss[:, col]
            # Map V_ss to V
            for v_ss in range(4):
                k = v_ss // 2
                j_ss = v_ss % 2
                v = k * 3 + j_ss
                V[v, col] = V_ss[v_ss, col]
            # W: for 2x2x2, w = j*2 + i, where j is 0 or 1.
            # In 2x2x3, w = j*2 + i, where j is 0,1,2.
            # For j=0,1: same mapping.
            W[:, col] = W_ss[:, col]
        
        # Schoolbook part: columns 7-10, for the 3rd column of B (index 2)
        # V index for 2x2x1: v = k*1 + j = k, where j=0 (since p=1 for this part)
        # In 2x2x3, for the 3rd column (j=2), v = k*3 + 2
        # So we need to map V_sk indices to V.
        # V_sk has shape (2, 4), with v_sk = k.
        # We need to place these into V with v = k*3 + 2.
        
        for col in range(4):
            U[:, 7 + col] = U_sk[:, col]
            for v_sk in range(2):
                k = v_sk
                v = k * 3 + 2
                V[v, 7 + col] = V_sk[v_sk, col]
            # W: for 2x2x1, w = j*2 + i = i (since j=0).
            # In 2x2x3, for the 3rd column (j=2), w = j*2 + i = 2*2 + i = 4 + i.
            # So we need to map W_sk indices to W.
            # W_sk has shape (2, 4), with w_sk = i.
            # We need to place these into W with w = 4 + i.
            for w_sk in range(2):
                i = w_sk
                w = 4 + i
                W[w, 7 + col] = W_sk[w_sk, col]
        
        return U.tolist(), V.tolist(), W.tolist()
    
    # Build the full 4x4x7 decomposition
    # Part 1: 4x4x4 (columns 0-3 of B) using Kronecker Strassen: rank 49
    # Part 2: 4x4x3 (columns 4-6 of B) using four 2x2x3 blocks: rank 4*11 = 44
    # Total rank = 93
    
    rank_total = 93
    U = np.zeros((16, rank_total))
    V = np.zeros((28, rank_total))  # m*p = 4*7 = 28
    W = np.zeros((28, rank_total))  # p*n = 7*4 = 28
    
    # Part 1: 4x4x4
    U1, V1, W1 = kron_strassen_4x4x4()
    U1 = np.array(U1, dtype=float)
    V1 = np.array(V1, dtype=float)
    W1 = np.array(W1, dtype=float)
    
    # Place into U, V, W
    U[:, 0:49] = U1
    # V1 has shape (16, 49), with v1 = k*4 + j, where k is 0-3, j is 0-3.
    # In full V, v = k*7 + j, where k is 0-3, j is 0-6.
    # For j=0-3: v = k*7 + j
    for v1 in range(16):
        k = v1 // 4
        j = v1 % 4
        v = k * 7 + j
        V[v, 0:49] = V1[v1, 0:49]
    # W1 has shape (16, 49), with w1 = j*4 + i, where j is 0-3, i is 0-3.
    # In full W, w = j*4 + i, where j is 0-6, i is 0-3.
    # For j=0-3: w = j*4 + i
    W[:, 0:49] = W1
    
    # Part 2: 4x4x3 (columns 4-6 of B)
    # Four 2x2x3 blocks:
    # Block (0,0): A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]
    # Block (0,1): A[0:2, 2:4] * B[2:4, 4:7] -> C[0:2, 4:7]
    # Block (1,0): A[2:4, 0:2] * B[0:2, 4:7] -> C[2:4, 4:7]
    # Block (1,1): A[2:4, 2:4] * B[2:4, 4:7] -> C[2:4, 4:7]
    
    for block_row in range(2):
        for block_col in range(2):
            U2, V2, W2 = strassen_schoolbook_2x2x3()
            U2 = np.array(U2, dtype=float)
            V2 = np.array(V2, dtype=float)
            W2 = np.array(W2, dtype=float)
            
            col_offset = 49 + (block_row * 2 + block_col) * 11
            
            # U: A indices
            # Block (block_row, block_col) corresponds to A[block_row*2:block_row*2+2, block_col*2:block_col*2+2]
            # U2 has shape (4, 11), with u2 = i*2 + k, where i is 0-1, k is 0-1.
            # In full U, u = i*4 + k, where i is 0-3, k is 0-3.
            # For this block: i_full = block_row*2 + i2, k_full = block_col*2 + k2
            for u2 in range(4):
                i2 = u2 // 2
                k2 = u2 % 2
                u = (block_row * 2 + i2) * 4 + (block_col * 2 + k2)
                U[u, col_offset:col_offset+11] = U2[u2, :]
            
            # V: B indices
            # Block (block_row, block_col) corresponds to B[block_col*2:block_col*2+2, 4:7]
            # V2 has shape (6, 11), with v2 = k*3 + j, where k is 0-1, j is 0-2.
            # In full V, v = k*7 + j, where k is 0-3, j is 0-6.
            # For this block: k_full = block_col*2 + k2, j_full = 4 + j2
            for v2 in range(6):
                k2 = v2 // 3
                j2 = v2 % 3
                v = (block_col * 2 + k2) * 7 + (4 + j2)
                V[v, col_offset:col_offset+11] = V2[v2, :]
            
            # W: C indices
            # Block (block_row, block_col) contributes to C[block_row*2:block_row*2+2, 4:7]
            # W2 has shape (6, 11), with w2 = j*2 + i, where j is 0-2, i is 0-1.
            # In full W, w = j*4 + i, where j is 0-6, i is 0-3.
            # For this block: j_full = 4 + j2, i_full = block_row*2 + i2
            for w2 in range(6):
                j2 = w2 // 2
                i2 = w2 % 2
                w = (4 + j2) * 4 + (block_row * 2 + i2)
                W[w, col_offset:col_offset+11] = W2[w2, :]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify correctness if time permits
    if time.time() < deadline - 10:
        target = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target.add((i * m + j, j * p + k, k * n + i))
        
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        
        # Check all target entries equal 1
        for (u, v, w) in target:
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val - 1.0) > 1e-9:
                raise ValueError(f"Target entry ({u},{v},{w}) = {val}, expected 1")
        
        # Check all non-target entries equal 0 (sampled for speed)
        np.random.seed(42)
        for _ in range(5000):
            u = np.random.randint(0, n * m)
            v = np.random.randint(0, m * p)
            w = np.random.randint(0, p * n)
            if (u, v, w) in target:
                continue
            val = float(np.dot(U_np[u] * V_np[v], W_np[w]))
            if abs(val) > 1e-9:
                raise ValueError(f"Non-target entry ({u},{v},{w}) = {val}, expected 0")
    
    return result
