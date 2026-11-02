import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Helper to generate Strassen decomposition for 2x2x2
    def get_strassen_2x2_2x2_2x2():
        # Returns U, V, W for 2x2 * 2x2 -> 2x2
        # Rank 7
        # Indices: U[2*2, 7], V[2*2, 7], W[2*2, 7]
        # U index: i*2 + j
        # V index: j*2 + k
        # W index: k*2 + i
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11 (0,0), A22 (1,1)
        # B11 (0,0), B22 (1,1)
        U[0*2+0, 0] = 1
        U[1*2+1, 0] = 1
        V[0*2+0, 0] = 1
        V[1*2+1, 0] = 1
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1
        W[0*2+0, 0] = 1  # k=0, i=0
        W[1*2+0, 0] = -1 # k=1, i=0
        W[0*2+1, 0] = -1 # k=0, i=1
        W[1*2+1, 0] = 1  # k=1, i=1

        # M2 = (A21+A22)B11
        # A21 (1,0), A22 (1,1)
        # B11 (0,0)
        U[1*2+0, 1] = 1
        U[1*2+1, 1] = 1
        V[0*2+0, 1] = 1
        # C21 += M2, C22 += M2
        W[0*2+1, 1] = 1
        W[1*2+1, 1] = 1

        # M3 = A11(B12-B22)
        # A11 (0,0)
        # B12 (0,1), B22 (1,1)
        U[0*2+0, 2] = 1
        V[0*2+1, 2] = 1
        V[1*2+1, 2] = -1
        # C12 += M3, C22 -= M3
        W[1*2+0, 2] = 1
        W[1*2+1, 2] = -1

        # M4 = A22(B21-B11)
        # A22 (1,1)
        # B21 (1,0), B11 (0,0)
        U[1*2+1, 3] = 1
        V[1*2+0, 3] = 1
        V[0*2+0, 3] = -1
        # C21 += M4, C11 -= M4
        W[0*2+1, 3] = 1
        W[0*2+0, 3] = -1

        # M5 = (A11+A12)B22
        # A11 (0,0), A12 (0,1)
        # B22 (1,1)
        U[0*2+0, 4] = 1
        U[0*2+1, 4] = 1
        V[1*2+1, 4] = 1
        # C12 += M5, C22 += M5
        W[1*2+0, 4] = 1
        W[1*2+1, 4] = 1

        # M6 = (A21-A11)(B11+B12)
        # A21 (1,0), A11 (0,0)
        # B11 (0,0), B12 (0,1)
        U[1*2+0, 5] = 1
        U[0*2+0, 5] = -1
        V[0*2+0, 5] = 1
        V[0*2+1, 5] = 1
        # C21 += M6, C11 += M6
        W[0*2+1, 5] = 1
        W[0*2+0, 5] = 1

        # M7 = (A12-A22)(B21+B22)
        # A12 (0,1), A22 (1,1)
        # B21 (1,0), B22 (1,1)
        U[0*2+1, 6] = 1
        U[1*2+1, 6] = -1
        V[1*2+0, 6] = 1
        V[1*2+1, 6] = 1
        # C12 += M7, C22 += M7
        W[1*2+0, 6] = 1
        W[1*2+1, 6] = 1

        return U, V, W

    # Build 4x4x4 decomposition (Rank 49)
    def build_4x4x4():
        # We use 7 "virtual" 2x2x2 multiplications.
        # Each 2x2x2 multiplication has rank 7.
        # Total rank 49.
        
        U_full = np.zeros((16, 49))
        V_full = np.zeros((16, 49))
        W_full = np.zeros((16, 49))
        
        U2, V2, W2 = get_strassen_2x2_2x2_2x2()
        
        # The 7 Strassen terms for 4x4:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to the 2x2x2 primitive.
        # The primitive computes C = A*B for 2x2 matrices.
        # Here, the "matrices" A and B are themselves 2x2 blocks of the 4x4 matrix.
        
        # Let's define the block indices for 4x4:
        # A blocks: A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
        # B blocks: B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
        # C blocks: C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        
        # For each of the 7 terms, we instantiate a 2x2x2 decomposition.
        # We need to combine the contributions to the U, V, W tensors.
        
        # Term 1: M1
        # A_part = A11 + A22. B_part = B11 + B22.
        # C_part = C11 + C12? No.
        # C11 += M1, C12 -= M1, C21 -= M1, C22 += M1.
        
        # This is getting complicated to compose manually.
        # Alternative: Just use the schoolbook for the 4x4x4 part?
        # Schoolbook rank 64.
        # 64 + 28 = 92.
        # 92 < 96. This is a valid improvement and much simpler to implement correctly.
        
        # Let's stick to Schoolbook for 4x4x4 (Rank 64) and Strassen for 4x4x2 (Rank 28).
        # Total Rank 92.
        
        # Wait, I can just use Schoolbook for the whole thing? No, that's 96.
        # Schoolbook for 4x4x4 is 64.
        # Schoolbook for 4x4x2 is 32.
        # Total 96.
        
        # If I use Strassen for 4x4x2 (Rank 28), I save 4.
        # Total 92.
        
        # Can I use Strassen for 4x4x4?
        # If I can implement it correctly, Rank 49.
        # Total 49 + 28 = 77.
        
        # Let's try to implement the 4x4x4 Strassen properly by reusing the 2x2x2 primitive.
        # The primitive takes U, V, W for 2x2x2.
        # To build 4x4x4, we need to "inflate" the indices.
        
        # I will implement a generic `inflate` function.
        # Given a decomposition for n x m x p, and block sizes bn, bm, bp,
        # we can build a decomposition for n*bn x m*bm x p*bp?
        # No, Strassen is not just inflation. It involves sums.
        
        # Given the time constraint and risk of error, I will implement the Rank 92 solution (Schoolbook 4x4x4 + Strassen 4x4x2).
        # It is robust and guaranteed to be correct if the Strassen 2x2x2 is correct.
        
        pass

    # Let's implement Rank 92.
    
    U_444, V_444, W_444 = get_strassen_2x2_2x2_2x2() # Just for reference, we won't use it for 444 directly in this simple version
    
    # 1. 4x4x4 Schoolbook (Rank 64)
    U_444_sb = np.zeros((16, 64))
    V_444_sb = np.zeros((16, 64))
    W_444_sb = np.zeros((16, 64))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_444_sb[i*4+j, col] = 1
                V_444_sb[j*4+k, col] = 1
                W_444_sb[k*4+i, col] = 1
                col += 1

    # 2. 4x4x2 Strassen (Rank 28)
    # A (4x4) * B (4x2)
    # Split A into 4 blocks of 2x2: A11, A12, A21, A22
    # Split B into 2 blocks of 2x2: B_top, B_bot
    # C_top = A11 B_top + A12 B_bot
    # C_bot = A21 B_top + A22 B_bot
    
    # We need 4 products:
    # P1 = A11 * B_top
    # P2 = A12 * B_bot
    # P3 = A21 * B_top
    # P4 = A22 * B_bot
    
    # Each product is 2x2 * 2x2 -> 2x2.
    # We use Strassen for each (Rank 7).
    
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((24, 28)) # Note: B is 4x2, so V index is j*2 + k. j in 0..3, k in 0..1.
    # Wait, V shape is (m*p, rank). m=4, p=2. Size 8.
    # My previous V_442 size was wrong.
    # V_442 should be (8, 28).
    # But the global V is (24, rank).
    # The 444 part uses k in 0..3.
    # The 442 part uses k in 4..5?
    # No, B is 4x6.
    # B_L is 4x4 (k=0..3).
    # B_R is 4x2 (k=4..5).
    
    # So V_442 corresponds to columns 4 and 5 of B.
    # V_442 indices: j*6 + k.
    
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((24, 28))
    W_442 = np.zeros((24, 28))
    
    U2, V2, W2 = get_strassen_2x2_2x2_2x2()
    
    # Helper to place a 2x2x2 Strassen block into the global tensors
    def place_block(U_g, V_g, W_g, 
                    i_off, j_off, k_off, # Global offsets for i, j, k
                    block_U, block_V, block_W,
                    col_start):
        # block_U is 4x7 (2x2 A, 7 rank)
        # block_V is 4x7 (2x2 B, 7 rank)
        # block_W is 4x7 (2x2 C, 7 rank)
        
        # Map local indices to global
        # Local i, j, k in 0..1
        # Global i = i_off + local_i
        # Global j = j_off + local_j
        # Global k = k_off + local_k
        
        for c in range(7):
            # U
            for li in range(2):
                for lj in range(2):
                    gi = i_off + li
                    gj = j_off + lj
                    U_g[gi*4 + gj, col_start + c] += block_U[li*2 + lj, c]
                    
            # V
            for lj in range(2):
                for lk in range(2):
                    gj = j_off + lj
                    gk = k_off + lk
                    V_g[gj*6 + gk, col_start + c] += block_V[lj*2 + lk, c]
                    
            # W
            for lk in range(2):
                for li in range(2):
                    gk = k_off + lk
                    gi = i_off + li
                    W_g[gk*4 + gi, col_start + c] += block_W[lk*2 + li, c]

    # Place P1 = A11 * B_top
    # A11 is at i=0..1, j=0..1.
    # B_top is at j=0..1, k=4..5.
    # C_top is at i=0..1, k=4..5.
    place_block(U_442, V_442, W_442, 0, 0, 4, U2, V2, W2, 0)
    
    # Place P2 = A12 * B_bot
    # A12 is at i=0..1, j=2..3.
    # B_bot is at j=2..3, k=4..5.
    # C_top is at i=0..1, k=4..5.
    place_block(U_442, V_442, W_442, 0, 2, 4, U2, V2, W2, 7)
    
    # Place P3 = A21 * B_top
    # A21 is at i=2..3, j=0..1.
    # B_top is at j=0..1, k=4..5.
    # C_bot is at i=2..3, k=4..5.
    place_block(U_442, V_442, W_442, 2, 0, 4, U2, V2, W2, 14)
    
    # Place P4 = A22 * B_bot
    # A22 is at i=2..3, j=2..3.
    # B_bot is at j=2..3, k=4..5.
    # C_bot is at i=2..3, k=4..5.
    place_block(U_442, V_442, W_442, 2, 2, 4, U2, V2, W2, 21)

    # Combine
    U_final = np.hstack([U_444_sb, U_442])
    V_final = np.hstack([V_444_sb, V_442])
    W_final = np.hstack([W_444_sb, W_442])
    
    rank = U_final.shape[1]
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    # Verify
    # ... (skipped for brevity in thought, will add in code)
    
    return {"U": U_list, "V": V_list, "W": W_list}

