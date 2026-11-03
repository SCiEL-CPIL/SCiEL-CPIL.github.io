import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 125

    def verify(U, V, W, rank):
        """Verify decomposition. Returns max error."""
        max_err = 0.0
        # Vectorized verification for speed
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need to check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        
        # Reshape to 3D tensors for easier indexing? 
        # Or just loop. n=5 is small.
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def construct_strassen_2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[0,2]=1; W[1,2]=1
        U[2,3]=1; U[3,3]=1; U[1,3]=-1; W[2,3]=1; W[3,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[2,5]=1; W[2,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[1,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
        return U, V, W

    def construct_laderman_3x3():
        U = np.zeros((9, 15)); V = np.zeros((9, 15)); W = np.zeros((9, 15))
        U[0,0]=1; U[4,0]=1; U[8,0]=1; V[0,0]=1; V[4,0]=1; V[8,0]=1
        W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=-1; W[4,0]=-1; W[5,0]=1; W[6,0]=1; W[7,0]=1; W[8,0]=-1
        U[1,1]=1; U[5,1]=1; V[0,1]=1; V[8,1]=1; W[1,1]=1; W[2,1]=1; W[4,1]=1; W[5,1]=1
        U[2,2]=1; U[6,2]=1; V[3,2]=1; V[8,2]=1; W[3,2]=1; W[4,2]=1; W[6,2]=1; W[7,2]=1
        U[0,3]=1; U[8,3]=1; V[0,3]=1; V[4,3]=1; W[0,3]=1; W[1,3]=1; W[3,3]=1; W[4,3]=1
        U[1,4]=1; U[5,4]=1; U[8,4]=1; V[3,4]=1; V[8,4]=1
        W[3,4]=1; W[4,4]=1; W[5,4]=1; W[6,4]=1; W[7,4]=1; W[8,4]=1
        U[2,5]=1; U[6,5]=1; V[0,5]=1; V[4,5]=1; W[0,5]=1; W[1,5]=1; W[3,5]=1; W[4,5]=1
        U[0,6]=1; U[4,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=1; W[1,6]=1; W[3,6]=1; W[4,6]=1
        U[0,7]=1; U[4,7]=1; V[0,7]=1; V[8,7]=1; W[0,7]=1; W[1,7]=1; W[5,7]=1; W[6,7]=1
        U[1,8]=1; U[5,8]=1; V[3,8]=1; V[8,8]=1; W[3,8]=1; W[4,8]=1; W[6,8]=1; W[7,8]=1
        U[0,9]=1; U[8,9]=1; V[0,9]=1; V[8,9]=1; W[0,9]=1; W[1,9]=1; W[5,9]=1; W[6,9]=1
        U[2,10]=1; U[6,10]=1; V[0,10]=1; V[4,10]=1; V[8,10]=1
        W[0,10]=1; W[1,10]=1; W[3,10]=1; W[4,10]=1; W[5,10]=1; W[6,10]=1
        U[0,11]=1; U[4,11]=1; U[8,11]=1; V[3,11]=1; V[8,11]=1
        W[3,11]=1; W[4,11]=1; W[5,11]=1; W[6,11]=1; W[7,11]=1; W[8,11]=1
        U[1,12]=1; U[5,12]=1; U[8,12]=1; V[0,12]=1; V[8,12]=1; W[0,12]=1; W[1,12]=1; W[5,12]=1; W[6,12]=1
        U[0,13]=1; U[4,13]=1; V[3,13]=1; V[8,13]=1; W[3,13]=1; W[4,13]=1; W[6,13]=1; W[7,13]=1
        U[2,14]=1; U[6,14]=1; V[3,14]=1; V[8,14]=1; W[3,14]=1; W[4,14]=1; W[6,14]=1; W[7,14]=1
        return U, V, W

    def build_block_decomposition():
        """
        Constructs the 5x5 decomposition using 2x2 Strassen and 3x3 Laderman blocks.
        Rank = 7 (TL) + 12 (TR) + 18 (ML) + 12 (BL) + 18 (BR cross) + 15 (BR Laderman) = 82
        """
        laderman_result = construct_laderman_3x3()
        use_laderman = laderman_result is not None
        laderman_rank = 15 if use_laderman else 27
        total_rank = 7 + 12 + 18 + 12 + 18 + laderman_rank
        
        U = np.zeros((25, total_rank)); V = np.zeros((25, total_rank)); W = np.zeros((25, total_rank))
        col_offset = 0
        
        # Top-Left 2x2 block: uses Strassen (rank 7)
        U_a, V_a, W_a = construct_strassen_2x2()
        # Indices for TL block in 5x5: i in 0..1, j in 0..1, k in 0..1
        # U row: i*5+j, V row: j*5+k, W row: k*5+i
        for idx_u, idx_v, idx_w in [(0,0,0), (1,1,0), (2,2,0), (3,3,0)]: # Wait, Strassen indices are (i,j) for U, (j,k) for V, (k,i) for W
            # Let's map carefully.
            # Strassen U is 4x7. Rows correspond to (i,j) in 2x2.
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # V is 4x7. Rows correspond to (j,k) in 2x2.
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # W is 4x7. Rows correspond to (k,i) in 2x2.
            # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            
            pass # We'll do it by explicit mapping to be safe
        
        # TL Block (2x2)
        # U rows: 0,1,5,6 (i=0,1; j=0,1) -> 0,1,5,6
        # V rows: 0,1,5,6 (j=0,1; k=0,1) -> 0,1,5,6
        # W rows: 0,1,5,6 (k=0,1; i=0,1) -> 0,1,5,6
        
        # Strassen U row 0 (i=0,j=0) -> 5x5 U row 0
        # Strassen U row 1 (i=0,j=1) -> 5x5 U row 1
        # Strassen U row 2 (i=1,j=0) -> 5x5 U row 5
        # Strassen U row 3 (i=1,j=1) -> 5x5 U row 6
        
        u_map = [0, 1, 5, 6]
        v_map = [0, 1, 5, 6]
        w_map = [0, 1, 5, 6]
        
        for s_idx in range(4):
            U[u_map[s_idx], col_offset:col_offset+7] = U_a[s_idx]
            V[v_map[s_idx], col_offset:col_offset+7] = V_a[s_idx]
            W[w_map[s_idx], col_offset:col_offset+7] = W_a[s_idx]
        col_offset += 7

        # Top-Right Block (2x3): i in 0..1, j in 0..1, k in 2..4
        # Rank 2*3*3 = 18? No, simple tensor product.
        # We can just use schoolbook for these mixed blocks to keep it simple and correct.
        # Or use Strassen if we split 3 into 2+1? No, let's stick to the previous logic which was:
        # TR: i(2), j(2), k(3) -> 12 terms? 
        # In the previous code:
        # for i_prime in range(2):
        #     for j_prime in range(2):
        #         for k_prime in range(3):
        #             ...
        #             t = col_offset + i_prime*6 + j_prime*3 + k_prime
        #             U[i_prime*5+j_prime, t]=1
        #             V[j_prime*5+(2+k_prime), t]=1
        #             W[(2+k_prime)*5+i_prime, t]=1
        # This is rank 12.
        
        for i_prime in range(2):
            for j_prime in range(2):
                for k_prime in range(3):
                    t = col_offset + i_prime*6 + j_prime*3 + k_prime
                    U[i_prime*5+j_prime, t]=1
                    V[j_prime*5+(2+k_prime), t]=1
                    W[(2+k_prime)*5+i_prime, t]=1
        col_offset += 12

        # Middle-Left Block (3x2): i in 2..4, j in 0..1, k in 0..1
        # Previous code:
        # for i_prime in range(3):
        #     for j_prime in range(2):
        #         for k_prime in range(2):
        #             ...
        #             t = col_offset + i_prime*4 + j_prime*2 + k_prime
        #             U[(2+i_prime)*5+j_prime, t]=1
        #             V[j_prime*5+k_prime, t]=1
        #             W[k_prime*5+(2+i_prime), t]=1
        # Rank 12.
        
        for i_prime in range(3):
            for j_prime in range(2):
                for k_prime in range(2):
                    t = col_offset + i_prime*4 + j_prime*2 + k_prime
                    U[(2+i_prime)*5+j_prime, t]=1
                    V[j_prime*5+k_prime, t]=1
                    W[k_prime*5+(2+i_prime), t]=1
        col_offset += 12

        # Bottom-Left Block (3x3) but k in 0..1? No, i in 2..4, j in 2..4, k in 0..1?
        # Wait, the blocks cover the 5x5 grid.
        # TL: i 0-1, j 0-1, k 0-1 (2x2x2)
        # TR: i 0-1, j 0-1, k 2-4 (2x2x3)
        # ML: i 2-4, j 0-1, k 0-1 (3x2x2)
        # MR: i 2-4, j 0-1, k 2-4 (3x2x3)
        # BL: i 2-4, j 2-4, k 0-1 (3x3x2)
        # BR: i 2-4, j 2-4, k 2-4 (3x3x3)
        
        # The previous code had:
        # 3. ML (3x2x2) rank 12
        # 4. MR (3x2x3) rank 18
        # 5. BL (3x3x2) rank 18
        # 6. BR (3x3x3) rank 15 (Laderman)
        
        # Let's verify the previous code's blocks.
        # Block 3 in code:
        # for i_prime in range(3):
        #     for j_prime in range(2):
        #         for k_prime in range(3):
        #             U[(2+i_prime)*5+(2+j_prime), t]=1  # i in 2-4, j in 2-4 ??
        #             V[(2+j_prime)*5+(2+k_prime), t]=1  # j in 2-4, k in 2-4 ??
        #             W[(2+k_prime)*5+i_prime, t]=1      # k in 2-4, i in 0-2 ??
        # This looks like it's covering a 3x3x3 region but shifted?
        # Let's re-read the previous code carefully.
        
        # Previous Code Block 3:
        # for i_prime in range(3):
        #     for j_prime in range(3):
        #         for k_prime in range(3):
        #             t = col_offset + i_prime*9 + j_prime*3 + k_prime
        #             U[i_prime*5+(2+j_prime), t]=1
        #             V[(2+j_prime)*5+(2+k_prime), t]=1
        #             W[(2+k_prime)*5+i_prime, t]=1
        # This is i in 0-2, j in 2-4, k in 2-4.
        # This corresponds to i=0,1,2 (first 3 rows), j=2,3,4 (last 3 cols), k=2,3,4 (last 3 cols).
        # This is NOT a standard block partition. It's an overlapping or shifted partition?
        # Actually, matrix multiplication tensor T[i,j,k] = 1 if i,j,k are indices.
        # The decomposition is a sum of rank-1 tensors.
        # The "Block Decomposition" usually implies partitioning the indices.
        # If we partition i={0,1}, {2,3,4}; j={0,1}, {2,3,4}; k={0,1}, {2,3,4}.
        # We have 8 blocks.
        # 1. i01, j01, k01 (2x2x2) -> Strassen 7
        # 2. i01, j01, k234 (2x2x3) -> 12
        # 3. i01, j234, k01 (2x3x2) -> 12
        # 4. i01, j234, k234 (2x3x3) -> 18
        # 5. i234, j01, k01 (3x2x2) -> 12
        # 6. i234, j01, k234 (3x2x3) -> 18
        # 7. i234, j234, k01 (3x3x2) -> 18
        # 8. i234, j234, k234 (3x3x3) -> Laderman 15
        
        # Total Rank = 7+12+12+18+12+18+18+15 = 112.
        # The previous code claimed 82. How?
        # It seems the previous code did NOT use a full 8-block partition.
        # Let's look at the previous code's loops again.
        
        # Block 1: TL (2x2x2) Strassen. (7)
        # Block 2: 
        # for i_prime in range(2): # i in 0,1
        #     for j_prime in range(2): # j in 0,1
        #         for k_prime in range(3): # k in 0,1,2 (mapped to 2,3,4)
        #             U[i_prime*5+j_prime, t]=1
        #             V[j_prime*5+(2+k_prime), t]=1
        #             W[(2+k_prime)*5+i_prime, t]=1
        # This is i01, j01, k234. Rank 12.
        
        # Block 3:
        # for i_prime in range(3): # i in 0,1,2
        #     for j_prime in range(3): # j in 0,1,2
        #         for k_prime in range(3): # k in 0,1,2
        #             U[i_prime*5+(2+j_prime), t]=1
        #             V[(2+j_prime)*5+(2+k_prime), t]=1
        #             W[(2+k_prime)*5+i_prime, t]=1
        # This is i012, j234, k234.
        # This covers i=0,1,2; j=2,3,4; k=2,3,4.
        # This overlaps with Block 1 (i01, j01, k01)? No, j and k are different.
        # It overlaps with Block 2? Block 2 is i01, j01, k234.
        # Block 3 is i012, j234, k234.
        # So Block 3 covers the "Right" part of the matrix, but includes i=2.
        
        # Block 4:
        # for i_prime in range(3): # i in 0,1,2
        #     for j_prime in range(2): # j in 0,1
        #         for k_prime in range(2): # k in 0,1
        #             U[(2+i_prime)*5+j_prime, t]=1
        #             V[j_prime*5+k_prime, t]=1
        #             W[k_prime*5+(2+i_prime), t]=1
        # This is i234, j01, k01. Rank 12.
        
        # Block 5:
        # for i_prime in range(3): # i in 0,1,2
        #     for j_prime in range(3): # j in 0,1,2
        #         for k_prime in range(2): # k in 0,1
        #             U[(2+i_prime)*5+(2+j_prime), t]=1
        #             V[(2+j_prime)*5+k_prime, t]=1
        #             W[k_prime*5+(2+i_prime), t]=1
        # This is i234, j234, k01. Rank 18.
        
        # Block 6:
        # Laderman 3x3.
        # U rows (2+i_prime)*5+(2+j_prime) -> i234, j234
        # V rows (2+j_prime)*5+(2+k_prime) -> j234, k234
        # W rows (2+k_prime)*5+(2+i_prime) -> k234, i234
        # This is i234, j234, k234. Rank 15.
        
        # Let's check coverage.
        # i=0,1,2,3,4
        # j=0,1,2,3,4
        # k=0,1,2,3,4
        
        # Block 1: i01, j01, k01
        # Block 2: i01, j01, k234
        # Block 3: i012, j234, k234
        # Block 4: i234, j01, k01
        # Block 5: i234, j234, k01
        # Block 6: i234, j234, k234
        
        # Missing?
        # i=2 is in Block 3, 4, 5, 6.
        # i=0,1 are in Block 1, 2, 3.
        # j=0,1 are in Block 1, 2, 4.
        # j=2,3,4 are in Block 3, 5, 6.
        # k=0,1 are in Block 1, 4, 5.
        # k=2,3,4 are in Block 2, 3, 6.
        
        # Is it a partition?
        # (i,j,k)
        # 0,0,0 -> B1
        # 0,0,2 -> B2
        # 0,2,2 -> B3
        # 2,0,0 -> B4
        # 2,2,0 -> B5
        # 2,2,2 -> B6
        
        # What about (0, 2, 0)?
        # i=0 (in B1, B2, B3). j=2 (in B3, B5, B6). k=0 (in B1, B4, B5).
        # Intersection:
        # B1: i01, j01, k01. j=2 not in B1.
        # B2: i01, j01, k234. j=2 not in B2.
        # B3: i012, j234, k234. k=0 not in B3.
        # B4: i234, j01, k01. i=0 not in B4.
        # B5: i234, j234, k01. i=0 not in B5.
        # B6: i234, j234, k234. i=0 not in B6.
        # (0,2,0) is NOT covered!
        
        # The previous code's block decomposition is INVALID. It misses terms.
        # That's why the score was 1.0 (fallback to schoolbook or invalid?).
        # The verify function would have caught it.
        # "if max_err < 1e-8: ... else fallback"
        # So the current program likely returns the schoolbook rank 125 because the block decomposition failed verification.
        
        # We MUST fix the block decomposition to be a valid partition.
        
    # Correct Block Decomposition
    def build_correct_block_decomposition():
        # Partition:
        # I1 = {0,1}, I2 = {2,3,4}
        # J1 = {0,1}, J2 = {2,3,4}
        # K1 = {0,1}, K2 = {2,3,4}
        
        # Blocks:
        # 1. I1 J1 K1 (2x2x2) -> Strassen (7)
        # 2. I1 J1 K2 (2x2x3) -> Schoolbook (12)
        # 3. I1 J2 K1 (2x3x2) -> Schoolbook (12)
        # 4. I1 J2 K2 (2x3x3) -> Schoolbook (18)
        # 5. I2 J1 K1 (3x2x2) -> Schoolbook (12)
        # 6. I2 J1 K2 (3x2x3) -> Schoolbook (18)
        # 7. I2 J2 K1 (3x3x2) -> Schoolbook (18)
        # 8. I2 J2 K2 (3x3x3) -> Laderman (15)
        
        # Total Rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 15 = 112.
        # This is worse than 82? No, 82 was from the broken code.
        # 112 is still better than 125.
        
        U = np.zeros((25, 112)); V = np.zeros((25, 112)); W = np.zeros((25, 112))
        col = 0
        
        # 1. Strassen 2x2
        U_a, V_a, W_a = construct_strassen_2x2()
        u_map = [0, 1, 5, 6] # i0j0, i0j1, i1j0, i1j1
        v_map = [0, 1, 5, 6]
        w_map = [0, 1, 5, 6]
        for s in range(4):
            U[u_map[s], col:col+7] = U_a[s]
            V[v_map[s], col:col+7] = V_a[s]
            W[w_map[s], col:col+7] = W_a[s]
        col += 7
        
        # 2. I1 J1 K2 (2x2x3)
        for i in [0,1]:
            for j in [0,1]:
                for k in [2,3,4]:
                    U[i*5+j, col]=1
                    V[j*5+k, col]=1
                    W[k*5+i, col]=1
                    col += 1
        # col += 12
        
        # 3. I1 J2 K1 (2x3x2)
        for i in [0,1]:
            for j in [2,3,4]:
                for k in [0,1]:
                    U[i*5+j, col]=1
                    V[j*5+k, col]=1
                    W[k*5+i, col]=1
                    col += 1
        # col += 12
        
        # 4. I1 J2 K2 (2x3x3)
        for i in [0,1]:
            for j in [2,3,4]:
                for k in [2,3,4]:
                    U[i*5+j, col]=1
                    V[j*5+k, col]=1
                    W[k*5+i, col]=1
                    col += 1
        # col += 18
        
        # 5. I2 J1 K1 (3x2x2)
        for i in [2,3,4]:
            for j in [0,1]:
                for k in [0,1]:
                    U[i*5+j, col]=1
                    V[j*5+k, col]=1
                    W[k*5+i, col]=1
                    col += 1
        # col += 12
        
        # 6. I2 J1 K2 (3x2x3)
        for i in [2,3,4]:
            for j in [0,1]:
                for k in [2,3,4]:
                    U[i*5+j, col]=1
                    V[j*5+k, col]=1
                    W[k*5+i, col]=1
                    col += 1
        # col += 18
        
        # 7. I2 J2 K1 (3x3x2)
        for i in [2,3,4]:
            for j in [2,3,4]:
                for k in [0,1]:
                    U[i*5+j, col]=1
                    V[j*5+k, col]=1
                    W[k*5+i, col]=1
                    col += 1
        # col += 18
        
        # 8. I2 J2 K2 (3x3x3) Laderman
        U_l, V_l, W_l = construct_laderman_3x3()
        # Map 3x3 indices to 5x5 indices
        # i_l in 0..2 -> i in 2..4
        # j_l in 0..2 -> j in 2..4
        # k_l in 0..2 -> k in 2..4
        
        for t in range(15):
            for i_l in range(3):
                for j_l in range(3):
                    U[(2+i_l)*5+(2+j_l), col+t] = U_l[i_l*3+j_l, t]
            for j_l in range(3):
                for k_l in range(3):
                    V[(2+j_l)*5+(2+k_l), col+t] = V_l[j_l*3+k_l, t]
            for k_l in range(3):
                for i_l in range(3):
                    W[(2+k_l)*5+(2+i_l), col+t] = W_l[k_l*3+i_l, t]
        col += 15
        
        return U, V, W, col

    # 1. Build correct block decomposition
    U_block, V_block, W_block, block_rank = build_correct_block_decomposition()
    max_err = verify(U_block, V_block, W_block, block_rank)
    
    best_rank = 125
    best_result = None
    
    if max_err < 1e-8:
        best_rank = block_rank
        best_result = (U_block, V_block, W_block)
    else:
        # Fallback to schoolbook
        U_s = np.zeros((25, 125)); V_s = np.zeros((25, 125)); W_s = np.zeros((25, 125))
        c = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U_s[i*5+j, c]=1; V_s[j*5+k, c]=1; W_s[k*5+i, c]=1
                    c += 1
        best_rank = 125
        best_result = (U_s, V_s, W_s)

    # 2. Try to improve with AM search
    # We try ranks lower than best_rank
    # Given the complexity, we might only try a few ranks.
    
    def try_am_rank(rank, max_restarts=2, max_iters=20):
        if time.time() > deadline - 30:
            return None
            
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(rank * 1000 + restart * 7)
            U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(25, rank))
            V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(25, rank))
            W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(25, rank))
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                    
                # Update U
                for i in range(5):
                    for j in range(5):
                        B = np.array([[V[j*5+k, c] * W[k*5+i, c] for k in range(5)] for c in range(rank)])
                        try:
                            U[i*5+j, :] = np.linalg.lstsq(B, np.ones(5), rcond=None)[0]
                        except:
                            pass
                            
                # Update V
                for j in range(5):
                    for k in range(5):
                        A = np.array([[U[i*5+j, c] * W[k*5+i, c] for i in range(5)] for c in range(rank)])
                        try:
                            V[j*5+k, :] = np.linalg.lstsq(A, np.ones(5), rcond=None)[0]
                        except:
                            pass
                            
                # Update W
                for k in range(5):
                    for i in range(5):
                        C = np.array([[U[i*5+j, c] * V[j*5+k, c] for j in range(5)] for c in range(rank)])
                        try:
                            W[k*5+i, :] = np.linalg.lstsq(C, np.ones(5), rcond=None)[0]
                        except:
                            pass
                
                if it % 5 == 4:
                    max_err = verify(U, V, W, rank)
                    if max_err < 1e-6:
                        # Round
                        U_r = np.round(U * 2) / 2.0
                        V_r = np.round(V * 2) / 2.0
                        W_r = np.round(W * 2) / 2.0
                        err_r = verify(U_r, V_r, W_r, rank)
                        if err_r < 1e-8:
                            return (U_r, V_r, W_r)
        return None

    # Try to find rank < best_rank
    # Start from best_rank - 1
    # We have ~100s left.
    # Let's try a few ranks.
    for target_rank in range(best_rank - 1, max(60, best_rank - 10), -1):
        if time.time() > deadline - 20:
            break
            
        result = try_am_rank(target_rank)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = (U, V, W)
            # If we find one, we can stop or try lower.
            # Let's try lower.
            continue

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
