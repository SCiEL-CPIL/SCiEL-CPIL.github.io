"""Matrix multiplication 4,4,6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Recursive Strassen-like decomposition.
    - 4x4 * 4x6 is split into 7 sub-problems of type 2x2 * 2x3 using Strassen's 7 terms.
    - Each 2x2 * 2x3 is split into:
        * 2x2 * 2x1 (rank 4 schoolbook)
        * 2x2 * 2x2 (rank 7 Strassen)
    - Total rank = 7 * (4 + 7) = 77.
    
    This is a structural improvement over the schoolbook rank of 96.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    def build_2x2x1():
        """Rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1."""
        rank = 4
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows (i*2+j)
        V = [[0]*rank for _ in range(2)]  # 2*1 = 2 rows (j*1+k = j)
        W = [[0]*rank for _ in range(2)]  # 1*2 = 2 rows (k*2+i = i)
        
        # C[i] = sum_j A[i,j] B[j]
        # Term 0: A[0,0]*B[0] -> C[0]
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 1: A[0,1]*B[1] -> C[0]
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        # Term 2: A[1,0]*B[0] -> C[1]
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # Term 3: A[1,1]*B[1] -> C[1]
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        
        return U, V, W

    def build_2x2x2_strassen():
        """Rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        rank = 7
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows (i*2+j)
        V = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows (j*2+k)
        W = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows (k*2+i)
        
        # Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Term 0: M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # Term 2: M3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # Term 3: M4
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # Term 4: M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # Term 5: M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def build_2x2x3():
        """
        Rank-11 decomposition for 2x2 * 2x3 -> 2x3.
        Split B into B1 (2x1) and B2 (2x2).
        A*B1 uses rank-4 schoolbook.
        A*B2 uses rank-7 Strassen.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]  # 2*2 = 4 rows
        V = [[0]*rank for _ in range(6)]  # 2*3 = 6 rows (j*3+k)
        W = [[0]*rank for _ in range(6)]  # 3*2 = 6 rows (k*2+i)
        
        # Part 1: A*B1 (rank 4), B1 is column 0 of B, C1 is column 0 of C.
        # B1[0] -> V index 0*3+0 = 0
        # B1[1] -> V index 1*3+0 = 3
        # C1[0] -> W index 0*2+0 = 0
        # C1[1] -> W index 0*2+1 = 1
        
        U1, V1, W1 = build_2x2x1()
        for c in range(4):
            for r in range(4):
                if U1[r][c] != 0:
                    U[r][c] += U1[r][c]
            for r in range(2):
                if V1[r][c] != 0:
                    V[r*3][c] += V1[r][c]
            for r in range(2):
                if W1[r][c] != 0:
                    W[r][c] += W1[r][c]
        
        # Part 2: A*B2 (rank 7), B2 is columns 1,2 of B, C2 is columns 1,2 of C.
        # B2[0,0] -> B[0,1] -> V index 0*3+1 = 1
        # B2[0,1] -> B[0,2] -> V index 0*3+2 = 2
        # B2[1,0] -> B[1,1] -> V index 1*3+1 = 4
        # B2[1,1] -> B[1,2] -> V index 1*3+2 = 5
        # C2[0,0] -> C[0,1] -> W index 1*2+0 = 2
        # C2[0,1] -> C[0,2] -> W index 2*2+0 = 4
        # C2[1,0] -> C[1,1] -> W index 1*2+1 = 3
        # C2[1,1] -> C[1,2] -> W index 2*2+1 = 5
        
        U2, V2, W2 = build_2x2x2_strassen()
        base = 4
        for c in range(7):
            for r in range(4):
                if U2[r][c] != 0:
                    U[r][base + c] += U2[r][c]
            # Map V2 indices (j*2+k) to global V indices (j*3 + (k+1))
            v2_map = {0: 1, 1: 2, 2: 4, 3: 5}
            for r in range(4):
                if V2[r][c] != 0:
                    V[v2_map[r]][base + c] += V2[r][c]
            # Map W2 indices (k*2+i) to global W indices
            # W2[0] = C2[0,0] -> C[0,1] -> W[1*2+0] = 2
            # W2[1] = C2[0,1] -> C[0,2] -> W[2*2+0] = 4
            # W2[2] = C2[1,0] -> C[1,1] -> W[1*2+1] = 3
            # W2[3] = C2[1,1] -> C[1,2] -> W[2*2+1] = 5
            w2_map = {0: 2, 1: 4, 2: 3, 3: 5}
            for r in range(4):
                if W2[r][c] != 0:
                    W[w2_map[r]][base + c] += W2[r][c]
        
        return U, V, W

    def build_4x4x6():
        """
        Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6.
        Split A into 2x2 blocks, B into 2x3 blocks.
        Apply Strassen's 7 terms, each sub-product is 2x2 * 2x3 (rank 11).
        Total rank = 7 * 11 = 77.
        """
        n, m, p = 4, 4, 6
        n2, m2, p2 = 2, 2, 3
        
        # Get the 2x2x3 sub-decomposition
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])  # 11
        total_rank = 7 * rank_sub  # 77
        
        U = [[0]*total_rank for _ in range(n*m)]  # 16 rows
        V = [[0]*total_rank for _ in range(m*p)]  # 24 rows
        W = [[0]*total_rank for _ in range(p*n)]  # 24 rows
        
        # Strassen terms for 4x4 * 4x6:
        # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3),
        #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
        # B blocks: B11 (rows 0-1, cols 0-2), B12 (rows 0-1, cols 3-5),
        #           B21 (rows 2-3, cols 0-2), B22 (rows 2-3, cols 3-5)
        # C blocks: C11 (rows 0-1, cols 0-2), C12 (rows 0-1, cols 3-5),
        #           C21 (rows 2-3, cols 0-2), C22 (rows 2-3, cols 3-5)
        
        # U index: i*m + j = i*4 + j
        # V index: j*p + k = j*6 + k
        # W index: k*n + i = k*4 + i
        
        # Sub-problem local indices:
        # U_sub: i*m2 + j = i*2 + j
        # V_sub: j*p2 + k = j*3 + k
        # W_sub: k*n2 + i = k*2 + i
        
        # Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Define the 7 Strassen terms as (U_blocks, V_blocks, W_blocks)
        # Each block is (block_idx, sign) where block_idx is 0..3 for A, B, C blocks
        # A blocks: 0=A11, 1=A12, 2=A21, 3=A22
        # B blocks: 0=B11, 1=B12, 2=B21, 3=B22
        # C blocks: 0=C11, 1=C12, 2=C21, 3=C22
        
        strassen_terms = [
            # (U_blocks, V_blocks, W_blocks)
            # M1: (A11+A22)(B11+B22)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            # M2: (A21+A22)B11
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            # M3: A11(B12-B22)
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            # M4: A22(B21-B11)
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            # M5: (A11+A12)B22
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            # M6: (A21-A11)(B11+B12)
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            # M7: (A12-A22)(B21+B22)
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        # Block offsets:
        # A block 0 (A11): rows 0-1, cols 0-1 -> U offset (0,0)
        # A block 1 (A12): rows 0-1, cols 2-3 -> U offset (0,2)
        # A block 2 (A21): rows 2-3, cols 0-1 -> U offset (2,0)
        # A block 3 (A22): rows 2-3, cols 2-3 -> U offset (2,2)
        
        # B block 0 (B11): rows 0-1, cols 0-2 -> V offset (0,0)
        # B block 1 (B12): rows 0-1, cols 3-5 -> V offset (0,3)
        # B block 2 (B21): rows 2-3, cols 0-2 -> V offset (2,0)
        # B block 3 (B22): rows 2-3, cols 3-5 -> V offset (2,3)
        
        # C block 0 (C11): rows 0-1, cols 0-2 -> W offset (0,0)
        # C block 1 (C12): rows 0-1, cols 3-5 -> W offset (0,3)
        # C block 2 (C21): rows 2-3, cols 0-2 -> W offset (2,0)
        # C block 3 (C22): rows 2-3, cols 3-5 -> W offset (2,3)
        
        a_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
        b_offsets = [(0, 0), (0, 3), (2, 0), (2, 3)]
        c_offsets = [(0, 0), (0, 3), (2, 0), (2, 3)]
        
        for term_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_terms):
            col_offset = term_idx * rank_sub
            
            # Add U contributions
            for block_idx, sign in u_blocks:
                row_off, col_off = a_offsets[block_idx]
                for i in range(2):
                    for j in range(2):
                        local_u_idx = i * 2 + j
                        global_u_idx = (row_off + i) * 4 + (col_off + j)
                        if U_sub[local_u_idx][0] != 0:
                            # We need to add the entire sub-decomposition
                            pass
                        # Actually, we need to add all rank_sub columns
                        for c in range(rank_sub):
                            if U_sub[local_u_idx][c] != 0:
                                U[global_u_idx][col_offset + c] += sign * U_sub[local_u_idx][c]
            
            # Add V contributions
            for block_idx, sign in v_blocks:
                row_off, col_off = b_offsets[block_idx]
                for j in range(2):
                    for k in range(3):
                        local_v_idx = j * 3 + k
                        global_v_idx = (row_off + j) * 6 + (col_off + k)
                        for c in range(rank_sub):
                            if V_sub[local_v_idx][c] != 0:
                                V[global_v_idx][col_offset + c] += sign * V_sub[local_v_idx][c]
            
            # Add W contributions
            for block_idx, sign in w_blocks:
                row_off, col_off = c_offsets[block_idx]
                for k in range(3):
                    for i in range(2):
                        local_w_idx = k * 2 + i
                        global_w_idx = (row_off + k) * 4 + (col_off + i)
                        for c in range(rank_sub):
                            if W_sub[local_w_idx][c] != 0:
                                W[global_w_idx][col_offset + c] += sign * W_sub[local_w_idx][c]
        
        return U, V, W

    # Build the decomposition
    U, V, W = build_4x4x6()
    
    # Verify correctness (optional, for debugging)
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta
    # Actually, the decomposition is: C[i,k] = sum_r sum_j U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k] * W[k,i,r]
    # Wait, the standard tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T is the matrix multiplication tensor.
    
    # Let's verify with random matrices
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        A_test = rng.random((4, 4))
        B_test = rng.random((4, 6))
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
        # Actually: C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
        
        C_computed = np.zeros((4, 6))
        for r in range(77):
            # Compute U_part[i] = sum_j U[i,j,r] * A[i,j]
            U_part = np.zeros(4)
            for i in range(4):
                for j in range(4):
                    U_part[i] += U_np[i*4+j][r] * A_test[i,j]
            
            # Compute V_part[k] = sum_j V[j,k,r] * B[j,k]
            V_part = np.zeros(6)
            for k in range(6):
                for j in range(4):
                    V_part[k] += V_np[j*6+k][r] * B_test[j,k]
            
            # C[i,k] += U_part[i] * V_part[k] * W[k,i,r]
            for i in range(4):
                for k in range(6):
                    C_computed[i,k] += U_part[i] * V_part[k] * W_np[k*4+i][r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            rank = 96
            U = [[0] * rank for _ in range(16)]
            V = [[0] * rank for _ in range(24)]
            W = [[0] * rank for _ in range(24)]
            column = 0
            for i in range(4):
                for j in range(4):
                    for k in range(6):
                        U[i * 4 + j][column] = 1
                        V[j * 6 + k][column] = 1
                        W[k * 4 + i][column] = 1
                        column += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": W, "W": W}  # Placeholder, should not reach here

# EVOLVE-BLOCK-END
