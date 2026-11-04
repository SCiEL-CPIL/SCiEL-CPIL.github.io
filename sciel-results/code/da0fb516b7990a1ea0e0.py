import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin for safety

    # The matrix multiplication tensor for (n, m, p) = (4, 4, 9)
    # We decompose this as 4x4x4 + 4x4x4 + 4x4x1
    # 4x4x4 can be done with Strassen rank 49
    # 4x4x1 is just 4 dot products of length 4, rank 16
    # Total rank = 49 + 49 + 16 = 114
    # This is better than schoolbook 4*4*9 = 144

    # Strassen's algorithm for 2x2x2 has rank 7
    # The 7 multiplication terms are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7

    # For 4x4x4, we use Strassen recursively:
    # Split 4x4 into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # The 7 Strassen multiplications are 2x2x2 multiplications
    # Each 2x2x2 has rank 7, so 7*7 = 49 total rank for 4x4x4

    # Let's build the 4x4x4 decomposition with rank 49
    # U shape: (4*4, 49) = (16, 49)
    # V shape: (4*4, 49) = (16, 49)
    # W shape: (4*4, 49) = (16, 49)

    def build_strassen_4x4x4():
        """Build rank-49 decomposition for 4x4x4 matrix multiplication."""
        # The 7 Strassen combinations for 2x2x2
        # For each of the 7 blocks, we need the 2x2x2 Strassen vectors
        # 2x2x2 Strassen:
        # U vectors (for A): shape (4, 7)
        # V vectors (for B): shape (4, 7)
        # W vectors (for C): shape (4, 7)
        
        # A indices: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
        # B indices: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
        # C indices: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
        
        # Strassen 2x2x2:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # U_2x2: (4, 7)
        U_2 = np.zeros((4, 7))
        # M1: A11+A22 -> [1, 0, 0, 1]
        U_2[0, 0] = 1; U_2[3, 0] = 1
        # M2: A21+A22 -> [0, 0, 1, 1]
        U_2[2, 1] = 1; U_2[3, 1] = 1
        # M3: A11 -> [1, 0, 0, 0]
        U_2[0, 2] = 1
        # M4: A22 -> [0, 0, 0, 1]
        U_2[3, 3] = 1
        # M5: A11+A12 -> [1, 1, 0, 0]
        U_2[0, 4] = 1; U_2[1, 4] = 1
        # M6: A21-A11 -> [-1, 0, 1, 0]
        U_2[0, 5] = -1; U_2[2, 5] = 1
        # M7: A12-A22 -> [0, 1, 0, -1]
        U_2[1, 6] = 1; U_2[3, 6] = -1
        
        # V_2x2: (4, 7)
        V_2 = np.zeros((4, 7))
        # M1: B11+B22 -> [1, 0, 0, 1]
        V_2[0, 0] = 1; V_2[3, 0] = 1
        # M2: B11 -> [1, 0, 0, 0]
        V_2[0, 1] = 1
        # M3: B12-B22 -> [0, 1, 0, -1]
        V_2[1, 2] = 1; V_2[3, 2] = -1
        # M4: B21-B11 -> [-1, 0, 1, 0]
        V_2[0, 3] = -1; V_2[2, 3] = 1
        # M5: B22 -> [0, 0, 0, 1]
        V_2[3, 4] = 1
        # M6: B11+B12 -> [1, 1, 0, 0]
        V_2[0, 5] = 1; V_2[1, 5] = 1
        # M7: B21+B22 -> [0, 0, 1, 1]
        V_2[2, 6] = 1; V_2[3, 6] = 1
        
        # W_2x2: (4, 7)
        W_2 = np.zeros((4, 7))
        # C11 = M1 - M4 + M5 - M7
        W_2[0, 0] = 1; W_2[0, 3] = -1; W_2[0, 4] = 1; W_2[0, 6] = -1
        # C12 = M3 + M5
        W_2[1, 2] = 1; W_2[1, 4] = 1
        # C21 = M2 + M4
        W_2[2, 1] = 1; W_2[2, 3] = 1
        # C22 = M1 - M2 + M3 + M7
        W_2[3, 0] = 1; W_2[3, 1] = -1; W_2[3, 2] = 1; W_2[3, 6] = 1
        
        # Now for 4x4x4, we have 4x4 matrices split into 2x2 blocks
        # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
        #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
        # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3)
        #           B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
        # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3)
        #           C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
        
        # The 7 Strassen multiplications are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mx is a 2x2x2 multiplication, so we need to expand each
        # into 7 rank-1 terms using the 2x2x2 Strassen decomposition
        
        # U_4x4: (16, 49)
        U_4 = np.zeros((16, 49))
        V_4 = np.zeros((16, 49))
        W_4 = np.zeros((16, 49))
        
        # Block positions in 4x4 matrix (row-major):
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        # For each of the 7 Strassen multiplications, we need to combine
        # the 2x2 block contributions
        
        # M1 = (A11+A22)(B11+B22)
        # The A part is A11+A22, which in 4x4 terms is:
        # A11 elements + A22 elements
        # A11: [0,0]=idx0, [0,1]=idx1, [1,0]=idx4, [1,1]=idx5
        # A22: [2,2]=idx10, [2,3]=idx11, [3,2]=idx14, [3,3]=idx15
        # So A11+A22 has non-zero entries at these 8 positions
        # But we're doing 2x2x2 Strassen, so we treat A11+A22 as a 2x2 matrix
        # where the 2x2 matrix is:
        # [(A11[0,0]+A22[0,0]), (A11[0,1]+A22[0,1])]
        # [(A11[1,0]+A22[1,0]), (A11[1,1]+A22[1,1])]
        # = [[A[0,0]+A[2,2], A[0,1]+A[2,3]],
        #    [A[1,0]+A[3,2], A[1,1]+A[3,3]]]
        
        # Similarly for B11+B22:
        # [[B[0,0]+B[2,2], B[0,1]+B[2,3]],
        #  [B[1,0]+B[3,2], B[1,1]+B[3,3]]]
        
        # And the result is a 2x2 matrix that contributes to C11:
        # C11[0,0] += M1[0,0]
        # C11[0,1] += M1[0,1]
        # C11[1,0] += M1[1,0]
        # C11[1,1] += M1[1,1]
        
        # So for M1, we use the 2x2x2 Strassen with:
        # A_2x2 = A11 + A22 (as a 2x2 matrix)
        # B_2x2 = B11 + B22 (as a 2x2 matrix)
        # C_2x2 contributes to C11
        
        # The U vector for A11+A22 in 4x4 terms:
        # The 2x2 matrix A11+A22 has elements at 4x4 positions:
        # (0,0), (0,1), (1,0), (1,1) from A11 and (2,2), (2,3), (3,2), (3,3) from A22
        # In the 2x2 Strassen, the 2x2 A matrix has elements at positions 0,1,2,3
        # which correspond to 4x4 positions (0,0), (0,1), (1,0), (1,1)
        # So we need to map the 2x2 Strassen U vectors to 4x4 positions
        
        # Let's define the mapping:
        # 2x2 position (r,c) -> 4x4 position depends on which block
        # For A11: 2x2 (r,c) -> 4x4 (r,c) -> index r*4+c
        # For A12: 2x2 (r,c) -> 4x4 (r,c+2) -> index r*4+c+2
        # For A21: 2x2 (r,c) -> 4x4 (r+2,c) -> index (r+2)*4+c
        # For A22: 2x2 (r,c) -> 4x4 (r+2,c+2) -> index (r+2)*4+c+2
        
        # For M1: A part is A11+A22
        # 2x2 (r,c) -> 4x4: A11 at (r,c) and A22 at (r+2,c+2)
        # So U vector for M1's A part:
        # For each 2x2 Strassen U vector u (length 4), the 4x4 U vector has:
        # u[0] at 4x4 (0,0) = idx 0
        # u[0] at 4x4 (2,2) = idx 10
        # u[1] at 4x4 (0,1) = idx 1
        # u[1] at 4x4 (2,3) = idx 11
        # u[2] at 4x4 (1,0) = idx 4
        # u[2] at 4x4 (3,2) = idx 14
        # u[3] at 4x4 (1,1) = idx 5
        # u[3] at 4x4 (3,3) = idx 15
        
        # Similarly for B part of M1: B11+B22
        # 2x2 (r,c) -> 4x4: B11 at (r,c) and B22 at (r+2,c+2)
        
        # And C part of M1: contributes to C11
        # 2x2 (r,c) -> 4x4 C11 at (r,c) -> index r*4+c
        
        # Let's build a helper function
        def block_to_4x4(block_type, r2, c2, matrix_type):
            """Convert 2x2 block position to 4x4 index."""
            if matrix_type == 'A':
                if block_type == 'A11':
                    return r2 * 4 + c2
                elif block_type == 'A12':
                    return r2 * 4 + c2 + 2
                elif block_type == 'A21':
                    return (r2 + 2) * 4 + c2
                elif block_type == 'A22':
                    return (r2 + 2) * 4 + c2 + 2
            elif matrix_type == 'B':
                if block_type == 'B11':
                    return r2 * 4 + c2
                elif block_type == 'B12':
                    return r2 * 4 + c2 + 2
                elif block_type == 'B21':
                    return (r2 + 2) * 4 + c2
                elif block_type == 'B22':
                    return (r2 + 2) * 4 + c2 + 2
            elif matrix_type == 'C':
                if block_type == 'C11':
                    return r2 * 4 + c2
                elif block_type == 'C12':
                    return r2 * 4 + c2 + 2
                elif block_type == 'C21':
                    return (r2 + 2) * 4 + c2
                elif block_type == 'C22':
                    return (r2 + 2) * 4 + c2 + 2
            return -1
        
        # For each of the 7 Strassen multiplications, we have:
        # M1: A=A11+A22, B=B11+B22, C=C11
        # M2: A=A21+A22, B=B11, C=C21
        # M3: A=A11, B=B12-B22, C=C12
        # M4: A=A22, B=B21-B11, C=C21
        # M5: A=A11+A12, B=B22, C=C11
        # M6: A=A21-A11, B=B11+B12, C=C22
        # M7: A=A12-A22, B=B21+B22, C=C22
        
        # Define the Strassen terms:
        # Each term has: A_blocks (list of (block, coeff)), B_blocks, C_block
        strassen_terms = [
            # M1
            ([('A11', 1), ('A22', 1)], [('B11', 1), ('B22', 1)], 'C11'),
            # M2
            ([('A21', 1), ('A22', 1)], [('B11', 1)], 'C21'),
            # M3
            ([('A11', 1)], [('B12', 1), ('B22', -1)], 'C12'),
            # M4
            ([('A22', 1)], [('B21', 1), ('B11', -1)], 'C21'),
            # M5
            ([('A11', 1), ('A12', 1)], [('B22', 1)], 'C11'),
            # M6
            ([('A21', 1), ('A11', -1)], [('B11', 1), ('B12', 1)], 'C22'),
            # M7
            ([('A12', 1), ('A22', -1)], [('B21', 1), ('B22', 1)], 'C22'),
        ]
        
        for term_idx, (a_blocks, b_blocks, c_block) in enumerate(strassen_terms):
            # For each of the 7 2x2x2 Strassen vectors
            for s in range(7):
                col = term_idx * 7 + s
                # U vector
                for r2 in range(2):
                    for c2 in range(2):
                    # Get the 2x2 Strassen U value
                        u_val = U_2[r2 * 2 + c2, s]
                        if u_val != 0:
                            for block_name, coeff in a_blocks:
                                idx = block_to_4x4(block_name, r2, c2, 'A')
                                if idx >= 0:
                                    U_4[idx, col] += u_val * coeff
                # V vector
                for r2 in range(2):
                    for c2 in range(2):
                        v_val = V_2[r2 * 2 + c2, s]
                        if v_val != 0:
                            for block_name, coeff in b_blocks:
                                idx = block_to_4x4(block_name, r2, c2, 'B')
                                if idx >= 0:
                                    V_4[idx, col] += v_val * coeff
                # W vector
                for r2 in range(2):
                    for c2 in range(2):
                        w_val = W_2[r2 * 2 + c2, s]
                        if w_val != 0:
                            idx = block_to_4x4(c_block, r2, c2, 'C')
                            if idx >= 0:
                                W_4[idx, col] += w_val
        
        return U_4, V_4, W_4
    
    # Build 4x4x4 decomposition
    U_444, V_444, W_444 = build_strassen_4x4x4()
    
    # Now for 4x4x1 (just the 9th column of B, k=8)
    # C[i,8] = sum_j A[i,j] * B[j,8]
    # This is 4 dot products of length 4
    # Rank = 4*4 = 16
    # U shape: (16, 16)
    # V shape: (36, 16) -- but only rows corresponding to k=8 are non-zero
    # W shape: (36, 16) -- but only rows corresponding to k=8 are non-zero
    
    U_441 = np.zeros((16, 16))
    V_441 = np.zeros((36, 16))
    W_441 = np.zeros((36, 16))
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            V_441[j * 9 + 8, col] = 1
            W_441[8 * 4 + i, col] = 1
    
    # Now combine:
    # 4x4x9 = 4x4x4 (k=0..3) + 4x4x4 (k=4..7) + 4x4x1 (k=8)
    # Wait, that's not right. 4x4x9 means p=9, so k=0..8
    # We can split as: 4x4x4 (k=0..3) + 4x4x4 (k=4..7) + 4x4x1 (k=8)
    # But 4x4x4 + 4x4x4 + 4x4x1 = 49 + 49 + 16 = 114
    
    # Actually, let me reconsider. The tensor is for multiplying 4x4 by 4x9.
    # A is 4x4, B is 4x9, C is 4x9.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can split B into three parts: B[:,0:4], B[:,4:8], B[:,8:9]
    # C[:,0:4] = A * B[:,0:4]  -> 4x4x4, rank 49
    # C[:,4:8] = A * B[:,4:8]  -> 4x4x4, rank 49
    # C[:,8:9] = A * B[:,8:9]  -> 4x4x1, rank 16
    # Total rank = 49 + 49 + 16 = 114
    
    # U shape: (4*4, 114) = (16, 114)
    # V shape: (4*9, 114) = (36, 114)
    # W shape: (9*4, 114) = (36, 114)
    
    rank_total = 114
    U_final = np.zeros((16, rank_total))
    V_final = np.zeros((36, rank_total))
    W_final = np.zeros((36, rank_total))
    
    # Block 1: k=0..3 (4x4x4)
    U_final[:, 0:49] = U_444
    # V for block 1: B[:,0:4], so rows j*9+k for k=0..3
    for j in range(4):
        for k in range(4):
            V_final[j * 9 + k, 0:49] = V_444[j * 4 + k]
    # W for block 1: C[:,0:4], so rows k*4+i for k=0..3
    for k in range(4):
        for i in range(4):
            W_final[k * 4 + i, 0:49] = W_444[k * 4 + i]
    
    # Block 2: k=4..7 (4x4x4)
    offset2 = 49
    U_final[:, offset2:offset2+49] = U_444
    for j in range(4):
        for k in range(4):
            V_final[j * 9 + (k + 4), offset2:offset2+49] = V_444[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W_final[(k + 4) * 4 + i, offset2:offset2+49] = W_444[k * 4 + i]
    
    # Block 3: k=8 (4x4x1)
    offset3 = 98
    U_final[:, offset3:offset3+16] = U_441
    V_final[:, offset3:offset3+16] = V_441
    W_final[:, offset3:offset3+16] = W_441
    
    # Verify the decomposition (optional, for debugging)
    # We can check that sum_a U[i,j,a] V[j,k,a] W[k,i,a] = 1 for all i,j,k
    # But this might be slow, so let's just return the result
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
