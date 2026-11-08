"""Tensor decomposition for 2x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # 140s limit, leaving 10s margin for 150s wall-clock

    n, m, p = DIMENSIONS  # 2, 5, 5
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for the tensor T[i,j,k] = delta, we have:
    # T = sum_c U[:,c] * V[:,c] * W[:,c]
    # i.e., U[i,j,c] * V[j,k,c] * W[k,i,c] summed over c = 1 if the tensor entry is 1, else 0
    
    # The matrix multiplication tensor for multiplying A (n x m) by B (m x p) gives C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] which is 1 (since the product A[i,j]*B[j,k] contributes to C[i,k])
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Start with schoolbook: rank = n*m*p = 2*5*5 = 50
    # Try to improve using block structure.
    
    # Block decomposition approach:
    # A is 2x5, B is 5x5, C is 2x5
    # Split A into A1 (2x2) and A2 (2x3)
    # Split B into B1 (2x5) and B2 (3x5)
    # C = A1*B1 + A2*B2
    
    # For A1*B1: 2x2 * 2x5 -> use Strassen-like or simple
    # For A2*B2: 2x3 * 3x5 -> simple rank 30... that's worse
    
    # Let's think more carefully. The schoolbook gives rank 50.
    # 
    # Block decomposition:
    # A (2x5) = [A_left (2x2) | A_right (2x3)]
    # B (5x5) = [B_top (2x5); B_bottom (3x5)]
    # C (2x5) = A_left @ B_top + A_right @ B_bottom
    #
    # Sub-problem 1: (2x2) @ (2x5) -> rank at most 4*5 = 20
    #   Further split B_top into (2x2) and (2x3):
    #   (2x2)@(2x2): Strassen rank 7
    #   (2x2)@(2x3): rank 12 (schoolbook)
    #   Total: 7 + 12 = 19
    #
    # Sub-problem 2: (2x3) @ (3x5) -> rank at most 6*5 = 30
    #   Split B_bottom into (3x2) and (3x3):
    #   (2x3)@(3x2): rank 12
    #   (2x3)@(3x3): rank 18
    #   Total: 12 + 18 = 30
    #
    # Grand total: 19 + 30 = 49. Not better than 50.
    
    # Alternative: use the fact that Strassen gives 7 for 2x2x2.
    # Let's try a different split.
    
    # Actually, let me just implement the schoolbook first and verify it works,
    # then try a Strassen-based decomposition.
    
    # Schoolbook: rank = n*m*p = 50
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Verify schoolbook
    def verify(U_mat, V_mat, W_mat):
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W):
        pass  # Schoolbook is valid
    
    # Now try Strassen-based decomposition for better rank.
    # For 2x2x2, Strassen uses 7 terms.
    # We can decompose the 2x5x5 problem using blocks.
    
    # Let's build a decomposition using Strassen for the 2x2x2 sub-blocks.
    # A (2x5), B (5x5)
    # Split column index j (of A) and row index j (of B) into two groups: j in {0,1} and j in {2,3,4}
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:5] (2x3)
    # B_top = B[0:2, :] (2x5), B_bottom = B[2:5, :] (3x5)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left @ B_top (2x2 @ 2x5):
    #   Split B_top into B_top_left (2x2) and B_top_right (2x3)
    #   A_left @ B_top_left: 2x2x2, Strassen rank 7
    #   A_left @ B_top_right: 2x2x3, schoolbook rank 12
    #   Total: 19
    
    # For A_right @ B_bottom (2x3 @ 3x5):
    #   Split B_bottom into B_bottom_left (3x2) and B_bottom_right (3x3)
    #   A_right @ B_bottom_left: 2x3x2, schoolbook rank 12
    #   A_right @ B_bottom_right: 2x3x3, schoolbook rank 18
    #   Total: 30
    
    # Grand total: 49. Still not better.
    
    # Let me try a different approach: use the full Strassen structure more cleverly.
    # Actually, for the 2x2x2 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me construct the Strassen decomposition for 2x2x2 explicitly.
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen. n=2, m=2, p=2. rank=7."""
        n, m, p = 2, 2, 2
        rank = 7
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # M1 = (A11+A22)(B11+B22)
        # A11 = A[0,0], A22 = A[1,1], B11 = B[0,0], B22 = B[1,1]
        # U: A[0,0]=1, A[1,1]=1 -> U[0*2+0,0]=1, U[1*2+1,0]=1
        # V: B[0,0]=1, B[1,1]=1 -> V[0*2+0,0]=1, V[1*2+1,0]=1
        # W: C[0,0] gets +1, C[1,1] gets +1 -> W[0*2+0,0]=1, W[1*2+1,0]=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        # A21 = A[1,0], A22 = A[1,1], B11 = B[0,0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0*2+1, 1] = 1  # C[1,0]
        
        # M3 = A11(B12-B22)
        # A11 = A[0,0], B12 = B[0,1], B22 = B[1,1]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1*2+0, 2] = 1  # C[0,1]
        
        # M4 = A22(B21-B11)
        # A22 = A[1,1], B21 = B[1,0], B11 = B[0,0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0*2+1, 3] = 1  # C[1,0]
        W[1*2+1, 3] = 1  # C[1,1]
        
        # M5 = (A11+A12)B22
        # A11 = A[0,0], A12 = A[0,1], B22 = B[1,1]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1*2+0, 4] = 1  # C[0,1]
        W[1*2+1, 4] = 1  # C[1,1]
        
        # M6 = (A21-A11)(B11+B12)
        # A21 = A[1,0], A11 = A[0,0], B11 = B[0,0], B12 = B[0,1]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1*2+1, 5] = 1  # C[1,1]
        
        # M7 = (A12-A22)(B21+B22)
        # A12 = A[0,1], A22 = A[1,1], B21 = B[1,0], B22 = B[1,1]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0*2+0, 6] = 1  # C[0,0]
        W[1*2+0, 6] = 1  # C[0,1]
        
        return U, V, W
    
    # Verify Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    n_s, m_s, p_s = 2, 2, 2
    valid_strassen = True
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                val = 0.0
                for c in range(7):
                    val += U_s[i*m_s+j, c] * V_s[j*p_s+k, c] * W_s[k*n_s+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid_strassen = False
                    break
            if not valid_strassen:
                break
        if not valid_strassen:
            break
    
    if valid_strassen:
        print("Strassen 2x2x2 verified")
    
    # Now build the full 2x5x5 decomposition using blocks.
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x5) = [B1 (2x5); B2 (3x5)]
    # C (2x5) = A1 @ B1 + A2 @ B2
    
    # Sub-problem 1: A1 (2x2) @ B1 (2x5)
    #   Split B1 = [B1a (2x2) | B1b (2x3)]
    #   A1 @ B1a: 2x2x2 Strassen, rank 7
    #   A1 @ B1b: 2x2x3 schoolbook, rank 12
    #   Total: 19
    
    # Sub-problem 2: A2 (2x3) @ B2 (3x5)
    #   Split B2 = [B2a (3x2) | B2b (3x3)]
    #   A2 @ B2a: 2x3x2 schoolbook, rank 12
    #   A2 @ B2b: 2x3x3 schoolbook, rank 18
    #   Total: 30
    
    # Grand total: 49
    
    # Let's build this explicitly.
    def build_block_decomp():
        n, m, p = 2, 5, 5
        # Block boundaries
        # A columns: 0-1 (A1), 2-4 (A2)
        # B rows: 0-1 (B1), 2-4 (B2)
        # B1 cols: 0-1 (B1a), 2-4 (B1b)
        # B2 cols: 0-1 (B2a), 2-4 (B2b)
        
        # Sub-problem 1a: A1 (2x2) @ B1a (2x2), Strassen rank 7
        # Global indices:
        # A1: rows i=0,1; cols j=0,1 -> U indices: i*5+j
        # B1a: rows j=0,1; cols k=0,1 -> V indices: j*5+k
        # C: rows i=0,1; cols k=0,1 -> W indices: k*2+i
        
        # Sub-problem 1b: A1 (2x2) @ B1b (2x3), schoolbook rank 12
        # A1: rows i=0,1; cols j=0,1
        # B1b: rows j=0,1; cols k=2,3,4
        # C: rows i=0,1; cols k=2,3,4
        
        # Sub-problem 2a: A2 (2x3) @ B2a (3x2), schoolbook rank 12
        # A2: rows i=0,1; cols j=2,3,4
        # B2a: rows j=2,3,4; cols k=0,1
        # C: rows i=0,1; cols k=0,1
        
        # Sub-problem 2b: A2 (2x3) @ B2b (3x3), schoolbook rank 18
        # A2: rows i=0,1; cols j=2,3,4
        # B2b: rows j=2,3,4; cols k=2,3,4
        # C: rows i=0,1; cols k=2,3,4
        
        total_rank = 7 + 12 + 12 + 18
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col_offset = 0
        
        # Sub-problem 1a: Strassen for A1 @ B1a
        # A1 indices: i in {0,1}, j in {0,1} -> U row = i*5 + j
        # B1a indices: j in {0,1}, k in {0,1} -> V row = j*5 + k
        # C indices: i in {0,1}, k in {0,1} -> W row = k*2 + i
        
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, c] != 0:
                        U[i*5+j, col_offset+c] = U_s[i*2+j, c]
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k, c] != 0:
                        V[j*5+k, col_offset+c] = V_s[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i, c] != 0:
                        W[k*2+i, col_offset+c] = W_s[k*2+i, c]
        col_offset += 7
        
        # Sub-problem 1b: schoolbook for A1 (2x2) @ B1b (2x3)
        # i in {0,1}, j in {0,1}, k in {2,3,4}
        for i in range(2):
            for j in range(2):
                for k in range(2, 5):
                    U[i*5+j, col_offset] = 1.0
                    V[j*5+k, col_offset] = 1.0
                    W[k*2+i, col_offset] = 1.0
                    col_offset += 1
        # 2*2*3 = 12 terms
        
        # Sub-problem 2a: schoolbook for A2 (2x3) @ B2a (3x2)
        # i in {0,1}, j in {2,3,4}, k in {0,1}
        for i in range(2):
            for j in range(2, 5):
                for k in range(2):
                    U[i*5+j, col_offset] = 1.0
                    V[j*5+k, col_offset] = 1.0
                    W[k*2+i, col_offset] = 1.0
                    col_offset += 1
        # 2*3*2 = 12 terms
        
        # Sub-problem 2b: schoolbook for A2 (2x3) @ B2b (3x3)
        # i in {0,1}, j in {2,3,4}, k in {2,3,4}
        for i in range(2):
            for j in range(2, 5):
                for k in range(2, 5):
                    U[i*5+j, col_offset] = 1.0
                    V[j*5+k, col_offset] = 1.0
                    W[k*2+i, col_offset] = 1.0
                    col_offset += 1
        # 2*3*3 = 18 terms
        
        return U, V, W
    
    U_bd, V_bd, W_bd = build_block_decomp()
    
    # Verify block decomposition
    n, m, p = 2, 5, 5
    valid_bd = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U_bd.shape[1]):
                    val += U_bd[i*m+j, c] * V_bd[j*p+k, c] * W_bd[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid_bd = False
                    break
            if not valid_bd:
                break
        if not valid_bd:
            break
    
    if valid_bd:
        print("Block decomposition verified, rank =", U_bd.shape[1])
        if U_bd.shape[1] < best_rank:
            best_U, best_V, best_W = U_bd, V_bd, W_bd
            best_rank = U_bd.shape[1]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Return best found so far
        pass
    
    # Try to see if we can do even better with a different block structure.
    # Alternative: Split B into 5 single columns and use Strassen where applicable.
    # This is getting complex. Let's try another approach:
    # Use the 2x2x2 Strassen for overlapping blocks.
    
    # Actually, let me try a completely different decomposition.
    # What if we split the 5x5 B matrix into a 2x2 block and a 3x3 block,
    # and use Strassen for the 2x2 part?
    
    # Let's try: A (2x5), B (5x5)
    # Split by rows of B / columns of A:
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [B1 (2x5); B2 (3x5)]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x5):
    # Use the fact that B1 is 2x5. We can write B1 = [b1, b2, b3, b4, b5] (each 2x1)
    # A1*B1 = A1*b1 + A1*b2 + A1*b3 + A1*b4 + A1*b5
    # Each A1*bi is 2x2 * 2x1, which is just 2 multiplications each.
    # Total: 5 * 2 = 10. But Strassen for 2x2x2 gives 7 for the 2x2x2 part.
    
    # Actually, the 2x2 * 2x5 problem: we can think of it as computing 5 separate 2x2*2x1 products.
    # But we can batch them. The rank of 2x2x5 is at most 2*2*5 = 20, but can be less.
    # Using Strassen: split B1 into two 2x2 and one 2x1 blocks? No, 5 is odd.
    # Split B1 into (2x2) and (2x3):
    # 2x2x2: Strassen rank 7
    # 2x2x3: This is 2x2 * 2x3. Rank at most 12. Can we do better?
    #   2x2x3: A is 2x2, B is 2x3, C is 2x3.
    #   Schoolbook: 12.
    #   Can we use Strassen-like? Split B into (2x2) and (2x1):
    #   2x2x2: rank 7
    #   2x2x1: rank 4 (each of 2 outputs needs 2 products)
    #   Total: 11
    #   So 2x2x3 can be done in rank 11.
    
    # Let me reconsider:
    # 2x2x3: A (2x2), B (2x3), C (2x3)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split k into k=0,1 and k=2:
    # Part 1: C[:, 0:2] = A @ B[:, 0:2], this is 2x2x2, Strassen rank 7
    # Part 2: C[:, 2] = A @ B[:, 2], this is 2x2x1
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #   This needs 4 products (or 2 if we factor, but in tensor decomposition it's 4 terms)
    #   Actually, it's rank 4: 4 terms with U, V, W.
    # Total: 7 + 4 = 11
    
    # So revised block decomposition:
    # Sub-problem 1: A1 (2x2) @ B1 (2x5)
    #   Split B1 into B1a (2x2) and B1b (2x3)
    #   A1 @ B1a: 2x2x2, Strassen rank 7
    #   A1 @ B1b: 2x2x3, rank 11 (as computed above)
    #   Total: 18
    
    # Sub-problem 2: A2 (2x3) @ B2 (3x5)
    #   Split B2 into B2a (3x2) and B2b (3x3)
    #   A2 @ B2a: 2x3x2, schoolbook rank 12
    #   A2 @ B2b: 2x3x3, schoolbook rank 18
    #   Total: 30
    
    # Grand total: 18 + 30 = 48
    
    # Let me implement this.
    def build_improved_block_decomp():
        n, m, p = 2, 5, 5
        total_rank = 7 + 11 + 12 + 18
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col_offset = 0
        
        # Sub-problem 1a: A1 (2x2) @ B1a (2x2), Strassen rank 7
        # A1: i in {0,1}, j in {0,1} -> U row = i*5+j
        # B1a: j in {0,1}, k in {0,1} -> V row = j*5+k
        # C: i in {0,1}, k in {0,1} -> W row = k*2+i
        
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, c] != 0:
                        U[i*5+j, col_offset+c] = U_s[i*2+j, c]
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k, c] != 0:
                        V[j*5+k, col_offset+c] = V_s[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i, c] != 0:
                        W[k*2+i, col_offset+c] = W_s[k*2+i, c]
        col_offset += 7
        
        # Sub-problem 1b: A1 (2x2) @ B1b (2x3), rank 11
        # B1b: j in {0,1}, k in {2,3,4}
        # Split into:
        #   Part 1: k in {2,3}: 2x2x2, Strassen rank 7
        #   Part 2: k in {4}: 2x2x1, rank 4
        
        # Part 1: A1 @ B1b[:, 0:2] (which is B1b with k=2,3)
        # This is 2x2x2 with k indices 2,3
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, c] != 0:
                        U[i*5+j, col_offset+c] = U_s[i*2+j, c]
            for j in range(2):
                for k_local in range(2):
                    k = 2 + k_local
                    if V_s[j*2+k_local, c] != 0:
                        V[j*5+k, col_offset+c] = V_s[j*2+k_local, c]
            for k_local in range(2):
                k = 2 + k_local
                for i in range(2):
                    if W_s[k_local*2+i, c] != 0:
                        W[k*2+i, col_offset+c] = W_s[k_local*2+i, c]
        col_offset += 7
        
        # Part 2: A1 @ B1b[:, 2] (which is B1b with k=4)
        # 2x2x1: C[i,4] = sum_j A[i,j] B[j,4]
        # 4 terms
        for i in range(2):
            for j in range(2):
                U[i*5+j, col_offset] = 1.0
                V[j*5+4, col_offset] = 1.0
                W[4*2+i, col_offset] = 1.0
                col_offset += 1
        # 4 terms
        
        # Sub-problem 2a: A2 (2x3) @ B2a (3x2), schoolbook rank 12
        # A2: i in {0,1}, j in {2,3,4}
        # B2a: j in {2,3,4}, k in {0,1}
        # C: i in {0,1}, k in {0,1}
        for i in range(2):
            for j in range(2, 5):
                for k in range(2):
                    U[i*5+j, col_offset] = 1.0
                    V[j*5+k, col_offset] = 1.0
                    W[k*2+i, col_offset] = 1.0
                    col_offset += 1
        # 2*3*2 = 12 terms
        
        # Sub-problem 2b: A2 (2x3) @ B2b (3x3), schoolbook rank 18
        # A2: i in {0,1}, j in {2,3,4}
        # B2b: j in {2,3,4}, k in {2,3,4}
        # C: i in {0,1}, k in {2,3,4}
        for i in range(2):
            for j in range(2, 5):
                for k in range(2, 5):
                    U[i*5+j, col_offset] = 1.0
                    V[j*5+k, col_offset] = 1.0
                    W[k*2+i, col_offset] = 1.0
                    col_offset += 1
        # 2*3*3 = 18 terms
        
        return U, V, W
    
    U_imp, V_imp, W_imp = build_improved_block_decomp()
    
    # Verify improved block decomposition
    n, m, p = 2, 5, 5
    valid_imp = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U_imp.shape[1]):
                    val += U_imp[i*m+j, c] * V_imp[j*p+k, c] * W_imp[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid_imp = False
                    break
            if not valid_imp:
                break
        if not valid_imp:
            break
    
    if valid_imp:
        print("Improved block decomposition verified, rank =", U_imp.shape[1])
        if U_imp.shape[1] < best_rank:
            best_U, best_V, best_W = U_imp, V_imp, W_imp
            best_rank = U_imp.shape[1]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 140:
        pass
    
    # Try even better: improve the 2x3x2 and 2x3x3 sub-problems.
    # 2x3x2: A (2x3), B (3x2), C (2x2)
    # Schoolbook: 12. Can we do better?
    # C[i,k] = sum_j A[i,j] B[j,k], j in {0,1,2}, i,k in {0,1}
    # This is a 2x3x2 tensor. The rank is at least... let's think.
    # Actually, this is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
    # The rank of this tensor is at most 2*3*2 = 12, but can be less.
    # I believe the rank of 2x3x2 is 11 (using a Strassen-like trick).
    # Actually, I'm not sure. Let me just use schoolbook for now.
    
    # Let's try to verify if we can get below 48.
    # The current best is 48.
    
    # One more idea: use a different split for the 2x3x3 problem.
    # 2x3x3: A (2x3), B (3x3), C (2x3)
    # Schoolbook: 18.
    # Can we do better? Maybe using Strassen for a 2x2x2 sub-block.
    # Split A into A1 (2x2) and A2 (2x1):
    # A1 @ B[:, 0:2] (2x2x2): Strassen rank 7
    # A1 @ B[:, 2] (2x2x1): rank 4
    # A2 @ B (2x1x3): rank 3 (each C[i,2] = A[2,0]*B[0,2], but A is 2x1 so...)
    # Wait, A2 is 2x1, so A2 @ B is (2x1)@(3x3) which doesn't make sense dimensionally.
    # I messed up. A is 2x3, so splitting columns of A:
    # A = [A1 (2x2) | A2 (2x1)]
    # B is 3x3.
    # C = A1 @ B[:, 0:2] + A2 @ B[:, 2]
    # A1 @ B[:, 0:2]: (2x2) @ (2x3), rank 11 (as before)
    # A2 @ B[:, 2]: (2x1) @ (1x3), rank 6 (each of 2*3=6 outputs needs 1 product)
    # Total: 11 + 6 = 17
    
    # So 2x3x3 can be done in rank 17 instead of 18.
    
    # Revised total: 7 + 11 + 12 + 17 = 47
    
    def build_best_block_decomp():
        n, m, p = 2, 5, 5
        total_rank = 7 + 11 + 12 + 17
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col_offset = 0
        
        # Sub-problem 1a: A1 (2x2) @ B1a (2x2), Strassen rank 7
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, c] != 0:
                        U[i*5+j, col_offset+c] = U_s[i*2+j, c]
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k, c] != 0:
                        V[j*5+k, col_offset+c] = V_s[j*2+k, c]
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i, c] != 0:
                        W[k*2+i, col_offset+c] = W_s[k*2+i, c]
        col_offset += 7
        
        # Sub-problem 1b: A1 (2x2) @ B1b (2x3), rank 11
        # Part 1: k in {2,3}: 2x2x2, Strassen rank 7
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, c] != 0:
                        U[i*5+j, col_offset+c] = U_s[i*2+j, c]
            for j in range(2):
                for k_local in range(2):
                    k = 2 + k_local
                    if V_s[j*2+k_local, c] != 0:
                        V[j*5+k, col_offset+c] = V_s[j*2+k_local, c]
            for k_local in range(2):
                k = 2 + k_local
                for i in range(2):
                    if W_s[k_local*2+i, c] != 0:
                        W[k*2+i, col_offset+c] = W_s[k_local*2+i, c]
        col_offset += 7
        
        # Part 2: k in {4}: 2x2x1, rank 4
        for i in range(2):
            for j in range(2):
                U[i*5+j, col_offset] = 1.0
                V[j*5+4, col_offset] = 1.0
                W[4*2+i, col_offset] = 1.0
                col_offset += 1
        col_offset += 4  # Already incremented in loop
        
        # Sub-problem 2a: A2 (2x3) @ B2a (3x2), schoolbook rank 12
        for i in range(2):
            for j in range(2, 5):
                for k in range(2):
                    U[i*5+j, col_offset] = 1.0
                    V[j*5+k, col_offset] = 1.0
                    W[k*2+i, col_offset] = 1.0
                    col_offset += 1
        # 12 terms
        
        # Sub-problem 2b: A2 (2x3) @ B2b (3x3), rank 17
        # Split A2 into A21 (2x2) and A22 (2x1)
        # A21 @ B2b[:, 0:2] (2x2x2): Strassen rank 7
        # A21 @ B2b[:, 2] (2x2x1): rank 4
        # A22 @ B2b[:, 2] (2x1x1): rank 2 (but wait, B2b[:, 2] is 3x1, and A22 is 2x1, so A22 @ B2b[:, 2] doesn't work)
        
        # I made an error. Let me redo this.
        # A2 is 2x3, B2b is 3x3.
        # C2b[i,k] = sum_j A2[i,j] B2b[j,k]
        # Split j into j=0,1 and j=2:
        # C2b = A2[:, 0:2] @ B2b[0:2, :] + A2[:, 2] @ B2b[2, :]
        # A2[:, 0:2] is 2x2, B2b[0:2, :] is 2x3: 2x2x3, rank 11
        # A2[:, 2] is 2x1, B2b[2, :] is 1x3: 2x1x3, rank 6
        # Total: 11 + 6 = 17
        
        # Part 1: A2[:, 0:2] @ B2b[0:2, :], 2x2x3, rank 11
        # A2[:, 0:2]: i in {0,1}, j in {2,3} (global j indices)
        # B2b[0:2, :]: j in {2,3}, k in {2,3,4}
        # C: i in {0,1}, k in {2,3,4}
        
        # Use the 2x2x3 decomposition (rank 11):
        # Split k into {2,3} and {4}
        # k in {2,3}: 2x2x2, Strassen rank 7
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            for i in range(2):
                for j_local in range(2):
                    j = 2 + j_local
                    if U_s[i*2+j_local, c] != 0:
                        U[i*5+j, col_offset+c] = U_s[i*2+j_local, c]
            for j_local in range(2):
                j = 2 + j_local
                for k_local in range(2):
                    k = 2 + k_local
                    if V_s[j_local*2+k_local, c] != 0:
                        V[j*5+k, col_offset+c] = V_s[j_local*2+k_local, c]
            for k_local in range(2):
                k = 2 + k_local
                for i in range(2):
                    if W_s[k_local*2+i, c] != 0:
                        W[k*2+i, col_offset+c] = W_s[k_local*2+i, c]
        col_offset += 7
        
        # k in {4}: 2x2x1, rank 4
        for i in range(2):
            for j_local in range(2):
                j = 2 + j_local
                U[i*5+j, col_offset] = 1.0
                V[j*5+4, col_offset] = 1.0
                W[4*2+i, col_offset] = 1.0
                col_offset += 1
        # 4 terms
        
        # Part 2: A2[:, 2] @ B2b[2, :], 2x1x3, rank 6
        # A2[:, 2]: i in {0,1}, j=4 (global)
        # B2b[2, :]: j=4, k in {2,3,4}
        # C: i in {0,1}, k in {2,3,4}
        for i in range(2):
            for k in range(2, 5):
                U[i*5+4, col_offset] = 1.0
                V[4*5+k, col_offset] = 1.0
                W[k*2+i, col_offset] = 1.0
                col_offset += 1
        # 2*3 = 6 terms
        
        return U, V, W
    
    U_best, V_best, W_best = build_best_block_decomp()
    
    # Verify
    n, m, p = 2, 5, 5
    valid_best = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U_best.shape[1]):
                    val += U_best[i*m+j, c] * V_best[j*p+k, c] * W_best[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid_best = False
                    break
            if not valid_best:
                break
        if not valid_best:
            break
    
    if valid_best:
        print("Best block decomposition verified, rank =", U_best.shape[1])
        if U_best.shape[1] < best_rank:
            best_U, best_V, best_W = U_best, V_best, W_best
            best_rank = U_best.shape[1]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 145:
        pass
    
    # Convert to lists for the return value
    def mat_to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": mat_to_list(best_U), "V": mat_to_list(best_V), "W": mat_to_list(best_W)}
