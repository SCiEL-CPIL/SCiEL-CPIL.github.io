"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin within 150s limit
    
    n, m, p = DIMENSIONS
    
    # Matrix multiplication tensor: T[i,j,k] = 1 for all i,j,k
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Trace convention: W[k*n + i, r]
    
    # Strategy: Decompose 2x3 * 3x8 as:
    # - Split A (2x3) into two blocks: A1 (2x2) and A2 (2x1)
    # - Split B (3x8) into two blocks: B1 (2x8) and B2 (1x8)
    # - C = A1*B1 + A2*B2
    # - Use Strassen for 2x2 * 2x8 (rank 7*8/2 = 28... wait, let me think more carefully)
    #
    # Actually, let's use a different approach:
    # The 2x3 * 3x8 multiplication can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can split j into two groups: j=0,1 (2 elements) and j=2 (1 element)
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] + A[i,2] * B[2,k]
    #
    # The first term is a 2x2 * 2x8 multiplication, which can be done with rank 7*4=28 using Strassen-like methods
    # Actually, for 2x2 * 2x8, we can use Strassen on the 2x2 part and schoolbook on the 8 columns
    # This gives rank 7*4 = 28 (Strassen for 2x2*2x2 is rank 7, times 4 for the 4 groups of 2 columns)
    # Wait, that's not right either.
    #
    # Let me think about this differently. The known result from the inspirations is rank 44.
    # Let's construct that decomposition.
    #
    # For 2x3 * 3x8:
    # We can write this as a block multiplication:
    # A = [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # B = [B1; B2] where B1 is 2x8 and B2 is 1x8
    # C = A1*B1 + A2*B2
    #
    # A1*B1 is 2x2 * 2x8: This can be done with rank 7*4 = 28 (using Strassen for each pair of columns)
    # Actually, for 2x2 * 2x8, we can use the Strassen algorithm which has rank 7 for 2x2*2x2,
    # and since we have 4 independent 2x2*2x2 subproblems (for each pair of columns), the total rank is 7*4 = 28.
    #
    # A2*B2 is 2x1 * 1x8: This is just outer product, rank 2*8 = 16... wait, no.
    # A2 is 2x1, B2 is 1x8. C2 = A2*B2 is 2x8.
    # C2[i,k] = A2[i,0] * B2[0,k]
    # This is an outer product, which has tensor rank 1! No wait, that's not right either.
    # Actually, for a 2x1 * 1x8 multiplication, the tensor rank is 1 because it's just an outer product.
    # But we need to be careful about the index structure.
    #
    # Hmm, let me reconsider. Let me just construct the rank-44 decomposition directly.
    #
    # Actually, let's use a simpler approach: use Strassen for the 2x2 part and schoolbook for the rest.
    #
    # Split the 3 middle dimension into 2+1:
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # B = [B_left; B_right] where B_left is 2x8, B_right is 1x8
    #
    # C = A_left * B_left + A_right * B_right
    #
    # For A_left * B_left (2x2 * 2x8):
    # We can use Strassen's algorithm. Strassen decomposes 2x2 * 2x2 into 7 multiplications.
    # For 2x2 * 2x8, we have 4 independent 2x2 * 2x2 subproblems (columns 0-1, 2-3, 4-5, 6-7).
    # So the rank for this part is 7 * 4 = 28.
    #
    # For A_right * B_right (2x1 * 1x8):
    # This is C_right[i,k] = A_right[i,0] * B_right[0,k]
    # This is an outer product. The tensor rank is 1 (it's a rank-1 tensor).
    # Wait, but we need to check the index structure.
    # U has shape (n*m, rank) = (2*3, rank) = (6, rank)
    # V has shape (m*p, rank) = (3*8, rank) = (24, rank)
    # W has shape (p*n, rank) = (8*2, rank) = (16, rank)
    #
    # For the outer product part:
    # U[i*m + j, r] where i in {0,1}, j=2 (the right part), so rows 2, 3 of U
    # V[j*p + k, r] where j=2, k in {0,...,7}, so rows 16-23 of V
    # W[k*n + i, r] where k in {0,...,7}, i in {0,1}, so all 16 rows of W
    #
    # For the outer product: U[2+i, r] * V[16+k, r] * W[k*2+i, r] = A_right[i,0] * B_right[0,k]
    # We can set r=0: U[2, 0]=1, U[3, 0]=1, V[16+k, 0]=1 for all k, W[k*2+i, 0] = 1 for all k,i
    # Then sum_r U[2+i, r] * V[16+k, r] * W[k*2+i, r] = 1 * 1 * 1 = 1
    # But we need this to equal A_right[i,0] * B_right[0,k], not 1.
    #
    # I think I'm overcomplicating this. Let me just use the known rank-44 construction.
    #
    # Actually, let me try a different approach: use the fact that the tensor rank of 
    # 2x3 * 3x8 is at most 44, and construct it explicitly.
    
    # Let's use the following construction:
    # Split the middle dimension (size 3) into 2+1
    # A = [A1 | A2], B = [B1; B2]
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x8): Use Strassen-like decomposition
    # Strassen for 2x2*2x2 has rank 7. For 2x2*2x8, we have 4 groups of 2 columns,
    # so rank = 7 * 4 = 28
    
    # For A2*B2 (2x1 * 1x8): This is an outer product, rank = 1
    # Wait, that gives total rank 29, which seems too low. Let me reconsider.
    
    # Actually, the issue is that the index structures don't align simply.
    # Let me just construct a valid decomposition with rank 44 as suggested by the inspirations.
    
    # Construct rank-44 decomposition:
    # We'll use a combination of Strassen and schoolbook terms.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for 2x2 * 2x8 (rank 28)
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 2x2 * 2x8, we apply this to each pair of columns (0-1, 2-3, 4-5, 6-7)
    # So we have 4 * 7 = 28 terms
    
    col = 0
    for col_pair in range(4):  # 4 pairs of columns
        k1 = col_pair * 2
        k2 = col_pair * 2 + 1
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11 = A[0,0], A22 = A[1,1]
        # B11 = B[0,k1], B22 = B[1,k1]
        U[0 * m + 0, col] = 1.0  # A[0,0]
        U[1 * m + 1, col] = 1.0  # A[1,1]
        V[0 * p + k1, col] = 1.0  # B[0,k1]
        V[1 * p + k1, col] = 1.0  # B[1,k1]
        W[k1 * n + 0, col] = 1.0  # C[0,k1]
        W[k1 * n + 1, col] = 1.0  # C[1,k1]
        W[k2 * n + 0, col] = 1.0  # C[0,k2]
        W[k2 * n + 1, col] = 1.0  # C[1,k2]
        col += 1
        
        # M2 = (A21 + A22) * B11
        # A21 = A[1,0], A22 = A[1,1]
        # B11 = B[0,k1]
        U[1 * m + 0, col] = 1.0  # A[1,0]
        U[1 * m + 1, col] = 1.0  # A[1,1]
        V[0 * p + k1, col] = 1.0  # B[0,k1]
        W[k1 * n + 1, col] = 1.0  # C[1,k1]
        col += 1
        
        # M3 = A11 * (B12 - B22)
        # A11 = A[0,0]
        # B12 = B[0,k2], B22 = B[1,k2]
        U[0 * m + 0, col] = 1.0  # A[0,0]
        V[0 * p + k2, col] = 1.0  # B[0,k2]
        V[1 * p + k2, col] = -1.0  # -B[1,k2]
        W[k2 * n + 0, col] = 1.0  # C[0,k2]
        col += 1
        
        # M4 = A22 * (B21 - B11)
        # A22 = A[1,1]
        # B21 = B[1,k1], B11 = B[0,k1]
        U[1 * m + 1, col] = 1.0  # A[1,1]
        V[1 * p + k1, col] = 1.0  # B[1,k1]
        V[0 * p + k1, col] = -1.0  # -B[0,k1]
        W[k1 * n + 1, col] = 1.0  # C[1,k1]
        col += 1
        
        # M5 = (A11 + A12) * B22
        # A11 = A[0,0], A12 = A[0,1]
        # B22 = B[1,k2]
        U[0 * m + 0, col] = 1.0  # A[0,0]
        U[0 * m + 1, col] = 1.0  # A[0,1]
        V[1 * p + k2, col] = 1.0  # B[1,k2]
        W[k2 * n + 0, col] = 1.0  # C[0,k2]
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        # A21 = A[1,0], A11 = A[0,0]
        # B11 = B[0,k1], B12 = B[0,k2]
        U[1 * m + 0, col] = 1.0  # A[1,0]
        U[0 * m + 0, col] = -1.0  # -A[0,0]
        V[0 * p + k1, col] = 1.0  # B[0,k1]
        V[0 * p + k2, col] = 1.0  # B[0,k2]
        W[k1 * n + 1, col] = 1.0  # C[1,k1]
        W[k2 * n + 1, col] = 1.0  # C[1,k2]
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A12 = A[0,1], A22 = A[1,1]
        # B21 = B[1,k1], B22 = B[1,k2]
        U[0 * m + 1, col] = 1.0  # A[0,1]
        U[1 * m + 1, col] = -1.0  # -A[1,1]
        V[1 * p + k1, col] = 1.0  # B[1,k1]
        V[1 * p + k2, col] = 1.0  # B[1,k2]
        W[k1 * n + 1, col] = 1.0  # C[1,k1]
        W[k2 * n + 1, col] = 1.0  # C[1,k2]
        col += 1
    
    # Part 2: Schoolbook for 2x1 * 1x8 (rank 16)
    # A_right is 2x1 (column 2 of A), B_right is 1x8 (row 2 of B)
    # C_right[i,k] = A[i,2] * B[2,k]
    # For each (i,k) pair, we need one term
    for i in range(n):
        for k in range(p):
            U[i * m + 2, col] = 1.0  # A[i,2]
            V[2 * p + k, col] = 1.0  # B[2,k]
            W[k * n + i, col] = 1.0  # C[i,k]
            col += 1
    
    # Total rank should be 28 + 16 = 44
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n * m, rank_schoolbook))
        V_sb = np.zeros((m * p, rank_schoolbook))
        W_sb = np.zeros((p * n, rank_schoolbook))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
