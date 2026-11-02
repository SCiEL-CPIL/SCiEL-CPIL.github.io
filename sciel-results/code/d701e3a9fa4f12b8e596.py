"""Matrix multiplication 4x4x6 tensor decomposition."""
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # For 4x4x4: use Strassen, rank = 7^2 = 49
    # For 4x4x2: use schoolbook, rank = 4*2 = 8
    # Total rank = 57
    
    def strassen_4x4_4x4():
        """Return U, V, W factors for 4x4x4 matrix multiplication using Strassen."""
        # Strassen for 2x2 block matrices, each block is 2x2
        # 7 multiplications, each is a 2x2 product (rank 4)
        # Total rank = 7 * 4 = 28? No, that's not how it works.
        
        # Actually, the rank of the 4x4x4 tensor is at most 49 (7^2)
        # using recursive Strassen, but the actual minimal rank is unknown.
        
        # Let me just use schoolbook for 4x4x4: rank = 64
        # And schoolbook for 4x4x2: rank = 32
        # Total: 96 (same as full schoolbook)
        
        # I need a better approach. Let me think about this differently.
        
        # The tensor T[i,j,k] for i,j in [0,3], k in [0,5]
        # I can express this as a sum of rank-1 tensors.
        
        # One approach: for each output column k, express C[:,k] = A @ B[:,k]
        # This requires rank 4 for each k (one per row of A, with B[:,k] fixed)
        # Total rank = 6 * 4 = 24
        
        # Wait, that's not right. For a fixed k, C[i,k] = sum_j A[i,j] B[j,k]
        # This is a bilinear form in A and B[:,k]. The rank is 4 (one per j).
        # So for 6 output columns, total rank = 6 * 4 = 24.
        
        # But that's not minimal either. The issue is that the B factors for different k
        # are independent, so I can't share terms across different k.
        
        # Actually, I think the rank is exactly 24 for this structure.
        # Let me verify: T[i,j,k] = 1 for all i,j,k.
        # A rank-1 tensor is u_i * v_j * w_k.
        # To get T[i,j,k] = 1, I need sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k.
        # This is equivalent to expressing the all-ones tensor as a sum of rank-1 tensors.
        # The rank of the all-ones tensor of shape (n,m,p) is max(n,m,p) if we allow
        # arbitrary coefficients, but here we need the specific structure of matrix multiplication.
        
        # For matrix multiplication, the tensor is T[i,j,k] = 1, and the rank is the
        # minimum number of rank-1 tensors needed. This is a well-studied problem.
        
        # For [n,n,n], the rank is at most n^2 (schoolbook) and at least n.
        # For [4,4,6], I believe the rank is 24 (using the "column-wise" decomposition).
        
        # Let me implement the column-wise decomposition:
        # For each output column k (0 to 5):
        #   For each row i of A (0 to 3):
        #     U[i*4 + j, col] = 1 for all j? No, that's not right.
        
        # Actually, for C[i,k] = sum_j A[i,j] B[j,k], I can write:
        # C[i,k] = sum_j (A[i,j]) * (B[j,k])
        # This is a sum of 4 rank-1 terms for each (i,k) pair, but they share the A factors.
        
        # The correct decomposition is:
        # For each j (0 to 3) and each k (0 to 5):
        #   U[i*4 + j, col] = 1 for all i? No.
        
        # Let me think again. The standard schoolbook decomposition is:
        # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # This gives rank = n*m*p = 96.
        
        # But I can do better by noting that for fixed j and k, the term A[i,j]*B[j,k]
        # contributes to all i. So I can use:
        # For each (j,k): U[i*m+j, col] = 1 for all i? No, U has shape (n*m, rank).
        
        # I think I'm confusing myself. Let me just use the schoolbook and return it.
        # It's correct, even if not optimal.
        
        pass
    
    # Just use schoolbook for now
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
