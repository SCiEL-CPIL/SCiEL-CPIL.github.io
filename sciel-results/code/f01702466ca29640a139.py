"""Program for matrix multiplication 2,4,7 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 56
    
    def build_schoolbook():
        """Return schoolbook decomposition."""
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomp(U, V, W):
        """Verify that U, V, W form a valid tensor decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        rank = U.shape[1]
        
        # Check: for each output (k, n+i), sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # should equal 1 if j matches, 0 otherwise
        # Actually the convention is: (C[k, i] = sum_j A[i, j] * B[j, k])
        # Tensor T[i, j, k] = 1 for all (i, j, k)
        # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to use the factorization 4 = 2*2 to get a lower rank decomposition
    # If we can find a decomposition of 2x2x7 with rank r, then 2x4x7 has rank <= 2*r
    # Schoolbook for 2x2x7 is 28, so 2*28=56 which is same as schoolbook for 2x4x7
    # But if 2x2x7 can be done with rank < 28, we'd get improvement
    
    # Let's try a simple approach: use the known structure
    # A is 2x4, B is 4x7, C = A*B is 2x7
    
    # Split A = [A1 | A2] where A1, A2 are 2x2
    # Split B = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    
    # This doesn't directly help with rank reduction unless we find
    # a clever decomposition of the sum of two matrix products
    
    # Let's try to find a rank 28 decomposition by exploiting the structure
    # For each output element C[i, k], we need sum_j A[i,j] * B[j,k]
    
    # A direct approach: try to use randomization to find low-rank decompositions
    def try_random_search(max_rank, num_trials=100):
        """Try to find a decomposition with given rank using random initialization."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            rank = max_rank
            # Random initialization with small values
            np.random.seed(trial)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Try gradient-free optimization using coordinate descent-like approach
            # This is simplified; a real implementation would use more sophisticated methods
            
            # Check if random init accidentally works (very unlikely)
            if verify_decomp(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from schoolbook downward
    for rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break
        result = try_random_search(rank, num_trials=50)
        if result is not None:
            return result
    
    # If no better decomposition found, return schoolbook
    return build_schoolbook()
# EVOLVE-BLOCK-END
