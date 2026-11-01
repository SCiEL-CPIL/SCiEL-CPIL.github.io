import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Construct the rank-49 decomposition using Strassen's algorithm
    # This is a known optimal or near-optimal construction for 2x2x2 blocks.
    result = _construct_strassen_decomposition(n, m, p)
    
    # Verify the decomposition with a strict deadline
    if result and _verify_decomposition(result, n, m, p, deadline):
        return result
    
    # Fallback to schoolbook if verification fails or timeout
    # Schoolbook rank is n*m*p = 56
    return _construct_schoolbook(n, m, p)


def _construct_strassen_decomposition(n, m, p):
    """
    Constructs a rank-49 decomposition for 2x2x14 matrix multiplication.
    Uses 7 Strassen products for each of the 7 2x2 blocks of B.
    """
    if n != 2 or m != 2 or p % 2 != 0:
        return None
        
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank) -> (4, rank)
    # V shape: (m*p, rank) -> (28, rank)
    # W shape: (p*n, rank) -> (28, rank)
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    
    # Strassen's 7 products for 2x2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    
    # Indices for A (flattened row-major):
    # a=0, b=1, c=2, d=3
    
    for b in range(num_blocks):
        base_col = b * 7
        # B row indices for this block: 2b, 2b+1
        # B col indices for this block: 2b, 2b+1
        # V index: j*p + k. Here j is 0 or 1, k is 2b or 2b+1.
        # V[0, 2b] -> 2b
        # V[0, 2b+1] -> 2b+1
        # V[1, 2b] -> 2+2b
        # V[1, 2b+1] -> 2+2b+1 = 2b+3
        
        # W indices: C[k, i] -> k*n + i. n=2.
        # C[2b, 0] -> 4b
        # C[2b, 1] -> 4b+1
        # C[2b+1, 0] -> 4b+2
        # C[2b+1, 1] -> 4b+3
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,2b] + B[1,2b+1])
        U[0][base_col] += 1
        U[3][base_col] += 1
        V[2 * b][base_col] += 1
        V[2 * b + 3][base_col] += 1
        # Contribution to C[2b,0] and C[2b+1,1]
        W[4 * b][base_col] += 1
        W[4 * b + 3][base_col] += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,2b]
        U[2][base_col + 1] += 1
        U[3][base_col + 1] += 1
        V[2 * b][base_col + 1] += 1
        # Contribution to C[2b,1] and C[2b+1,1]
        W[4 * b + 1][base_col + 1] += 1
        W[4 * b + 3][base_col + 1] += -1
        
        # M3 = A[0,0] * (B[0,2b+1] - B[1,2b+1])
        U[0][base_col + 2] += 1
        V[2 * b + 1][base_col + 2] += 1
        V[2 * b + 3][base_col + 2] += -1
        # Contribution to C[2b,0] and C[2b+1,0]
        W[4 * b][base_col + 2] += 1
        W[4 * b + 2][base_col + 2] += 1
        
        # M4 = A[1,1] * (B[1,2b] - B[0,2b])
        U[3][base_col + 3] += 1
        V[2 * b + 2][base_col + 3] += 1
        V[2 * b][base_col + 3] += -1
        # Contribution to C[2b,0] and C[2b+1,0]
        W[4 * b][base_col + 3] += 1
        W[4 * b + 2][base_col + 3] += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,2b+1]
        U[0][base_col + 4] += 1
        U[1][base_col + 4] += 1
        V[2 * b + 3][base_col + 4] += 1
        # Contribution to C[2b,0] and C[2b+1,0]
        W[4 * b][base_col + 4] += -1
        W[4 * b + 2][base_col + 4] += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,2b] + B[0,2b+1])
        U[2][base_col + 5] += 1
        U[0][base_col + 5] += -1
        V[2 * b][base_col + 5] += 1
        V[2 * b + 1][base_col + 5] += 1
        # Contribution to C[2b+1,1]
        W[4 * b + 3][base_col + 5] += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,2b] + B[1,2b+1])
        U[1][base_col + 6] += 1
        U[3][base_col + 6] += -1
        V[2 * b + 2][base_col + 6] += 1
        V[2 * b + 3][base_col + 6] += 1
        # Contribution to C[2b,0]
        W[4 * b][base_col + 6] += 1
        
    # Convert to lists of lists as required by interface?
    # The interface says "factor matrices of shapes ... with integer/half-integer entries".
    # It doesn't explicitly say numpy arrays, but the seed returned lists.
    # "return {'U': U, 'V': V, 'W': W}"
    # I will return numpy arrays to be safe, or lists. The seed used lists.
    # The evaluator likely accepts numpy arrays or lists.
    # I'll return numpy arrays for efficiency in verification.
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(decomp, n, m, p, deadline, num_tests=10):
    """
    Verifies the decomposition by checking C = A @ B against the tensor product.
    Uses vectorized NumPy operations for speed.
    """
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    
    rank = U.shape[1]
    
    # Reshape for broadcasting
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    for _ in range(num_tests):
        if time.time() > deadline:
            return False
            
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k])
        
        # Efficient calculation:
        # 1. Compute intermediate tensors
        # A_U = A[i,j] * U[i,j,r] -> (n, m, rank)
        A_U = A[:, :, np.newaxis] * U_reshaped
        # B_V = B[j,k] * V[j,k,r] -> (m, p, rank)
        B_V = B[:, :, np.newaxis] * V_reshaped
        
        # 2. Contract over j: sum_j A_U[i,j,r] * B_V[j,k,r]
        # A_U: (n, m, rank), B_V: (m, p, rank)
        # We want result[i,k,r] = sum_j A_U[i,j,r] * B_V[j,k,r]
        # A_U_batch = (rank, n, m)
        # B_V_batch = (rank, m, p)
        A_U_batch = A_U.transpose(2, 0, 1)  # (rank, n, m)
        B_V_batch = B_V.transpose(2, 0, 1)  # (rank, m, p)
        
        # Matrix multiply: (rank, n, p)
        Intermediate = np.matmul(A_U_batch, B_V_batch) # (rank, n, p)
        
        # 3. Multiply by W and sum over r
        # W_reshaped: (p, n, rank)
        # W_batch = (rank, p, n)
        W_batch = W_reshaped.transpose(2, 0, 1) # (rank, p, n)
        
        # Element-wise multiply: (rank, p, n) * (rank, n, p)
        # Intermediate is (rank, n, p). W_batch is (rank, p, n).
        # We need Intermediate[r, i, k] * W_batch[r, k, i]
        # W_batch.transpose(0, 2, 1) is (rank, n, p)
        C_computed = np.sum(Intermediate * W_batch.transpose(0, 2, 1), axis=0)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True
# EVOLVE-BLOCK-END
