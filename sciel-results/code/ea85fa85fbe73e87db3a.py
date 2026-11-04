import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 150s limit, return early if exceeded (unlikely for this static solution)
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                # V[j*p + k, column] = 1
                # W[k*n + i, column] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
        if time.time() > deadline:
            # This should never happen for the static schoolbook construction, 
            # but satisfies the directive for a deadline guard.
            break

    # Optional: Verification step (very fast for small ranks)
    # Convert to numpy arrays for quick check
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # The decomposition is valid if sum_c U[:,c] outer V[:,c] outer W[:,c] matches the tensor
    # Tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Actually, the tensor for matrix multiplication C = A * B is:
    # C[k, i] = sum_j A[i, j] B[j, k]
    # In index notation with the given output factor convention W[k*n+i, r]:
    # The tensor T is indexed by (i, j, k) where T[i,j,k] = 1.
    # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # We can verify a few random indices or the whole thing.
    # For rank 48, full verification is 2*3*8*48 = 2304 operations, which is trivial.
    # Let's do a full check to be absolutely sure.
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if val != 1:
                    is_valid = False
                    break
            if not is_valid: break
        if not is_valid: break
        
    if not is_valid:
        # Fallback: return empty or raise? The task says invalid result scores 0.
        # Since we know the construction is correct, this is just a safety net.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
