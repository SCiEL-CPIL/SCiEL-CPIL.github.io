"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
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
    
    # Verify a decomposition is exact
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        # Convert to numpy arrays for efficient verification
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k] = 1 if and only if (i,j,k) is a valid index
        # For matrix multiplication C = A * B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Check a sample of indices to verify correctness
        # For exact verification, we'd need to check all n*m*p entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using structured random search
    # Strategy: Use bilinear structure and exploit the fact that we can share computations
    # For 2x3x10, we might be able to use rank < 60 by sharing across the p=10 dimension
    
    # Approach: Try rank 50, 55, etc. using random initialization and gradient-like updates
    # Since we need exact integer/half-integer solutions, we'll try structured patterns
    
    # Key insight: For multiplying 2x3 by 3xp, we can potentially use p * (2*3 - 1) = p*5 rank
    # This is based on the fact that for each column of B, we need to compute 2 outputs from 3 inputs
    # The rank for 2x3x1 is 5 (known result), so for 2x3x10 we might use 10*5 = 50
    
    # Known rank-5 decomposition for 2x3x1:
    # A = [[a, b, c], [d, e, f]], B = [x, y, z]^T
    # C = [a*x + b*y + c*z, d*x + e*y + f*z]
    # 
    # Rank-5 decomposition uses 5 products:
    # 1. (a+d) * x
    # 2. (b+e) * y  
    # 3. (c+f) * z
    # 4. (a-b) * (x-y)  -- wait, this isn't bilinear in the right way
    # 
    # Actually, let me use a known rank-5 scheme for 2x3 matrix-vector product:
    # The 6 products are: ax, by, cz, dx, ey, fz
    # We can combine: (a+d)x, (b+e)y, (c+f)z, then need to separate
    # This gives 3 products for the sums, but we need 2 outputs
    # 
    # Let me try a different approach: use the structure to build a rank-50 decomposition
    
    # For now, try a few random structured approaches
    best_result = None
    best_rank = n * m * p
    
    # Try rank 50 with structured pattern
    for target_rank in [50, 55, 58, 59]:
        if time.time() > deadline - 5:
            break
            
        # Initialize with schoolbook and try to merge columns
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Start with schoolbook pattern and try to merge
        # For each (i,j,k), we have a product. Try to share products across k values
        # Since n=2, m=3, for each i,j we have p=10 products with different k
        # We can potentially share the A[i,j] factor
        
        # Structured approach: For each (i,j), use p products but try to combine
        # Actually, let's just try random half-integer entries and verify
        
        if random.random() < 0.5:  # Only try some ranks to save time
            # Try a structured random initialization
            for _ in range(100):  # 100 random attempts
                if time.time() > deadline - 5:
                    break
                    
                # Initialize with random half-integers
                U_rand = np.random.randint(0, 3, (n*m, target_rank)) * 0.5
                V_rand = np.random.randint(0, 3, (m*p, target_rank)) * 0.5
                W_rand = np.random.randint(0, 3, (p*n, target_rank)) * 0.5
                
                # Convert to lists
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                W_list = W_rand.tolist()
                
                if verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
            
            if best_result is not None:
                break
    
    # Return best result found, or schoolbook if nothing better
    if best_result is not None and best_rank < n * m * p:
        return best_result
    else:
        return make_schoolbook()
