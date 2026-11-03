"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Matrix multiplication tensor for M_{n,m,p}:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid (i,j,k)
    # Decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index conventions:
    #   U row index: i*m + j
    #   V row index: j*p + k
    #   W row index: k*n + i
    
    # Build a structured decomposition using block decomposition:
    # A is 3x3, B is 3x6 (two 3x3 blocks), C is 3x6
    # Split A into 2x2 + cross terms, B into two 3x3 blocks
    
    # Strategy: Use Strassen for 2x2*2x2 (rank 7), and schoolbook for remaining terms
    # A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    # B = [[B00, B01, B02, B03, B04, B05],
    #      [B10, B11, B12, B13, B14, B15],
    #      [B20, B21, B22, B23, B24, B25]]
    # C = [[C00, C01, C02, C03, C04, C05],
    #      [C10, C11, C12, C13, C14, C15],
    #      [C20, C21, C22, C23, C24, C25]]
    
    # Let's use a block approach:
    # Split A into top-left 2x2 block and the rest
    # Split B into two 3x3 blocks (columns 0-2 and 3-5)
    
    # For each 3x3 block of B, compute A * B_block (3x3 result)
    # We can use rank-23 decomposition for 3x3*3x3 if we had one, but let's build from scratch
    
    # Actually, let's try a simpler approach: direct construction with rank minimization
    # using the fact that we can share computations.
    
    # Let's build the schoolbook decomposition first (rank 54)
    def schoolbook_decomp():
        rank = n * m * p
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
        return U, V, W, rank
    
    # Verify function
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a better decomposition using structured blocks
    # Key insight: B is 3x6 = two 3x3 blocks
    # If we can compute A*B1 and A*B2 sharing some intermediate products, we save rank
    
    # Let's try: compute all products A[i,j]*B[j,k] but group them
    # For a fixed j, A[i,j] is shared across all k
    # For a fixed (i,k), we sum over j
    
    # Alternative: use the fact that p=6 and try to find linear dependencies
    
    # Let's try a randomized search with verification
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook_decomp()
    
    # Try various candidate ranks
    for target_rank in [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute error tensor
                error = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            error[i,j,k] = 1.0 - val
                
                max_err = np.max(np.abs(error))
                if max_err < 1e-10:
                    # Found valid decomposition
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    found = True
                    break
                
                # Gradient descent on the loss function
                # L = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # dL/dU[i*m+j,r] = 2*error[i,j,k]*V[j*p+k,r]*W[k*n+i,r]
                # etc.
                
                eta = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = error[i,j,k]
                            if abs(err) < 1e-12:
                                continue
                            for r in range(target_rank):
                                u_val = U[i*m+j, r]
                                v_val = V[j*p+k, r]
                                w_val = W[k*n+i, r]
                                U[i*m+j, r] -= eta * 2 * err * v_val * w_val
                                V[j*p+k, r] -= eta * 2 * err * u_val * w_val
                                W[k*n+i, r] -= eta * 2 * err * u_val * v_val
                
                if it % 50 == 0:
                    # Periodic check
                    pass
        
        if found:
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
