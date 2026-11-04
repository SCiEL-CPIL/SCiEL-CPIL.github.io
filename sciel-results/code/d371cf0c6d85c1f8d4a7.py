"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)  <-- Note: W index is k*n + i (output C is n x p, but trace convention says W[k*n+i, r])
    
    # Strategy: Split B into two 3x3 blocks: B = [B1 | B2]
    # C = A*B = A*B1 + A*B2
    # Each A*Bx is a 3x3 * 3x3 multiplication
    # Rank of 3x3x3 tensor is 23 (Landsberg)
    # Total rank = 23 + 23 = 46
    
    # First, build schoolbook as fallback
    rank_school = n * m * p  # 54
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find rank-46 decomposition using block structure
    # For each 3x3x3 block, we need a rank-23 decomposition
    # Let's try to find one numerically with good initialization
    
    def find_3x3x3_decomposition(target_rank, deadline):
        """Find rank-target_rank decomposition for 3x3x3 matrix multiplication."""
        nn, mm, pp = 3, 3, 3
        # U: (nn*mm, rank), V: (mm*pp, rank), W: (pp*nn, rank)
        # Equation: sum_r U[i*mm+j, r] * V[j*pp+k, r] * W[k*nn+i, r] = 1
        
        # Use structured initialization based on known low-rank decompositions
        # Start with rank 23 and try to converge
        
        for seed in range(50):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(seed)
            
            # Initialize with a mix of structured and random
            U = np.zeros((nn * mm, target_rank))
            V = np.zeros((mm * pp, target_rank))
            W = np.zeros((pp * nn, target_rank))
            
            # First few terms: use a simple pattern
            # Term 0: diagonal-like
            for r in range(min(target_rank, nn * mm * pp)):
                # Spread the identity across terms
                pass
            
            # Use random initialization with scaling
            U = np.random.randn(nn * mm, target_rank) * 0.05
            V = np.random.randn(mm * pp, target_rank) * 0.05
            W = np.random.randn(pp * nn, target_rank) * 0.05
            
            # ALS iterations
            max_iters = 2000
            prev_error = float('inf')
            stalled = 0
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Update U
                for i in range(nn):
                    for j in range(mm):
                        M = np.outer(V[j*pp:j*pp+pp, :], W[i:i+pp, :])
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(pp), rcond=None)
                            U[i * mm + j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(mm):
                    for k in range(pp):
                        M = np.outer(U[:, j], W[k*nn:(k+1)*nn, :])
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(nn), rcond=None)
                            V[j * pp + k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(pp):
                    for i in range(nn):
                        M = np.zeros((mm, target_rank))
                        for jj in range(mm):
                            M[jj, :] = U[i*mm+jj, :] * V[jj*pp+k, :]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(mm), rcond=None)
                            W[k * nn + i, :] = sol
                        except:
                            pass
                
                # Check convergence every 50 iterations
                if iteration % 50 == 0:
                    error = 0.0
                    for i in range(nn):
                        for j in range(mm):
                            for k in range(pp):
                                val = np.sum(U[i*mm+j, :] * V[j*pp+k, :] * W[k*nn+i, :])
                                error += (val - 1.0) ** 2
                    
                    if error < 1e-10:
                        return (U, V, W)
                    
                    if error > prev_error * 0.99:
                        stalled += 1
                    else:
                        stalled = 0
                    prev_error = error
                    
                    if stalled > 5:
                        break
            
            # Final check
            error = 0.0
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        val = np.sum(U[i*mm+j, :] * V[j*pp+k, :] * W[k*nn+i, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-8:
                return (U, V, W)
        
        return None
    
    # Try to find two rank-23 decompositions for the two 3x3x3 blocks
    # Block 1: k in [0,2], Block 2: k in [3,5]
    
    # For block 1 (cols 0-2 of B, cols 0-2 of C):
    # U1: (9, 23), V1: (9, 23) for k=0,1,2, W1: (9, 23) for k=0,1,2
    # For block 2 (cols 3-5 of B, cols 3-5 of C):
    # U2: (9, 23), V2: (9, 23) for k=3,4,5, W2: (9, 23) for k=3,4,5
    
    # But we can share U between blocks since A is the same!
    # Actually, we can use the same U for both blocks.
    # So total rank = 23 + 23 = 46, but U is shared.
    
    # Let's try: use one U of rank 46, with first 23 cols for block 1, next 23 for block 2
    # V: first 23 cols have nonzeros only in rows j*3+k for k in [0,2]
    #     next 23 cols have nonzeros only in rows j*3+k for k in [3,5]
    # W: first 23 cols have nonzeros only in rows k*3+i for k in [0,2]
    #     next 23 cols have nonzeros only in rows k*3+i for k in [3,5]
    
    target_rank = 46
    U_full = np.zeros((n * m, target_rank))
    V_full = np.zeros((m * p, target_rank))
    W_full = np.zeros((p * n, target_rank))
    
    # Find decomposition for block 1
    decomp1 = find_3x3x3_decomposition(23, deadline - 20)
    
    if decomp1 is not None:
        U1, V1, W1 = decomp1
        # Copy to first 23 columns
        U_full[:, :23] = U1
        # V1 has shape (9, 23), rows correspond to j*3+k for j in [0,2], k in [0,2]
        # In V_full, rows j*3+k for k in [0,2] are the same indices
        V_full[:, :23] = V1
        # W1 has shape (9, 23), rows correspond to k*3+i for k in [0,2]
        # In W_full, rows k*3+i for k in [0,2] are indices 0-8
        W_full[:, :23] = W1
    
    # Find decomposition for block 2
    decomp2 = find_3x3x3_decomposition(23, deadline - 10)
    
    if decomp2 is not None:
        U2, V2, W2 = decomp2
        # Copy to next 23 columns
        U_full[:, 23:46] = U2
        # V2 has shape (9, 23), rows correspond to j*3+k for j in [0,2], k in [0,2]
        # In V_full, we need to map k in [0,2] to k in [3,5]
        # Row index in V2: j*3 + k (k in 0,1,2)
        # Row index in V_full: j*3 + (k+3)
        for j in range(3):
            for k in range(3):
                V_full[j*3 + k + 3, 23:46] = V2[j*3 + k, :]
        # W2 has shape (9, 23), rows correspond to k*3+i for k in [0,2]
        # In W_full, we need to map k in [0,2] to k in [3,5]
        # Row index in W2: k*3 + i (k in 0,1,2)
        # Row index in W_full: (k+3)*3 + i
        for k in range(3):
            for i in range(3):
                W_full[(k+3)*3 + i, 23:46] = W2[k*3 + i, :]
    
    # Verify the full decomposition
    if decomp1 is not None and decomp2 is not None:
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_full[i*m+j, :] * V_full[j*p+k, :] * W_full[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        if error < 1e-8:
            best_U, best_V, best_W = U_full, V_full, W_full
            best_rank = target_rank
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # If block approach failed, try direct ALS on full problem with rank 46
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(seed + 1000)
        U = np.random.randn(n * m, 46) * 0.05
        V = np.random.randn(m * p, 46) * 0.05
        W = np.random.randn(p * n, 46) * 0.05
        
        max_iters = 1000
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.outer(V[j*p:j*p+p, :], W[i:i+p, :])
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        U[i * m + j, :] = sol
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.outer(U[:, j], W[k*n:(k+1)*n, :])
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        V[j * p + k, :] = sol
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, 46))
                    for jj in range(m):
                        M[jj, :] = U[i*m+jj, :] * V[jj*p+k, :]
                    try:
                        sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        W[k * n + i, :] = sol
                    except:
                        pass
            
            if iteration % 100 == 0:
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0) ** 2
                if error < 1e-8:
                    best_U, best_V, best_W = U, V, W
                    best_rank = 46
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
