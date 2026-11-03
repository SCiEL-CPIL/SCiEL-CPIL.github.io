import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    # Known rank-17 decomposition for 4x4x4 matrix multiplication
    # We embed 2x4x6 into 4x4x4 by padding, then prune
    # The 17-term decomposition for 4x4x4 is well-known (Strassen-like)
    #
    # For a 4x4 matrix multiplication C = A*B where A is 4x4, B is 4x4:
    # The 17 terms can be constructed. But our problem is 2x4 * 4x6 -> 2x6
    # 
    # Let's use a direct approach: try to find a low-rank decomposition
    # using the structure of the problem.
    #
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # We need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    #
    # Let's try a parametric family. Since n=2 is small, we can try to
    # use the fact that we only have 2 output rows.
    #
    # Strategy: Use the known rank-17 for 4x4x4 and embed.
    # Actually, let me try a direct numerical search with better structure.
    #
    # For the 2x4x6 problem, the minimum rank is at most 2*4*6=48 (schoolbook)
    # and at least max(2*4, 4*6, 2*6) = max(8,24,12) = 24 (by trivial bounds)
    # Actually the better lower bound is: rank >= n*p = 12 (from output dimension)
    # and rank >= m*max(n,p) ... no, that's not right.
    # The rank is at least max(n*p, m*p, n*m) = max(12, 24, 8) = 24? No.
    # The rank of the tensor is at least the "slice rank" in each direction.
    # Slice rank in i-direction: for each i, we have a 4x6 slice, rank <= 4, so slice rank >= 4*2=8? No.
    # Actually the rank of a tensor is at least the maximum over all slices.
    # 
    # Let me just do a numerical search for ranks 20-30.
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for rank in range(24, 14, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 7919 + rank * 31)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Alternating least squares with better convergence
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve min ||U[i*m+j,:] * (V*W)^T - ones||
                # For fixed V, W: for each i, j:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                # Let M[i,j] be a rank x p matrix where M[i,j,k,r] = V[j*p+k,r]*W[k*n+i,r]
                # Then we need U[i*m+j,:] @ M[i,j].T = ones(p)
                # This is an underdetermined system (rank unknowns, p equations)
                
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                        # Coeff matrix: coeff[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        coeff = np.zeros((p, rank))
                        for k in range(p):
                            coeff[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve: U[i*m+j,:] @ coeff.T = ones(p)
                        # Minimum norm solution: U_row = ones(p) @ pinv(coeff.T)
                        # = coeff @ pinv(coeff.T @ coeff) @ ones ... 
                        # Actually: minimize ||coeff.T @ U_row - ones|| 
                        # U_row = coeff @ (coeff.T @ coeff)^{-1} @ ones if invertible
                        # Or use lstsq
                        try:
                            U_row, _, _, _ = np.linalg.lstsq(coeff.T, np.ones(p), rcond=None)
                        except:
                            U_row = np.zeros(rank)
                        U_new[i*m+j, :] = U_row
                
                # Update V: for each (j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        coeff = np.zeros((n, rank))
                        for i in range(n):
                            coeff[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V_row, _, _, _ = np.linalg.lstsq(coeff.T, np.ones(n), rcond=None)
                        except:
                            V_row = np.zeros(rank)
                        V_new[j*p+k, :] = V_row
                
                # Update W: for each (k,i), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        coeff = np.zeros((m, rank))
                        for j in range(m):
                            coeff[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W_row, _, _, _ = np.linalg.lstsq(coeff.T, np.ones(m), rcond=None)
                        except:
                            W_row = np.zeros(rank)
                        W_new[k*n+i, :] = W_row
                
                diff = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
                U, V, W = U_new, V_new, W_new
                
                if diff < 1e-12:
                    break
            
            # Check error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: rank-24 decomposition using block structure
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split j into two groups of 2: j=0,1 and j=2,3
    # C[i,k] = sum_{j=0}^1 A[i,j]B[j,k] + sum_{j=2}^3 A[i,j]B[j,k]
    # First sum: 2x2 * 2x6, rank at most 2*2*6=24 but actually for 2x2*2x6 we can do better
    # Actually 2x2 matrix mult has rank 7 (Strassen), and we have 3 different k-blocks
    # Let's just use schoolbook: rank = 2*4*6 = 48
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
