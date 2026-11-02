import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, column] = 1
                V_sb[j * p + k, column] = 1
                W_sb[k * n + i, column] = 1
                column += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook
    
    # Strategy: Use known low-rank decompositions for smaller blocks
    # For 2x3x6, we can try to exploit structure.
    
    # Known: 2x2x2 has rank 7 (Strassen).
    # Can we build 2x3x6 from smaller blocks?
    # 2x3x6 can be seen as 2x(3x2)x(3x2)? No, dimensions don't align directly for Kronecker.
    # Let's look at the "inspiration 3" hint: "Uses two copies of a Rank 11 decomposition for 2x3x3 to construct a Rank 22 decomposition for 2x3x6".
    # This implies that if we can find a rank 11 decomposition for 2x3x3, we can combine two of them to get rank 22 for 2x3x6.
    # How? 
    # Let A be 2x3, B be 3x6.
    # Split B into two 3x3 matrices: B = [B1, B2].
    # Then C = A * B = [A*B1, A*B2].
    # If we have a rank 11 decomposition for multiplying 2x3 by 3x3, we can apply it to (A, B1) and (A, B2).
    # The rank would be 11 + 11 = 22.
    # This is much better than 36.
    
    # Do we have a rank 11 decomposition for 2x3x3?
    # Schoolbook for 2x3x3 is 18.
    # Is 11 possible?
    # Lower bound for 2x3x3: max(2*3, 3*3, 2*3) = 9.
    # Upper bound: 18.
    # Let's try to find a rank 11 decomposition for 2x3x3.
    # If we can't find one quickly, we can try rank 12, 13, etc.
    
    # Let's write a helper to verify if a set of U, V, W decomposes the tensor.
    def verify(U, V, W, n, m, p):
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i, j, k] = 1 for all i, j, k.
        # Note: The tensor for matrix multiplication C = A*B is T_{i,j,k} = 1 if we sum over j?
        # Actually, the standard tensor T such that (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} is T_{i,j,k} = 1 for all i,j,k?
        # No. The tensor T is defined on indices (i, j, k) where i is row of A, j is inner dim, k is col of B.
        # The contraction is sum_j A_{i,j} B_{j,k}.
        # This corresponds to a tensor T where T_{i,j,k} = 1 for all i,j,k.
        # Wait, if T_{i,j,k} = 1, then sum_j T_{i,j,k} A_{i,j} B_{j,k} = sum_j A_{i,j} B_{j,k}.
        # So yes, the target tensor is the all-ones tensor of shape (n, m, p).
        
        # We check if sum_r U[i, r] V[j, r] W[k, r] == 1 for all i, j, k.
        # U has shape (n*m, rank). Indexing: U[i*m + j, r]
        # V has shape (m*p, rank). Indexing: V[j*p + k, r]
        # W has shape (p*n, rank). Indexing: W[k*n + i, r]
        
        # Let's construct the tensor from U, V, W and check.
        # This is expensive for large rank, but for small search it's okay.
        
        # Efficient check:
        # For each r, we have a rank-1 tensor u_r v_r^T w_r (reshaped).
        # Sum them up.
        
        # Let's just do a direct check for small dimensions.
        # n=2, m=3, p=3.
        # 2*3*3 = 18 checks.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Let's try to find a low rank decomposition for 2x3x3.
    # We can use a simple local search or random restarts.
    # Given the 120s limit, we need to be careful.
    
    # Alternative: Use a known result or construct one.
    # Strassen for 2x2x2 is rank 7.
    # Can we embed 2x2x2 into 2x3x3?
    # Not directly.
    
    # Let's try a random search for 2x3x3 with rank 11.
    # If it fails, try 12, 13... up to 17.
    # If we find one, we can double it for 2x3x6.
    
    # However, random search in high dimensions is hard.
    # Let's try a more structured approach.
    # What if we use the Strassen algorithm for 2x2x2 and extend it?
    
    # Actually, let's look at the "Rank 11" claim.
    # Is 11 the optimal rank for 2x3x3?
    # According to some sources, the rank of 2x3x3 is 11 or 12.
    # If we can find it, we get 22 for 2x3x6.
    
    # Let's try to hardcode a known decomposition if we can find one online or derive it.
    # Since I cannot browse, I will attempt a local search for 2x3x3.
    
    # To save time, I will limit the search for 2x3x3 to a few seconds.
    # If I don't find a good one, I will return the schoolbook for 2x3x6 (rank 36).
    # Or, I can try to construct a rank 24 or so using other block decompositions.
    
    # Let's try to find a rank 12 decomposition for 2x3x3.
    
    def search_decomposition(n, m, p, target_rank, time_limit):
        start = time.time()
        # Random initialization
        # We want sum_r u_r v_r w_r = 1
        # Let's try random vectors.
        
        # To improve convergence, we can use a gradient-free method or simple random restarts.
        # Given the small size, maybe we can solve a system of linear equations if we fix some variables?
        # No, it's bilinear/trilinear.
        
        # Let's just try random restarts with small perturbations.
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for _ in range(100): # 100 restarts
            if time.time() - start > time_limit:
                break
                
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Normalize?
            
            # Simple local search
            for step in range(50):
                if time.time() - start > time_limit:
                    break
                
                # Calculate error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                            err += (s - 1.0)**2
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if err < 1e-9:
                        return U, V, W
                
                # Perturb
                U += np.random.randn(n*m, target_rank) * 0.1
                V += np.random.randn(m*p, target_rank) * 0.1
                W += np.random.randn(p*n, target_rank) * 0.1
                
        if best_err < 1e-9:
            return best_U, best_V, best_W
        return None

    # Let's try to find a decomposition for 2x3x3 with rank 11.
    # Time limit for this search: 10 seconds.
    U_233, V_233, W_233 = search_decomposition(2, 3, 3, 11, 10)
    
    if U_233 is not None:
        # We found a rank 11 decomposition for 2x3x3.
        # Now construct 2x3x6 by concatenating two such decompositions.
        # B is 3x6. Split into B1 (3x3) and B2 (3x3).
        # C = A * [B1 B2] = [A*B1 A*B2].
        # The decomposition for A*B1 uses U, V1, W1.
        # The decomposition for A*B2 uses U, V2, W2.
        # Note: U is the same for both because A is the same.
        # V1 corresponds to B1, V2 corresponds to B2.
        # W1 corresponds to the first half of C, W2 to the second half.
        
        # In the 2x3x6 tensor:
        # U has shape (2*3, rank).
        # V has shape (3*6, rank).
        # W has shape (6*2, rank).
        
        # For the first block (k=0..2):
        # V[j*6 + k, r] should match V_233[j*3 + k, r] for k<3.
        # W[k*2 + i, r] should match W_233[k*2 + i, r] for k<3.
        
        # For the second block (k=3..5):
        # Let k' = k - 3.
        # V[j*6 + k, r] should match V_233[j*3 + k', r] for k>=3.
        # W[k*2 + i, r] should match W_233[k'*2 + i, r] for k>=3.
        
        # Wait, the rank is the same for both blocks?
        # Yes, we use two copies of the rank 11 decomposition.
        # So total rank is 11 + 11 = 22.
        
        rank_236 = 22
        U_236 = np.zeros((2*3, rank_236))
        V_236 = np.zeros((3*6, rank_236))
        W_236 = np.zeros((6*2, rank_236))
        
        # Copy first block
        U_236[:, 0:11] = U_233
        for j in range(3):
            for k in range(3):
                V_236[j*6 + k, 0:11] = V_233[j*3 + k, 0:11]
        for k in range(3):
            for i in range(2):
                W_236[k*2 + i, 0:11] = W_233[k*2 + i, 0:11]
                
        # Copy second block
        U_236[:, 11:22] = U_233
        for j in range(3):
            for k in range(3):
                # k in 0..2 for the block, maps to 3..5 in the full matrix
                V_236[j*6 + (k+3), 11:22] = V_233[j*3 + k, 0:11]
        for k in range(3):
            for i in range(2):
                # k in 0..2 for the block, maps to 3..5 in the full matrix
                W_236[(k+3)*2 + i, 11:22] = W_233[k*2 + i, 0:11]
                
        # Verify
        if verify(U_236, V_236, W_236, 2, 3, 6):
            return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}
        else:
            # Fallback to schoolbook
            pass
            
    # If we didn't find a rank 11 decomposition, or verification failed,
    # try rank 12, 13 for 2x3x3?
    # Or just return schoolbook.
    
    # Let's try rank 12 for 2x3x3 quickly.
    U_233_12, V_233_12, W_233_12 = search_decomposition(2, 3, 3, 12, 5)
    if U_233_12 is not None:
        rank_236 = 24
        U_236 = np.zeros((2*3, rank_236))
        V_236 = np.zeros((3*6, rank_236))
        W_236 = np.zeros((6*2, rank_236))
        
        U_236[:, 0:12] = U_233_12
        for j in range(3):
            for k in range(3):
                V_236[j*6 + k, 0:12] = V_233_12[j*3 + k, 0:12]
        for k in range(3):
            for i in range(2):
                W_236[k*2 + i, 0:12] = W_233_12[k*2 + i, 0:12]
                
        U_236[:, 12:24] = U_233_12
        for j in range(3):
            for k in range(3):
                V_236[j*6 + (k+3), 12:24] = V_233_12[j*3 + k, 0:12]
        for k in range(3):
            for i in range(2):
                W_236[(k+3)*2 + i, 12:24] = W_233_12[k*2 + i, 0:12]
                
        if verify(U_236, V_236, W_236, 2, 3, 6):
            return {"U": U_236.tolist(), "V": V_236.tolist(), "W": W_236.tolist()}

    # Return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
