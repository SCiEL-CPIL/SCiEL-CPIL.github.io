"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 56
    
    # Known lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Strassen-like or other decompositions may exist below 56
    # For 2x4 by 4x7, the tensor rank is at most 56 (schoolbook)
    # Let's try to find a better decomposition
    
    # The matrix multiplication tensor T_{i,j,k} = A_{i,j} * B_{j,k}
    # We want: T = sum_{r=1}^{R} U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # This means: C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{r} W_{k*n+i, r} * (sum_j U_{i*m+j,r} * V_{j*p+k,r} ... )
    
    # Actually, the standard form is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor: T_{(i,j),(j,k),(k,i)} where first index is (i,j) in [0, n*m), second is (j,k) in [0, m*p), third is (k,i) in [0, p*n)
    # T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(k,i) for consistent i,j,k, else 0
    
    # We seek rank-R decomposition:
    # T_{a,b,c} = sum_{r=1}^{R} U_{a,r} V_{b,r} W_{c,r}
    
    # Let's use a randomized approach to try to find low-rank decompositions
    # Start with schoolbook and try to reduce rank
    
    # For small matrices, we can try various known decompositions
    # 2x2 by 2x2 has rank 7 (Strassen)
    # 2x4 by 4x7: we can think of it as block operations
    
    # One approach: use the fact that 2x4 * 4x7 can be decomposed
    # Let A be 2x4, B be 4x7
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    
    # Each 2x2 * 2x7 can be done... 
    # 2x2 * 2x2 has rank 7, so 2x2 * 2x7 = (2x2*2x2) + (2x2*2x2) + (2x2*2x2) + (2x2*2x2) = 4 blocks? No.
    # 2x2 * 2x7: result is 2x7. We can view B as four 2x2 blocks? No, 2x7 isn't divisible by 2 in the second dim.
    
    # Alternative: just return schoolbook for now, but try a few random initializations
    # to see if we can find a lower rank via iterative refinement
    
    # Actually, let's think about known results.
    # The rank of M_{2,4,7} is at most 56.
    # Can we do better?
    
    # A simple improvement: if one dimension is 1, rank = product of others.
    # Here all dims > 1.
    
    # Let's try a greedy/iterative approach:
    # Start with R = 28 (lower bound) and try to solve the system
    # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    
    # This is a nonlinear system. We can use alternating least squares:
    # Fix V, W: solve for U (linear)
    # Fix U, W: solve for V (linear)  
    # Fix U, V: solve for W (linear)
    
    # Let's implement ALS for tensor decomposition
    
    def try_als(rank, max_iters=100, tol=1e-10):
        """Try to find rank-R decomposition via alternating least squares."""
        if time.time() > deadline - 5:
            return None
        
        # Build the target tensor
        # T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
        # T[a,b,c] = 1 iff a=(i,j), b=(j',k), c=(k',i') with i=i', j=j', k=k'
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Initialize with random values
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each a, T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a] = U[a,:] @ (V^T * W) ... 
            # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Let M_r = V[:,r] * W[:,r] (element-wise product, shape (m*p, p*n))
            # T[a] = sum_r U[a,r] * M_r
            # This is a linear system: T[a] = U[a,:] @ M where M has columns M_r
            # M shape: (m*p * p*n, rank)
            
            # Vectorize T[a] to shape (m*p * p*n,)
            # M[:,r] = M_r.reshape(-1)
            
            # Build M once
            M = np.zeros((m*p * p*n, rank), dtype=np.float64)
            for r in range(rank):
                M[r::rank] = (V[:, r] * W[:, r]).reshape(-1)
            
            # For each a, solve U[a,:] to minimize ||T[a].reshape(-1) - U[a,:] @ M||^2
            # U[a,:] = T[a].reshape(-1) @ M^+ (pseudo-inverse)
            # Or use least squares
            
            T_flat = T.reshape(n*m, -1)  # shape (n*m, m*p*p*n)
            
            # Solve U @ M^T = T_flat^T, i.e., U = T_flat @ (M^T)^+
            # M shape: (m*p*p*n, rank), so M^T shape: (rank, m*p*p*n)
            # We want U (n*m, rank) such that U @ M^T ≈ T_flat (n*m, m*p*p*n)
            # U = T_flat @ pinv(M^T)
            
            try:
                U_new = T_flat @ np.linalg.pinv(M.T)
            except:
                return None
            
            # Update V: for each b, T[:,b,:] = sum_r V[b,r] * (U[:,r] outer W[:,r])
            # T[a,b,c] = sum_r V[b,r] U[a,r] W[c,r]
            # For fixed b: T[:,b,:] = V[b,:] @ N where N[:,r] = U[:,r] * W[:,r]
            N = np.zeros((n*m * p*n, rank), dtype=np.float64)
            for r in range(rank):
                N[r::rank] = (U[:, r] * W[:, r]).reshape(-1)
            
            # T[:,b,:] shape: (n*m, p*n), flatten to (n*m*p*n,)
            # We want V[b,:] (1, rank) such that V[b,:] @ N^T ≈ T[:,b,:].reshape(-1)
            # V[b,:] = T[:,b,:].reshape(-1) @ pinv(N^T)
            
            T_for_V = np.zeros((m*p, n*m * p*n), dtype=np.float64)
            for b in range(m*p):
                T_for_V[b] = T[:, b, :].reshape(-1)
            
            try:
                V_new = T_for_V @ np.linalg.pinv(N.T)
            except:
                return None
            
            # Update W: for each c, T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
            # T[a,b,c] = sum_r W[c,r] U[a,r] V[b,r]
            # For fixed c: T[:,:,c] = W[c,:] @ P where P[:,r] = U[:,r] * V[:,r]
            P = np.zeros((n*m * m*p, rank), dtype=np.float64)
            for r in range(rank):
                P[r::rank] = (U[:, r] * V[:, r]).reshape(-1)
            
            T_for_W = np.zeros((p*n, n*m * m*p), dtype=np.float64)
            for c in range(p*n):
                T_for_W[c] = T[:, :, c].reshape(-1)
            
            try:
                W_new = T_for_W @ np.linalg.pinv(P.T)
            except:
                return None
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            error = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
            if error < tol:
                # Verify and return
                return U, V, W
        
        # Check final error
        error = np.linalg.norm(T - np.einsum('ar,br,cr->abc', U, V, W))
        if error < 1e-6:
            return U, V, W
        return None
    
    # Try ranks from lower bound up to schoolbook
    best_result = None
    for rank in range(28, min(57, 40)):  # Try 28 to 39 first
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            result = try_als(rank, max_iters=50)
            if result is not None:
                U, V, W = result
                # Convert to lists
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If ALS didn't find anything, return schoolbook
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
# EVOLVE-BLOCK-END

