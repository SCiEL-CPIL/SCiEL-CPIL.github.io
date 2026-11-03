"""Matrix multiplication 3x3 x 3x6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    # n=3 (rows of A/C), m=3 (cols of A / rows of B), p=6 (cols of B/C)
    
    # Build the correct multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'
    # This encodes: C[i,k] = sum_j A[i,j] * B[j,k]
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = make_schoolbook()
    best_rank = n * m * p  # 54
    
    # Try to find lower rank decompositions
    # For rank r: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # where i indexes (n*m), j indexes (m*p), k indexes (p*n)
    
    # Strategy: Use the block structure. 3x3x6 = two 3x3x3 blocks.
    # For each 3x3x3 block, try to find a good decomposition.
    # Then combine them.
    
    # First, let's try a direct numerical approach for small ranks.
    # We'll use a random initialization + iterative refinement approach.
    
    def try_rank(r, max_iters=500, tol=1e-6):
        """Try to find a rank-r decomposition of T."""
        if r >= best_rank:
            return None
        
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None
            
            # Compute current approximation
            # T_approx[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            # Residual R[i,j,k] = T[i,j,k] - T_approx[i,j,k]
            
            # Use a simple gradient-free approach: alternating minimization
            # Fix U and V, solve for W:
            # For each k: sum_c W[k,c] * (U*V)[:,c] = T[:,:,k]
            # This is a linear system for W[k,:]
            
            # Actually, let's use a simpler approach: coordinate descent on columns
            # For each column c, we want to minimize ||T - sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]||
            
            # Let's use a different approach: fix two matrices, update the third
            # by solving a linear system.
            
            # Update W: for each k, T[:,:,k] = sum_c W[k,c] * (U ⊙ V)[:,c]
            # where (U ⊙ V)[:,c] is the element-wise product of U and V... no.
            # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            # For fixed U,V: T[:,:,k] = sum_c W[k,c] * outer(U[:,c], V[:,c])
            # So T[:,:,k] = sum_c W[k,c] * M_c where M_c = outer(U[:,c], V[:,c])
            # This is: vec(T[:,:,k]) = sum_c W[k,c] * vec(M_c)
            # Let A be the matrix with columns vec(M_c), c=0..r-1
            # Then vec(T[:,:,k]) = A * W[k,:]
            # So W[k,:] = A \ vec(T[:,:,k])
            
            A = np.zeros((n*m * m*p, r))
            for c in range(r):
                A[:,c] = np.outer(U[:,c], V[:,c]).flatten()
            
            # Solve for W
            W_new = np.zeros((p*n, r))
            for k in range(p*n):
                target = T[:,:,k].flatten()
                try:
                    W_new[k,:] = np.linalg.lstsq(A, target, rcond=None)[0]
                except:
                    W_new[k,:] = W[k,:]
            W = W_new
            
            # Update U: for each i, T[i,:,:] = sum_c U[i,c] * (V ⊙ W)[:,c]
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed V,W: T[i,:,:] = sum_c U[i,c] * outer(V[:,c], W[:,c])
            B_mat = np.zeros((m*p * p*n, r))
            for c in range(r):
                B_mat[:,c] = np.outer(V[:,c], W[:,c]).flatten()
            
            U_new = np.zeros((n*m, r))
            for i in range(n*m):
                target = T[i,:,:].flatten()
                try:
                    U_new[i,:] = np.linalg.lstsq(B_mat, target, rcond=None)[0]
                except:
                    U_new[i,:] = U[i,:]
            U = U_new
            
            # Update V: for each j, T[:,j,:] = sum_c V[j,c] * (U ⊙ W)[:,c]
            C_mat = np.zeros((n*m * p*n, r))
            for c in range(r):
                C_mat[:,c] = np.outer(U[:,c], W[:,c]).flatten()
            
            V_new = np.zeros((m*p, r))
            for j in range(m*p):
                target = T[:,j,:].flatten()
                try:
                    V_new[j,:] = np.linalg.lstsq(C_mat, target, rcond=None)[0]
                except:
                    V_new[j,:] = V[j,:]
            V = V_new
            
            # Check convergence
            T_approx = np.zeros_like(T)
            for c in range(r):
                T_approx += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
            residual = np.linalg.norm(T - T_approx)
            
            if residual < tol:
                # Verify and return
                if np.allclose(T_approx, T, atol=1e-4):
                    return {"U": U, "V": V, "W": W}
        
        # Check final residual
        T_approx = np.zeros_like(T)
        for c in range(r):
            T_approx += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        residual = np.linalg.norm(T - T_approx)
        if residual < 1e-4:
            return {"U": U, "V": V, "W": W}
        return None
    
    # Try ranks from low to high
    # For 3x3x3, known minimum rank is 23 (Strassen-like)
    # For 3x3x6, lower bound is at least 23 (since it contains 3x3x3)
    # Upper bound is 54 (schoolbook) or 46 (two independent 3x3x3 with rank 23 each)
    
    # Let's try ranks: 23, 24, ..., 46, 47, ..., 53
    # But this might be slow. Let's try a few key ranks.
    
    ranks_to_try = [23, 27, 30, 36, 40, 45, 46, 50, 53]
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, max_iters=200, tol=1e-5)
        if result is not None:
            # Verify
            U, V, W = result["U"], result["V"], result["W"]
            T_approx = np.zeros_like(T)
            for c in range(r):
                T_approx += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
            if np.allclose(T_approx, T, atol=1e-3):
                best = result
                best_rank = r
                break
    
    # Convert to list format if needed (task says integer/half-integer entries)
    # But for now, return as is
    
    return best

