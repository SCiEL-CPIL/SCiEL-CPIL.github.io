"""Matrix multiplication 3x4x8 - rank 48 decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a low-rank decomposition for 3x4x8 matrix multiplication.
    Uses alternating least squares to find a valid decomposition.
    Falls back to schoolbook rank 96 if no lower rank found.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 iff a%m == b//p and b%p == c//n and c%n == a//m
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def schoolbook():
        """Return schoolbook rank-96 decomposition."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Compute sum_s U[:,s] * V[:,s] * W[:,s] as a 3D tensor
        # Efficient: for each s, outer product of U[:,s], V[:,s], W[:,s]
        recon = np.zeros((nm, mp, pn))
        for s in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
        return np.max(np.abs(recon - T)) < tol
    
    def als_decompose(rank, max_iter=200, seed=42):
        """
        Alternating least squares for tensor decomposition.
        Find U(nm x rank), V(mp x rank), W(pn x rank) such that
        T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
        """
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Fix U, V; solve for W
            # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
            # For each c: T[:,:,c] = sum_s (U[:,s]*V[:,s]) * W[c,s]
            # This is: T[:,:,c] = (U ⊙ V) @ W[c,:]^T  (mode-3 unfolding)
            # Mode-3 unfolding of T: T_3 has shape (pn, nm*mp)
            # T_3[c, a*mp+b] = T[a,b,c]
            # T_3 = W @ (U ⊙ V)^T  where (U⊙V) has shape (nm*mp, rank)
            # So W = T_3 @ (U⊙V)^T @ ((U⊙V) @ (U⊙V)^T)^{-1}
            
            # Build (U ⊙ V) matrix of shape (nm*mp, rank)
            UV = np.zeros((nm * mp, rank))
            for s in range(rank):
                UV[:, s] = (U[:, s] * V[:, s]).flatten()
            
            # Mode-3 unfolding of T
            T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            
            # Solve: T3 = W @ UV^T  =>  W = T3 @ UV^T @ (UV @ UV^T)^{-1}
            # Or: (UV @ UV^T) @ W^T = UV @ T3^T
            try:
                A = UV @ UV.T  # (rank, rank)
                B_mat = UV @ T3.T  # (rank, pn)
                W = np.linalg.solve(A, B_mat).T  # (pn, rank)
            except np.linalg.LinAlgError:
                return None
            
            # Fix V, W; solve for U
            # T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
            # Mode-1 unfolding: T_1 has shape (nm, mp*pn)
            # T_1[a, b*pn+c] = T[a,b,c]
            # T_1 = U @ (V ⊙ W)^T
            # (V⊙W) has shape (mp*pn, rank)
            VW = np.zeros((mp * pn, rank))
            for s in range(rank):
                VW[:, s] = (V[:, s] * W[:, s]).flatten()
            
            T1 = T.transpose(0, 1, 2).reshape(nm, mp * pn)
            
            try:
                A = VW @ VW.T
                B_mat = VW @ T1.T
                U = np.linalg.solve(A, B_mat).T
            except np.linalg.LinAlgError:
                return None
            
            # Fix U, W; solve for V
            # Mode-2 unfolding: T_2 has shape (mp, nm*pn)
            # T_2[b, a*pn+c] = T[a,b,c]
            # T_2 = V @ (U ⊙ W)^T
            UW = np.zeros((nm * pn, rank))
            for s in range(rank):
                UW[:, s] = (U[:, s] * W[:, s]).flatten()
            
            T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            
            try:
                A = UW @ UW.T
                B_mat = UW @ T2.T
                V = np.linalg.solve(A, B_mat).T
            except np.linalg.LinAlgError:
                return None
            
            # Check convergence
            if it % 10 == 0:
                recon = np.zeros((nm, mp, pn))
                for s in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
                err = np.max(np.abs(recon - T))
                if err < 1e-9:
                    break
        
        # Final check
        if check_decomposition(U, V, W, tol=1e-8):
            return U, V, W
        return None
    
    # Try various ranks from low to high
    best_result = None
    
    for rank in [48, 56, 64, 72, 80, 88, 96]:
        if time.time() > deadline:
            break
        
        result = als_decompose(rank, max_iter=100, seed=rank)
        if result is not None:
            U, V, W = result
            # Verify
            if check_decomposition(U, V, W, tol=1e-8):
                best_result = (U, V, W, rank)
                break
    
    if best_result is None:
        # Try with different seeds for rank 48
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if time.time() > deadline:
                break
            result = als_decompose(48, max_iter=50, seed=seed + 100)
            if result is not None:
                U, V, W = result
                if check_decomposition(U, V, W, tol=1e-8):
                    best_result = (U, V, W, 48)
                    break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = schoolbook()
        rank = 96
    else:
        U, V, W, rank = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
