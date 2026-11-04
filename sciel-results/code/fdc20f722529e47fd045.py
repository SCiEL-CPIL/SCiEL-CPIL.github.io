"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, T, tol=1e-9):
    """Verify that U @ (V * W) == T (element-wise product then matmul)."""
    n, m, p = T.shape
    # T[i,j,k] should equal sum_c U[i,c] * V[j,c] * W[k,c]
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.max(np.abs(recon - T)) < tol

def _snap_to_rational(mat, max_denom=4):
    """Try to snap a matrix to rational values with small denominators."""
    snapped = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x = mat[i, j]
            best = x
            best_err = abs(x - best)
            for d in range(1, max_denom + 1):
                n = round(x * d)
                val = n / d
                err = abs(x - val)
                if err < best_err:
                    best_err = err
                    best = val
            snapped[i, j] = best
    return snapped

def _construct_rank5_2x3x3():
    """
    Construct a rank-5 decomposition for 2x3 * 3x3 = 2x3 matrix multiplication.
    
    The tensor M_{2,3,3} has entries: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Known rank-5 algorithm (Laderman-like for rectangular case):
    We need 5 products to compute all 6 outputs.
    
    Let's use the following identity:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    Rank-5 decomposition:
    P1 = A[0,0] * (B[0,0] + B[0,1] + B[0,2])
    P2 = A[0,1] * (B[1,0] + B[1,1] + B[1,2])
    P3 = A[0,2] * (B[2,0] + B[2,1] + B[2,2])
    P4 = (A[0,0] + A[1,0]) * (B[0,0] + B[0,1] + B[0,2])
    P5 = A[1,0] * (B[0,0] + B[0,1] + B[0,2])
    
    This doesn't work directly. Let me use a different approach.
    
    Actually, for 2x3x3, the rank is indeed 5. Here's a valid decomposition:
    
    Let s = B[0,0] + B[0,1] + B[0,2] (sum of first row of B)
    Let t = B[1,0] + B[1,1] + B[1,2] (sum of second row of B)
    Let u = B[2,0] + B[2,1] + B[2,2] (sum of third row of B)
    
    Hmm, this approach is getting complicated. Let me just use the schoolbook 
    for the sub-problems and rely on the overall structure.
    
    For now, I'll implement a direct search-based approach that tries to find
    low-rank decompositions using ALS with rational snapping.
    """
    pass

def _als_decompose(T, target_rank, max_iters=5000, tol=1e-10, seed=42):
    """
    Alternating Least Squares to find a rank-r decomposition of tensor T.
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    """
    n, m, p = T.shape
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n, target_rank)
    V = rng.randn(m, target_rank)
    W = rng.randn(p, target_rank)
    
    # Normalize to prevent scale issues
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Update U: fix V, W. For each i, solve linear system
        # T[i,:,:] = sum_c U[i,c] * V[:,c] * W[:,c]
        # Let M_c = V[:,c] * W[:,c] (element-wise), shape (m*p,)
        # T[i,:,:].reshape(-1) = sum_c U[i,c] * M_c
        # This is: T[i].reshape(-1) = U[i,:] @ M.T where M is (m*p, target_rank)
        
        # Build M: M[j,k,c] = V[j,c] * W[k,c], but we need it flattened
        # M_flat has shape (m*p, target_rank) where M_flat[j*p+k, c] = V[j,c]*W[k,c]
        M = np.einsum('jc,kc->jkc', V, W).reshape(m * p, target_rank)
        
        # For each i, U[i,:] = argmin ||T[i].reshape(-1) - U[i,:] @ M.T||^2
        # U[i,:] = T[i].reshape(-1) @ pinv(M.T)
        # Or solve the normal equations: (M @ M.T) U[i,:] = M @ T[i].reshape(-1)
        
        MMT = M @ M.T  # (m*p, m*p)
        MT_T = M @ T.reshape(n, m * p).T  # (m*p, n)
        
        # Solve MMT @ U.T = MT_T
        try:
            U_new = np.linalg.solve(MMT + 1e-10 * np.eye(m * p), MT_T).T
        except np.linalg.LinAlgError:
            U_new = np.linalg.lstsq(MMT + 1e-10 * np.eye(m * p), MT_T, rcond=None)[0].T
        
        # Update V
        MW = np.einsum('ic,kc->ikc', U, W).reshape(n * p, target_rank)
        MWM = MW @ MW.T
        MWT = MW @ T.reshape(n, m, p).reshape(n * p, m).T
        try:
            V_new = np.linalg.solve(MWM + 1e-10 * np.eye(n * p), MWT).T
        except np.linalg.LinAlgError:
            V_new = np.linalg.lstsq(MWM + 1e-10 * np.eye(n * p), MWT, rcond=None)[0].T
        
        # Update W
        MV = np.einsum('ic,jc->ijc', U, V).reshape(n * m, target_rank)
        MVM = MV @ MV.T
        MVT = MV @ T.reshape(n, m, p).reshape(n * m, p).T
        try:
            W_new = np.linalg.solve(MVM + 1e-10 * np.eye(n * m), MVT).T
        except np.linalg.LinAlgError:
            W_new = np.linalg.lstsq(MVM + 1e-10 * np.eye(n * m), MVT, rcond=None)[0].T
        
        U, V, W = U_new, V_new, W_new
        
        # Compute loss
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        loss = np.sum((recon - T) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build target tensor T[i,j,k] = 1 for the matrix multiplication structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Schoolbook fallback: rank n*m*p
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Try ranks from 16 to 41 (16 is a theoretical lower bound from composition)
    for target_rank in range(16, rank_sb):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
            
            seed = 12345 + target_rank * 7 + restart
            U, V, W, loss = _als_decompose(T, target_rank, max_iters=3000, tol=1e-12, seed=seed)
            
            if loss < 1e-8:
                # Try snapping to rationals
                for max_denom in [1, 2, 3, 4]:
                    U_r = _snap_to_rational(U, max_denom)
                    V_r = _snap_to_rational(V, max_denom)
                    W_r = _snap_to_rational(W, max_denom)
                    
                    recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                    loss_r = np.max(np.abs(recon - T))
                    
                    if loss_r < 1e-9:
                        best_result = (U_r, V_r, W_r)
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
