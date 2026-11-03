"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill
    
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.reshape(nm, -1)  # (nm, mp*pn)
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
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
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition exactly reconstructs T."""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - T_approx)) < tol
    
    def try_rank(target_rank, n_restarts=3, max_iter=50):
        """Try to find a rank-target_rank decomposition using ALS."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U: T_flat = U @ M_U where M_U[r, bc] = V[b,r]*W[c,r]
                M_U = np.einsum('br,cr->rbc', V, W).reshape(target_rank, mp * pn)
                try:
                    U_new = T_flat @ np.linalg.pinv(M_U)
                except:
                    break
                
                # Update V: for each b, T[:,b,:].reshape(-1) = M_V.T @ V[b,:]
                # M_V[r, ac] = U[a,r]*W[c,r]
                M_V = np.einsum('ar,cr->rac', U_new, W).reshape(target_rank, nm * pn)
                T_flat_v = T.reshape(mp, nm * pn)  # reshape: (mp, nm*pn)
                try:
                    V_new = T_flat_v @ np.linalg.pinv(M_V)
                except:
                    break
                
                # Update W: for each c, T[:,:,c].reshape(-1) = M_W.T @ W[c,:]
                # M_W[r, ab] = U[a,r]*V[b,r]
                M_W = np.einsum('ar,br->rab', U_new, V_new).reshape(target_rank, nm * mp)
                T_flat_w = T.reshape(pn, nm * mp)  # reshape: (pn, nm*mp)
                try:
                    W_new = T_flat_w @ np.linalg.pinv(M_W)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if check_decomposition(U, V, W, tol=1e-9):
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    return True
            # End iterations
        # End restarts
        return False
    
    # Try structured Strassen-like decomposition first
    # Partition 4x4 into 2x2 blocks, 4x7 into 2x2 and 2x1 blocks
    # This gives a structured approach that may find lower rank
    
    # Try ranks from 64 down to 56
    for target_rank in [64, 60, 56]:
        if time.time() > deadline - 10:
            break
        if try_rank(target_rank, n_restarts=2, max_iter=30):
            break
    
    # If we found something better, return it
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
