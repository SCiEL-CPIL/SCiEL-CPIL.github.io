import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    def check_solution(U, V, W):
        """Check if U, V, W form an exact decomposition with integer/half-integer entries."""
        # Check integer/half-integer
        U_int = np.round(2 * U) / 2
        V_int = np.round(2 * V) / 2
        W_int = np.round(2 * W) / 2
        
        if not np.allclose(U, U_int, atol=1e-9):
            return False
        if not np.allclose(V, V_int, atol=1e-9):
            return False
        if not np.allclose(W, W_int, atol=1e-9):
            return False
            
        # Check exactness
        # Reshape
        U_reshaped = U_int.reshape(n, m, -1)
        V_reshaped = V_int.reshape(m, p, -1)
        W_reshaped = W_int.reshape(p, n, -1)
        
        # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is a tensor contraction.
        # We can do this by looping or using einsum.
        # np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        # But R might be large, so einsum is efficient.
        try:
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False
            
        target = np.ones((n, m, p))
        return np.allclose(result, target, atol=1e-9)

    def als_decompose(rank, max_iters=50, tol=1e-6):
        """
        Alternating Least Squares for the matrix multiplication tensor decomposition.
        Returns (U, V, W) if converged to low residual, else None.
        """
        # Initialize randomly
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        prev_residual = np.inf
        for it in range(max_iters):
            # Update U
            # For each i, j, we have: sum_r U[i,j,r] * (sum_k V[j,k,r] * W[k,i,r]) = 1
            # Let Q[j,i,r] = sum_k V[j,k,r] * W[k,i,r]
            # Then sum_r U[i,j,r] * Q[j,i,r] = 1
            # This is a linear system: U[i,j,:] @ Q[j,i,:] = 1
            # U[i,j,:] is a row vector of length rank.
            # Q[j,i,:] is a row vector of length rank.
            # We want to find U[i,j,:] such that dot(U[i,j,:], Q[j,i,:]) = 1.
            # Minimum norm solution: U[i,j,:] = Q[j,i,:] / (Q[j,i,:] @ Q[j,i,:])
            
            Q = np.einsum('jkr,kir->jir', V_reshaped, W_reshaped)
            # Q shape: (m, n, rank)
            # We need to update U_reshaped (n, m, rank)
            # For each i, j:
            # Q[j,i,:] is the vector.
            # U_new[i,j,:] = Q[j,i,:] / sum(Q[j,i,:]**2)
            
            Q_norm_sq = np.sum(Q**2, axis=2) # (m, n)
            # Avoid division by zero
            Q_norm_sq_safe = np.where(Q_norm_sq > 1e-12, Q_norm_sq, 1.0)
            
            # U_new[i,j,r] = Q[j,i,r] / Q_norm_sq[j,i]
            # Note indices: U is (n, m, rank), Q is (m, n, rank)
            # So we need to transpose Q to match (n, m, rank)
            Q_transposed = np.transpose(Q, (1, 0, 2)) # (n, m, rank)
            U_reshaped = Q_transposed / Q_norm_sq_safe[:, :, np.newaxis]
            
            # Update V
            # For each j, k: sum_r V[j,k,r] * (sum_i U[i,j,r] * W[k,i,r]) = 1
            # Let P[j,k,r] = sum_i U[i,j,r] * W[k,i,r]
            # P shape: (m, p, rank)
            P = np.einsum('ijr,kir->jkr', U_reshaped, W_reshaped)
            P_norm_sq = np.sum(P**2, axis=2)
            P_norm_sq_safe = np.where(P_norm_sq > 1e-12, P_norm_sq, 1.0)
            V_reshaped = P / P_norm_sq_safe[:, :, np.newaxis]
            
            # Update W
            # For each k, i: sum_r W[k,i,r] * (sum_j U[i,j,r] * V[j,k,r]) = 1
            # Let S[k,i,r] = sum_j U[i,j,r] * V[j,k,r]
            # S shape: (p, n, rank)
            S = np.einsum('ijr,jkr->kir', U_reshaped, V_reshaped)
            S_norm_sq = np.sum(S**2, axis=2)
            S_norm_sq_safe = np.where(S_norm_sq > 1e-12, S_norm_sq, 1.0)
            W_reshaped = S / S_norm_sq_safe[:, :, np.newaxis]
            
            # Compute residual
            # result = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            residual = np.sum((result - 1.0)**2)
            
            if np.abs(prev_residual - residual) < tol:
                break
            prev_residual = residual
            
            # Optional: Snap to half-integers and check
            if it % 5 == 0:
                 U_snap = np.round(2 * U_reshaped) / 2
                 V_snap = np.round(2 * V_reshaped) / 2
                 W_snap = np.round(2 * W_reshaped) / 2
                 if check_solution(U_snap, V_snap, W_snap):
                     return U_snap, V_snap, W_snap

        # Final check
        U_flat = U_reshaped.reshape(-1, rank)
        V_flat = V_reshaped.reshape(-1, rank)
        W_flat = W_reshaped.reshape(-1, rank)
        
        # Try snapping
        U_snap = np.round(2 * U_flat) / 2
        V_snap = np.round(2 * V_flat) / 2
        W_snap = np.round(2 * W_flat) / 2
        
        if check_solution(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
            
        return None

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_solution = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try to find better rank
    for rank in range(rank_schoolbook - 1, max(10, rank_schoolbook - 20), -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            res = als_decompose(rank, max_iters=20, tol=1e-5)
            if res is not None:
                U, V, W = res
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                # If we found a solution, we can stop searching for this rank
                # and try a lower rank
                break
                
    return best_solution
# EVOLVE-BLOCK-END
