"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    nm = n * m  # 15
    mp = m * p  # 25
    pn = p * n  # 15
    
    # Schoolbook baseline (always valid)
    rank_sb = n * m * p  # 75
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form a valid decomposition.
        For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1.
        """
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)
    
    def compute_loss(U, V, W):
        """Compute the loss ||T - 1||^2."""
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.sum((T - 1.0)**2)
    
    def gradient_step(U, V, W, lr):
        """Compute one gradient descent step and return updated U, V, W."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        diff = 2.0 * (T - 1.0)
        
        grad_U = np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        grad_V = np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        grad_W = np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        U_new = U - lr * grad_U.reshape(nm, r)
        V_new = V - lr * grad_V.reshape(mp, r)
        W_new = W - lr * grad_W.reshape(pn, r)
        
        return U_new, V_new, W_new
    
    def als_step(U, V, W):
        """One ALS iteration: fix two matrices, solve for the third via least squares."""
        r = U.shape[1]
        
        # Solve for U: for each (i,j), U[i*m+j,:] = argmin ||sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1||^2
        # This is: for fixed i,j, minimize over u = U[i*m+j,:]
        # sum_k (sum_c u_c * V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # = sum_k (A_k @ u - 1)^2 where A_k[c] = V[j*p+k,c]*W[k*n+i,c]
        # This is a linear least squares problem: A @ u = b where b = ones(p)
        # A is p x r, b is p
        
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                # Build A: shape (p, r)
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ u = ones(p)
                try:
                    u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = u
                except:
                    pass
        
        # Solve for V: for each (j,k), V[j*p+k,:] = argmin
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = v
                except:
                    pass
        
        # Solve for W: for each (k,i), W[k*n+i,:] = argmin
        W_new = W.copy()
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = w
                except:
                    pass
        
        return U_new, V_new, W_new
    
    def optimize_from_init(U0, V0, W0, max_iter=200, use_als=True):
        """Optimize from given initialization, alternating between gradient and ALS."""
        if time.time() > deadline:
            return None
        
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        best_loss = compute_loss(U, V, W)
        best_state = (U.copy(), V.copy(), W.copy())
        
        lr = 0.05
        patience = 0
        max_patience = 30
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Alternate between ALS and gradient steps
            if use_als and it % 3 == 0:
                U, V, W = als_step(U, V, W)
            else:
                U, V, W = gradient_step(U, V, W, lr)
            
            loss = compute_loss(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_state = (U.copy(), V.copy(), W.copy())
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                # Reduce learning rate
                lr *= 0.5
                patience = 0
                max_patience = max(10, max_patience // 2)
            
            if best_loss < 1e-12:
                break
        
        return best_state
    
    def try_find_decomposition(target_rank, num_trials=5, max_iter=150):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        np.random.seed(42 + target_rank)  # Reproducible but different per rank
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Different initialization strategies
            if trial == 0:
                # Structured init: start from schoolbook and randomly project
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
            elif trial == 1:
                # Perturbed schoolbook projection
                # Take random linear combinations of schoolbook columns
                idx = np.random.choice(rank_sb, target_rank, replace=False)
                U = U_sb[:, idx] @ np.random.randn(target_rank, target_rank) * 0.5
                V = V_sb[:, idx] @ np.random.randn(target_rank, target_rank) * 0.5
                W = W_sb[:, idx] @ np.random.randn(target_rank, target_rank) * 0.5
            elif trial == 2:
                # Sparse random init
                U = np.random.randn(nm, target_rank)
                V = np.random.randn(mp, target_rank)
                W = np.random.randn(pn, target_rank)
                # Make sparse
                mask_U = np.random.rand(nm, target_rank) < 0.3
                mask_V = np.random.rand(mp, target_rank) < 0.3
                mask_W = np.random.rand(pn, target_rank) < 0.3
                U *= mask_U
                V *= mask_V
                W *= mask_W
                U *= 1.0 / np.sqrt(0.3)
                V *= 1.0 / np.sqrt(0.3)
                W *= 1.0 / np.sqrt(0.3)
            elif trial == 3:
                # Block-structured init
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                # Assign random blocks
                for c in range(target_rank):
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    U[i*m+j, c] = np.random.randn()
                    V[j*p+k, c] = np.random.randn()
                    W[k*n+i, c] = np.random.randn()
            else:
                # Pure random with different scale
                scale = 0.2 + 0.1 * trial
                U = np.random.randn(nm, target_rank) * scale
                V = np.random.randn(mp, target_rank) * scale
                W = np.random.randn(pn, target_rank) * scale
            
            result = optimize_from_init(U, V, W, max_iter=max_iter)
            if result is None:
                continue
            
            U_opt, V_opt, W_opt = result
            if check_decomposition(U_opt, V_opt, W_opt):
                return (U_opt, V_opt, W_opt)
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bounds: max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # But matrix multiplication rank is generally higher
    # Known results suggest rank around 40-60 for 3x5x5
    ranks_to_try = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate time based on how many ranks remain
        remaining_ranks = len(ranks_to_try) - ranks_to_try.index(target_rank)
        time_per_rank = (deadline - time.time()) / max(1, remaining_ranks)
        
        # More trials for lower ranks (harder to find)
        num_trials = 8 if target_rank < 40 else 5
        max_iter = 200 if target_rank < 40 else 150
        
        result = try_find_decomposition(target_rank, num_trials=num_trials, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                # Don't break - try lower ranks if time permits
                if target_rank <= 30:
                    break  # Good enough, stop
    
    # Final check
    if not check_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_rank = rank_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
