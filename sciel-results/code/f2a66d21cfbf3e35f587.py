"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition exactly using vectorized operations."""
    R = len(U[0]) if U else 0
    if R == 0:
        return False
        
    # Convert to numpy arrays for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Target tensor T_target[i, j, k] = 1 if i,j,k valid
    # Actually, the tensor is defined by T[i,j,k] = 1
    # We check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Efficient check:
    # For each r, add outer product? No, it's a sum of products.
    # T_rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Since n,m,p are small, we can just loop over R and accumulate
    T_rec = np.zeros((n, m, p))
    for r in range(R):
        u_r = U_np[:, r]
        v_r = V_np[:, r]
        w_r = W_np[:, r]
        
        # u_r is length n*m. Reshape to (n, m)
        # v_r is length m*p. Reshape to (m, p)
        # w_r is length p*n. Reshape to (p, n) -> Need to transpose to (n, p) or align indices
        
        # Index mapping:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        u_mat = u_r.reshape(n, m)
        v_mat = v_r.reshape(m, p)
        w_mat = w_r.reshape(p, n) # w_mat[k, i]
        
        # We want T_rec[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        # This is a contraction.
        # Loop over i, j, k is 48 ops. Loop over R is ~48. Total ~2304 ops. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_rec[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    
    # Target is all 1s
    if not np.allclose(T_rec, np.ones((n, m, p)), atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Generate schoolbook decomposition of rank n*m*p."""
    R = n * m * p
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank(R, n, m, p, max_time=10.0, seed=42):
    """Try to find a rank-R decomposition using structured initialization + refinement."""
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    if R < 1:
        return None
    
    # Get schoolbook solution for initialization
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    U_sb = np.array(U_sb, dtype=float)
    V_sb = np.array(V_sb, dtype=float)
    W_sb = np.array(W_sb, dtype=float)
    
    best_loss = float('inf')
    best_U = best_V = best_W = None
    
    # Number of attempts: we have limited time, so keep it reasonable
    num_attempts = 20
    
    for attempt in range(num_attempts):
        if time.time() - start_time > max_time:
            break
        
        # Initialization Strategy:
        # 1. Truncated Schoolbook: Pick R random columns from Schoolbook
        # 2. Perturbed Schoolbook: Take first R columns and add noise
        # 3. Random (fallback)
        
        if R <= 48: # Schoolbook rank
            if attempt % 2 == 0:
                # Truncated Schoolbook
                indices = np.random.choice(48, R, replace=False)
                U = U_sb[:, indices].copy()
                V = V_sb[:, indices].copy()
                W = W_sb[:, indices].copy()
                # Add small noise to break symmetry and allow movement
                U += np.random.randn(n*m, R) * 0.01
                V += np.random.randn(m*p, R) * 0.01
                W += np.random.randn(p*n, R) * 0.01
            else:
                # Perturbed First R Columns
                U = U_sb[:, :R].copy()
                V = V_sb[:, :R].copy()
                W = W_sb[:, :R].copy()
                U += np.random.randn(n*m, R) * 0.1
                V += np.random.randn(m*p, R) * 0.1
                W += np.random.randn(p*n, R) * 0.1
        else:
            # Random initialization
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
        # Gradient Descent with Momentum
        lr = 0.05
        momentum = 0.9
        vel_U = np.zeros_like(U)
        vel_V = np.zeros_like(V)
        vel_W = np.zeros_like(W)
        
        for it in range(500):
            if time.time() - start_time > max_time:
                break
            
            # Compute Loss and Gradients
            # T_rec[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # We compute gradients analytically
            # dL/dU[i*m+j, r] = 2 * sum_{j',k'} (T_rec[i,j',k'] - 1) * V[j'*p+k', r] * W[k'*n+i, r]
            # This is expensive to loop. Let's use a helper to compute T_rec and gradients.
            
            # Since n,m,p are small, we can compute T_rec first, then gradients.
            T_rec = np.zeros((n, m, p))
            for r in range(R):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                u_mat = u_r.reshape(n, m)
                v_mat = v_r.reshape(m, p)
                w_mat = w_r.reshape(p, n)
                # T_rec += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                # This is a specific contraction.
                # We can use np.einsum or just loops.
                # Loops are fine for 3x4x4.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_rec[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                            
            loss = np.sum((T_rec - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if best_loss < 1e-10:
                    break
            
            # Compute Gradients
            # grad_U[i*m+j, r] = 2 * sum_{j',k'} (T_rec[i,j',k'] - 1) * V[j'*p+k', r] * W[k'*n+i, r]
            # Let's compute this efficiently.
            # For fixed i, r:
            # sum_{j', k'} Err[i, j', k'] * V[j', k', r] * W[k', i, r]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            Err = T_rec - 1.0
            
            for r in range(R):
                v_r = V[:, r]
                w_r = W[:, r]
                
                # For U
                for i in range(n):
                    # Sum over j, k
                    # Err[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0.0
                    for j in range(m):
                        for k in range(p):
                            s += Err[i, j, k] * v_r[j*p+k] * w_r[k*n+i]
                    grad_U[i*m + j, r] = 2.0 * s # Wait, j is loop var, this is wrong.
                    
                # Let's redo the gradient loop properly.
                # grad_U[idx_u, r] where idx_u = i*m+j
                # sum_{j', k'} Err[i, j', k'] * V[j'*p+k', r] * W[k'*n+i, r]
                # Note: i is determined by idx_u // m. j is idx_u % m.
                # The sum is over j' and k'.
                
                # This double loop inside r is 4 * 4 * 3 * R = 48 * R ops.
                # R ~ 48. 48*48 = 2304. Very fast.
                
                for idx_u in range(n*m):
                    i = idx_u // m
                    j = idx_u % m
                    s = 0.0
                    for jp in range(m):
                        for k in range(p):
                            s += Err[i, jp, k] * v_r[jp*p+k] * w_r[k*n+i]
                    grad_U[idx_u, r] = 2.0 * s
                    
                # For V
                # grad_V[idx_v, r] where idx_v = j*p+k
                # sum_{i', j'} Err[i', j', k] * U[i'*m+j', r] * W[k*n+i', r]
                # Note: j is idx_v // p, k is idx_v % p.
                # Sum over i' and j'.
                
                for idx_v in range(m*p):
                    j = idx_v // p
                    k = idx_v % p
                    s = 0.0
                    for i in range(n):
                        for jp in range(m):
                            s += Err[i, jp, k] * U[i*m+jp, r] * w_r[k*n+i]
                    grad_V[idx_v, r] = 2.0 * s
                    
                # For W
                # grad_W[idx_w, r] where idx_w = k*n+i
                # sum_{i', j'} Err[i', j', k] * U[i'*m+j', r] * V[j'*p+k, r]
                # Note: k is idx_w // n, i is idx_w % n.
                # Sum over i' and j'.
                
                for idx_w in range(p*n):
                    k = idx_w // n
                    i = idx_w % n
                    s = 0.0
                    for jp in range(m):
                        for i_prime in range(n): # Wait, Err depends on i, j, k.
                            # Err[i_prime, jp, k]
                            s += Err[i_prime, jp, k] * U[i_prime*m+jp, r] * v_r[jp*p+k]
                    grad_W[idx_w, r] = 2.0 * s

            # Update with Momentum
            lr_eff = lr * (0.99 ** it)
            
            vel_U = momentum * vel_U - lr_eff * grad_U
            U += vel_U
            
            vel_V = momentum * vel_V - lr_eff * grad_V
            V += vel_V
            
            vel_W = momentum * vel_W - lr_eff * grad_W
            W += vel_W
            
        if best_loss < 1e-8:
            break
    
    if best_loss > 1e-6:
        return None
    
    # Round to nearest half-integer
    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    
    # Verify rounded version
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    # Try without rounding
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 3 by 4 matrix by a 4 by 4 matrix.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # Start from plausible low ranks
    # We know rank 48 works. Try 47, 46...
    # If we find one, we break.
    # Given the difficulty, we might not find one.
    # We will try a few ranks.
    
    for target_rank in range(rank_schoolbook - 1, 11, -1):
        if time.time() > deadline - 5:
            break
            
        # Reduce time per rank if we have many to try
        # We have ~110 seconds.
        # If we try 36 ranks, we get ~3s each.
        # Let's try 5s per rank, max 20 ranks.
        
        result = _try_rank(target_rank, n, m, p, max_time=5.0, seed=target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
