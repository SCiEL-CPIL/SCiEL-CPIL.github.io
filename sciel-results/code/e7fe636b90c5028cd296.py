"""Program for matrix multiplication tensor decomposition (3x4, 4x6)."""
import time
import numpy as np
import itertools

# The dimensions: n x m * m x p = 3x4 * 4x6
n_dim = 3
m_dim = 4
p_dim = 6

# Row indices:
# U rows: i*m + j  -> 12 rows
# V rows: j*p + k  -> 24 rows
# W rows: k*n + i  -> 18 rows

def _verify_fast(U, V, W, tensor, tol=1e-9):
    """Fast verification using vectorization."""
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # Tensor: (n, m, p)
    
    rank = U.shape[1]
    n, m, p = tensor.shape
    
    # We need to check if sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == tensor[i,j,k]
    
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute the sum over c
    # U[i, j, c] * V[j, k, c] * W[k, i, c]
    # This is a tensor contraction.
    # Let's loop over i, j, k is slow.
    # Better: 
    # For each c, we have a rank-1 tensor T_c[i,j,k] = U[i,j,c] V[j,k,c] W[k,i,c]
    # Sum over c.
    
    # We can do this by looping over c and accumulating, or using einsum.
    # np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    try:
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        err = np.max(np.abs(reconstructed - tensor))
        return err < tol
    except Exception:
        return False

def _als_step(U, V, W, tensor, n, m, p, rank):
    """Perform one step of Alternating Least Squares."""
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # 1. Solve for U
    # For each i, j: sum_c U[i,j,c] * (V[j,:,c] * W[:,i,c]) = tensor[i,j,:]
    # Let's form the matrix for each (i,j).
    # A[i,j,k,c] = V[j,k,c] * W[k,i,c]
    # This is a 4D tensor. We want to solve A[i,j,:,c] @ U[i,j,:] = tensor[i,j,:]
    
    # We can vectorize this.
    # For each i, j, we have a linear system of size p x rank.
    
    U_new = np.zeros((n, m, rank))
    for i in range(n):
        for j in range(m):
            # A is (p, rank)
            # A[k, c] = V[j, k, c] * W[k, i, c]
            A = V_reshaped[j, :, :] * W_reshaped[:, i, :] # (p, rank)
            # b is (p,)
            b = tensor[i, j, :]
            # Solve A @ U[i,j,:]^T = b
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U_new[i, j, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    U = U_new.reshape(n * m, rank)
    
    # 2. Solve for V
    # For each j, k: sum_c V[j,k,c] * (U[:,j,c] * W[k,:,c]) = tensor[:,j,k]
    # A[j,k,i,c] = U[i,j,c] * W[k,i,c]
    # We want A[j,k,:,c] @ V[j,k,:]^T = tensor[:,j,k]
    
    V_new = np.zeros((m, p, rank))
    for j in range(m):
        for k in range(p):
            # A is (n, rank)
            # A[i, c] = U[i, j, c] * W[k, i, c]
            A = U_reshaped[:, j, :] * W_reshaped[k, :, :] # (n, rank)
            # b is (n,)
            b = tensor[:, j, k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_new[j, k, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    V = V_new.reshape(m * p, rank)
    
    # 3. Solve for W
    # For each k, i: sum_c W[k,i,c] * (U[:, :, c] * V[:, k, c]) = tensor[:, :, k]
    # A[k,i,j,c] = U[i,j,c] * V[j,k,c]
    # We want A[k,i,:,c] @ W[k,i,:]^T = tensor[:,:,k]
    
    W_new = np.zeros((p, n, rank))
    for k in range(p):
        for i in range(n):
            # A is (m, rank)
            # A[j, c] = U[i, j, c] * V[j, k, c]
            A = U_reshaped[i, :, :] * V_reshaped[:, k, :] # (m, rank)
            # b is (m,)
            b = tensor[:, :, k] # Wait, tensor[i,j,k]. We need tensor[i,:,k]? No.
            # The equation is: sum_c W[k,i,c] * U[i,j,c] * V[j,k,c] = tensor[i,j,k]
            # For fixed k, i, we vary j.
            # So b is tensor[i, :, k] which is a vector of length m.
            b = tensor[i, :, k]
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W_new[k, i, :] = sol
            except np.linalg.LinAlgError:
                pass
    
    W = W_new.reshape(p * n, rank)
    
    return U, V, W

def construct():
    # Deadline guard
    deadline = 145  # seconds, safe margin before 150s limit
    
    n = n_dim
    m = m_dim
    p = p_dim
    
    # Create the standard tensor: tensor[i,j,k] = 1 for all i,j,k
    tensor = np.ones((n, m, p), dtype=np.float64)
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
    best_rank = rank_schoolbook
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Try ranks from low to high
    # Lower bound: The rank of the matrix multiplication tensor $T_{n \times m \times p}$ is at least $\max(n, m, p)$?
    # Actually, the rank is at least the matrix rank of any fiber.
    # Fiber U: n x (m*p) matrix. Rank <= min(n, mp).
    # Fiber V: m x (n*p) matrix. Rank <= min(m, np).
    # Fiber W: p x (n*m) matrix. Rank <= min(p, nm).
    # So rank >= max(n, m, p) is not a tight lower bound for tensor rank, but a necessary condition.
    # For 3x4x6, lower bound is 6.
    # Let's try a range of ranks.
    
    ranks_to_try = list(range(24, rank_schoolbook + 1))
    # Prioritize lower ranks
    ranks_to_try.sort()
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 5 if target_rank < 40 else 2
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = rng.standard_normal((n * m, target_rank))
            V = rng.standard_normal((m * p, target_rank))
            W = rng.standard_normal((p * n, target_rank))
            
            # Run ALS for a few iterations
            for iter in range(20):
                if time.time() > deadline - 2:
                    break
                U, V, W = _als_step(U, V, W, tensor, n, m, p, target_rank)
                
                # Check for convergence/exactness
                if _verify_fast(U, V, W, tensor):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
