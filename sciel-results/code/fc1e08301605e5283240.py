"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _validate(U, V, W, n, m, p, atol=1e-8):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each triple (i, j, k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other triples, the sum should be 0.
    
    # We can check this efficiently using tensor contraction.
    # The tensor T has shape (n, m, p) and T[i, j, k] = 1.
    # The decomposition gives: T_decomp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's construct the full tensor T_decomp.
    # This is O(n*m*p*rank), which is fine for small sizes.
    
    rank = U.shape[1]
    
    # Reshape U to (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_3d = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T_decomp[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
    # This is a bit tricky to vectorize directly.
    # Let's loop over k and compute slices.
    
    T_decomp = np.zeros((n, m, p))
    for k in range(p):
        # For a fixed k, T_decomp[:, :, k] = sum_c U[:, :, c] * V[:, k, c] * W[k, :, c]
        # U[:, :, c] has shape (n, m)
        # V[:, k, c] has shape (m,)
        # W[k, :, c] has shape (n,)
        # We need: T_decomp[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        # Let's use einsum or broadcasting.
        # For each c, we have outer product of (n, m) and (m,) and (n,).
        # Actually, it's: sum_c U[:, :, c] * V[:, k, c][None, :, :] * W[k, :, c][:, None, :]
        # Wait, V[:, k, c] is a vector of length m.
        # U[:, :, c] is a matrix of shape (n, m).
        # W[k, :, c] is a vector of length n.
        
        # T_decomp[:, :, k] = sum_c (U[:, :, c] * V[:, k, c][None, :]) * W[k, :, c][:, None]
        # This is: for each c, multiply U[:, :, c] (n, m) by V[:, k, c] (m,) along axis 1,
        # then multiply by W[k, :, c] (n,) along axis 0.
        
        # Let's do it step by step.
        for c in range(rank):
            # term[i, j] = U[i, j, c] * V[j, k, c] * W[k, i, c]
            # V[:, k, c] has shape (m,), so V[:, k, c][j] is the j-th element.
            # We need to multiply U[:, :, c] (n, m) by V[:, k, c] (m,) along the m-axis.
            # This is: U[:, :, c] * V[:, k, c][None, :]  -> shape (n, m)
            # Then multiply by W[k, :, c] (n,) along the n-axis.
            # This is: (U[:, :, c] * V[:, k, c][None, :]) * W[k, :, c][:, None]
            
            T_decomp[:, :, k] += U_3d[:, :, c] * V_3d[:, k, c][None, :] * W_3d[k, :, c][:, None]
    
    # Expected tensor
    T_expected = np.ones((n, m, p))
    
    # Check
    return np.allclose(T_decomp, T_expected, atol=atol)

def _random_decomposition(n, m, p, rank, seed=42):
    """Generate a random initial decomposition."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _loss_function(x, n, m, p, rank, U_fixed=None, V_fixed=None, W_fixed=None):
    """Loss function for optimization."""
    # x is a flattened array of length (n*m + m*p + p*n) * rank
    # But we might be optimizing only some of them.
    # For now, let's assume we optimize all.
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    
    U = x[:size_U].reshape(n * m, rank)
    V = x[size_U:size_U + size_V].reshape(m * p, rank)
    W = x[size_U + size_V:size_U + size_V + size_W].reshape(p * n, rank)
    
    # Compute T_decomp
    T_decomp = np.zeros((n, m, p))
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    for k in range(p):
        for c in range(rank):
            T_decomp[:, :, k] += U_3d[:, :, c] * V_3d[:, k, c][None, :] * W_3d[k, :, c][:, None]
    
    T_expected = np.ones((n, m, p))
    loss = np.sum((T_decomp - T_expected) ** 2)
    return loss

def construct():
    n, m, p = DIMENSIONS
    
    # Start with a reasonable rank.
    # Schoolbook is n*m*p = 112.
    # Let's try to find a decomposition with lower rank.
    # Known best for 4x4x7 is likely around 80-90.
    
    deadline = time.time() + 100  # Safety margin
    
    best_rank = n * m * p
    best_decomp = None
    
    # First, try to find a valid decomposition with a given rank using optimization.
    # We'll try ranks from best possible down to schoolbook.
    
    # What's the theoretical lower bound?
    # For matrix multiplication, the tensor rank is at least max(n*m, m*p, p*n) = 28.
    # But it's much higher.
    
    # Let's try a few ranks.
    ranks_to_try = [70, 75, 80, 85, 90, 95, 100, 105, 112]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Random initialization
            U, V, W = _random_decomposition(n, m, p, rank, seed=42 + restart)
            
            # Check if already valid (unlikely)
            if _validate(U, V, W, n, m, p):
                best_rank = rank
                best_decomp = (U, V, W)
                found = True
                break
            
            # Try to optimize
            x0 = np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
            
            # Use Nelder-Mead or Powell, as gradient computation is expensive.
            # But for high dimensions, it might be slow.
            
            # Let's try a few steps of coordinate descent or something simple.
            # Actually, let's just check if the random init is close enough to be refined.
            
            # For now, let's just use the random init if it's valid.
            # This is not a strong method.
            
            # A better approach: use known low-rank decompositions for smaller blocks.
    
    # If no better found, use schoolbook.
    if best_decomp is None:
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
    
    return {"U": best_decomp[0], "V": best_decomp[1], "W": best_decomp[2]}
# EVOLVE-BLOCK-END
