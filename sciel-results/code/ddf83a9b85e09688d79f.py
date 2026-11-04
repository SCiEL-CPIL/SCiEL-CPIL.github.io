import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses vectorized validation and structured search patterns to maximize trial count.
    Falls back to schoolbook decomposition (rank 100) if no improvement is found.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

    # Schoolbook decomposition as guaranteed fallback
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def validate_fast(U, V, W, n, m, p, rank):
        """
        Fast vectorized validation using einsum.
        T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        Need T[i,j,k] == 1 for all i in [0,n), j in [0,m), k in [0,p).
        """
        # Reshape U to (n, m, rank)
        U_reshaped = U.reshape(n, m, rank)
        # Reshape V to (m, p, rank)  
        V_reshaped = V.reshape(m, p, rank)
        # Reshape W to (p, n, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        # Check if all entries equal 1
        return np.all(T == 1)

    def try_search(target_rank, num_trials, value_range=(-2, 3), sparsity_prob=0.0):
        """
        Try random decompositions with optional sparsity structure.
        Returns (U, V, W) if valid, else (None, None, None).
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None
            
            # Generate random matrices
            U = np.random.randint(value_range[0], value_range[1], size=(n * m, target_rank), dtype=np.int64)
            V = np.random.randint(value_range[0], value_range[1], size=(m * p, target_rank), dtype=np.int64)
            W = np.random.randint(value_range[0], value_range[1], size=(p * n, target_rank), dtype=np.int64)
            
            # Apply sparsity if requested
            if sparsity_prob > 0:
                for r in range(target_rank):
                    if np.random.random() < sparsity_prob:
                        U[:, r] = np.where(np.random.random(n * m) < 0.5, 0, 1)
                    if np.random.random() < sparsity_prob:
                        V[:, r] = np.where(np.random.random(m * p) < 0.5, 0, 1)
                    if np.random.random() < sparsity_prob:
                        W[:, r] = np.where(np.random.random(p * n) < 0.5, 0, 1)
            
            if validate_fast(U, V, W, n, m, p, target_rank):
                return U, V, W
        
        return None, None, None

    # Try structured patterns based on algebraic properties
    # Pattern: Try to exploit the fact that matrix multiplication is bilinear
    # Use Kronecker product structure
    
    def try_kronecker_pattern(target_rank, num_trials=100):
        """
        Try patterns based on Kronecker product structure.
        For 4x5x5, we can try to embed into a larger structure.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None
            
            # Create base patterns
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            W = np.zeros((p * n, target_rank), dtype=np.int64)
            
            # Randomly assign structure
            for r in range(target_rank):
                # Randomly choose which (i,j) pairs are active for U
                u_mask = np.random.random(n * m) < 0.3
                v_mask = np.random.random(m * p) < 0.3
                w_mask = np.random.random(p * n) < 0.3
                
                U[u_mask, r] = np.random.randint(-1, 2, size=np.sum(u_mask))
                V[v_mask, r] = np.random.randint(-1, 2, size=np.sum(v_mask))
                W[w_mask, r] = np.random.randint(-1, 2, size=np.sum(w_mask))
            
            if validate_fast(U, V, W, n, m, p, target_rank):
                return U, V, W
        
        return None, None, None

    # Search strategy: Try lower ranks first with increasing effort
    # Focus on ranks that are theoretically plausible
    candidate_ranks = [20, 25, 30, 35, 40]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try different search strategies for each rank
        U_try, V_try, W_try = try_search(rank, num_trials=200, value_range=(-1, 2), sparsity_prob=0.3)
        if U_try is not None:
            best_rank = rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
        
        # Also try Kronecker-inspired patterns
        U_try, V_try, W_try = try_kronecker_pattern(rank, num_trials=100)
        if U_try is not None:
            best_rank = rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # If still at schoolbook rank, try a few more aggressive searches
    if best_rank == base_rank and time.time() < deadline - 10:
        for rank in [30, 40]:
            if time.time() > deadline:
                break
            
            # More trials with wider value range
            U_try, V_try, W_try = try_search(rank, num_trials=500, value_range=(-3, 4), sparsity_prob=0.0)
            if U_try is not None:
                best_rank = rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
