"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _als_pass(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """Alternating least squares to refine U, V, W."""
    prev_err = float('inf')
    for it in range(max_iters):
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, V, optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _polish(U, V, W, n, m, p, rank, max_iters=50):
    """High-precision polish pass with tighter tolerance."""
    return _als_pass(U, V, W, n, m, p, rank, max_iters=max_iters, tol=1e-14)

def _structured_init_winograd(n, m, p, rank, seed=42):
    """Winograd-style structured initialization using block transforms."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strategy: Use block structure where each "block" corresponds to a subset of indices
    # For 4x4x5, try to exploit the fact that 4 = 2^2 and 5 is prime
    
    # Initialize with sparse random structure
    for r in range(rank):
        # Assign each rank component to a "type" of interaction
        # Type 0: diagonal-like (i=j=k patterns)
        # Type 1: off-diagonal
        # Type 2: mixed
        
        # Random sparse initialization with bias toward structure
        for idx_u in range(n*m):
            i = idx_u // m
            j = idx_u % m
            if np.random.random() < 0.4:
                U[idx_u, r] = np.random.choice([-1.0, 1.0]) * 0.5
        
        for idx_v in range(m*p):
            j = idx_v // p
            k = idx_v % p
            if np.random.random() < 0.4:
                V[idx_v, r] = np.random.choice([-1.0, 1.0]) * 0.5
        
        for idx_w in range(p*n):
            k = idx_w // n
            i = idx_w % n
            if np.random.random() < 0.4:
                W[idx_w, r] = np.random.choice([-1.0, 1.0]) * 0.5
    
    return U, V, W

def _structured_init_hadamard(n, m, p, rank, seed=42):
    """Hadamard-inspired structured initialization."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Generate small Hadamard-like matrices
    def hadamard_4():
        return np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ], dtype=float) / 2.0
    
    H4 = hadamard_4()
    
    # For U: 4x4 = 16 entries, use blocks of H4
    for r in range(rank):
        # Randomly select a Hadamard block pattern
        block_i = np.random.randint(0, 4)
        block_j = np.random.randint(0, 4)
        sign = np.random.choice([-1.0, 1.0])
        
        for i in range(n):
            for j in range(m):
                # Use Hadamard structure
                U[i*m+j, r] = sign * H4[i, j] * 0.5
        
        for j in range(m):
            for k in range(p):
                # For V, use a different pattern
                V[j*p+k, r] = sign * H4[j, k % 4] * 0.5 if k < 4 else sign * 0.5
        
        for k in range(p):
            for i in range(n):
                W[k*n+i, r] = sign * H4[i, k % 4] * 0.5 if k < 4 else sign * 0.5
    
    return U, V, W

def _structured_init_block_diag(n, m, p, rank, seed=42):
    """Block diagonal-like structured initialization."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Divide rank into blocks
    num_blocks = min(4, rank // 4)
    block_size = max(1, rank // num_blocks)
    
    for b in range(num_blocks):
        start_r = b * block_size
        end_r = min(start_r + block_size, rank)
        
        for r in range(start_r, end_r):
            # Each block handles a subset of the multiplication
            block_i = b % n
            block_j = b % m
            
            for i in range(n):
                for j in range(m):
                    if i == block_i and j == block_j:
                        U[i*m+j, r] = 1.0
                    elif np.random.random() < 0.2:
                        U[i*m+j, r] = np.random.choice([-1.0, 1.0]) * 0.5
            
            for j in range(m):
                for k in range(p):
                    if j == block_j:
                        V[j*p+k, r] = 1.0
                    elif np.random.random() < 0.2:
                        V[j*p+k, r] = np.random.choice([-1.0, 1.0]) * 0.5
            
            for k in range(p):
                for i in range(n):
                    if i == block_i:
                        W[k*n+i, r] = 1.0
                    elif np.random.random() < 0.2:
                        W[k*n+i, r] = np.random.choice([-1.0, 1.0]) * 0.5
    
    return U, V, W

def _try_rank_structured(rank, n, m, p, seed=42, max_iters=80, polish_iters=30):
    """Try to find a decomposition using structured initialization and ALS."""
    init_methods = [
        _structured_init_winograd,
        _structured_init_hadamard,
        _structured_init_block_diag,
    ]
    
    for init_method in init_methods:
        U, V, W = init_method(n, m, p, rank, seed=seed)
        
        # ALS refinement
        U, V, W = _als_pass(U, V, W, n, m, p, rank, max_iters=max_iters)
        
        # Polish
        U, V, W = _polish(U, V, W, n, m, p, rank, max_iters=polish_iters)
        
        # Round to half-integers
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        
        # Verify
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def _try_rank_random(rank, n, m, p, seed=42, max_iters=60, polish_iters=20):
    """Try to find a decomposition using random initialization and ALS."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # ALS refinement
    U, V, W = _als_pass(U, V, W, n, m, p, rank, max_iters=max_iters)
    
    # Polish
    U, V, W = _polish(U, V, W, n, m, p, rank, max_iters=polish_iters)
    
    # Round to half-integers
    U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
    V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
    W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
    
    # Verify
    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110 second deadline (10s safety margin)
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try ranks from 75 down to 56
    for rank in range(75, 55, -1):
        if time.time() > deadline - 20:
            break
        
        # Try structured initialization with multiple seeds
        for seed in range(8):
            if time.time() > deadline - 20:
                break
            
            result = _try_rank_structured(rank, n, m, p, seed=seed, max_iters=60, polish_iters=20)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        # If structured didn't work, try random initialization
        if best_rank == rank_schoolbook and time.time() < deadline - 15:
            for seed in range(5):
                if time.time() > deadline - 15:
                    break
                
                result = _try_rank_random(rank, n, m, p, seed=seed, max_iters=40, polish_iters=15)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        
        if best_rank < rank_schoolbook:
            break
    
    return best_result
# EVOLVE-BLOCK-END
