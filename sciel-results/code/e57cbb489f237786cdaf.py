"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 144
    
    # Build schoolbook decomposition
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Strategy: Use structured ansatz and verify
    
    # For matrix multiplication, we can try various structured decompositions
    # Key insight: we can share factors across different (i,j,k) triples
    
    # Try rank values from schoolbook down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But realistically, for 4x4*4x9, it's probably between 60-100
    
    np.random.seed(42)
    
    for target_rank in range(rank_sb - 1, 35, -1):
        if time.time() > deadline - 20:
            break
        
        success = _try_find_decomposition(n, m, p, target_rank, deadline - 10)
        if success is not None:
            best_U, best_V, best_W = success
            best_rank = target_rank
            break  # Found a lower rank, return it
    
    # Convert to lists
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using structured initialization and refinement."""
    
    # Try multiple strategies
    for strategy in ['structured', 'random', 'hybrid']:
        if time.time() > deadline:
            return None
        
        U, V, W = _initialize(n, m, p, rank, strategy)
        
        # Verify initial guess
        if _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
            return (U, V, W)
        
        # Try to refine
        U, V, W = _refine_decomposition(n, m, p, U, V, W, deadline)
        
        if _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
            return (U, V, W)
    
    return None


def _initialize(n, m, p, rank, strategy):
    """Initialize U, V, W matrices."""
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    if strategy == 'structured':
        # Use a structured pattern based on modular arithmetic
        for c in range(rank):
            # Distribute columns across (i,j,k) space
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            
            if c < n * m * p:
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0
            else:
                # For extra columns, use different patterns
                i2 = c % n
                j2 = (c // n) % m
                k2 = (c // (n * m)) % p
                U[i2*m + j2, c] = 1.0
                V[j2*p + k2, c] = 1.0
                W[k2*n + i2, c] = 1.0
    
    elif strategy == 'random':
        np.random.seed(hash((n, m, p, rank)) % (2**32))
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
    
    elif strategy == 'hybrid':
        # Start with structured, add small random perturbation
        for c in range(min(rank, n * m * p)):
            i = c % n
            j = (c // n) % m
            k = (c // (n * m)) % p
            U[i*m + j, c] = 1.0
            V[j*p + k, c] = 1.0
            W[k*n + i, c] = 1.0
        
        if rank > n * m * p:
            for c in range(n * m * p, rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                U[i*m + j, c] = np.random.randn() * 0.5
                V[j*p + k, c] = np.random.randn() * 0.5
                W[k*n + i, c] = np.random.randn() * 0.5
    
    return U, V, W


def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _refine_decomposition(n, m, p, U, V, W, deadline):
    """Refine the decomposition using iterative updates."""
    rank = U.shape[1]
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :]
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # We need: sum_r U[idx, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system: A @ U[idx, :] = b
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p + k] * W[k*n + i]
                
                try:
                    U[idx, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                # We need: sum_r U[i*m+j, r] * V[idx, r] * W[k*n+i, r] = 1 for all i
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m + j] * W[k*n + i]
                
                try:
                    V[idx, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[idx, r] = 1 for all j
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m + j] * V[j*p + k]
                
                try:
                    W[idx, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
    
    return U, V, W
# EVOLVE-BLOCK-END
