"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build the tensor from decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_block_decomposition(n, m, p, time_limit):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    
    Strategy: Decompose the 3x5x6 problem into blocks and use known
    low-rank decompositions for smaller blocks where possible.
    
    For example, we can try:
    - 3x5x6 as (2x5x6) + (1x5x6)
    - 2x5x6 as (2x3x6) + (2x2x6)
    - etc.
    
    The key insight is that if we can find a decomposition where some
    blocks share rank components, we can reduce the total rank.
    """
    start_time = time.time()
    
    # Known minimal ranks for small cases (if known)
    # rank(1, m, p) = m*p (always, since each output needs m*p products)
    # rank(2, 2, 2) = 7 (Strassen)
    # rank(2, 3, 3) = ? (not known to be less than 18)
    # rank(2, 2, 3) = ? 
    # rank(3, 3, 3) = 23 (known upper bound, possibly minimal)
    
    # For our case 3x5x6, let's try a direct approach:
    # Use a randomized local search starting from a near-schoolbook initialization
    
    # Initialize with schoolbook-like structure but with some random perturbations
    # to allow the search to find better solutions
    
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from a reasonable upper bound down
    # Start with rank 85 and go down to see if we can find anything
    for target_rank in range(85, n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 5 if target_rank < 90 else 2
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            result = _local_search(target_rank, n, m, p, attempt, 
                                  time_limit - (time.time() - start_time))
            if result is not None:
                best_rank = target_rank
                best_solution = result
                break  # Found a solution for this rank, try lower
        
        if best_solution is not None and best_rank == target_rank:
            # Found solution, try to go lower
            continue
    
    return best_solution, best_rank

def _local_search(rank, n, m, p, seed, time_limit):
    """
    Local search for a rank-r decomposition.
    
    Start with a random initialization and iteratively improve.
    """
    start_time = time.time()
    np.random.seed(seed * 1000 + 42)
    
    # Initialize with random values
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Target tensor
    target = np.ones((n, m, p))
    
    # Compute initial error
    def compute_error(U, V, W):
        tensor = np.zeros((n, m, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += U_c[i, j] * V_c[j, k] * W_c[k, i]
        return np.linalg.norm(tensor - target)
    
    current_error = compute_error(U, V, W)
    
    # Iterative improvement: for each rank component, try to optimize
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for c in range(rank):
            if time.time() - start_time > time_limit:
                break
            
            # Try to optimize component c
            # Hold U and V fixed, optimize W
            # Or use a simpler heuristic: random perturbation
            
            # Simple approach: try small random perturbations
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                
                # Save current state
                U_old = U[:, c].copy()
                V_old = V[:, c].copy()
                W_old = W[:, c].copy()
                
                # Perturb
                U[:, c] += 0.1 * np.random.randn(n * m)
                V[:, c] += 0.1 * np.random.randn(m * p)
                W[:, c] += 0.1 * np.random.randn(p * n)
                
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                    improved = True
                else:
                    # Revert
                    U[:, c] = U_old
                    V[:, c] = V_old
                    W[:, c] = W_old
        
        if not improved:
            break
        
        # Check if we've found an exact solution
        if current_error < 1e-9:
            # Verify with exact check
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # First, try to find a better decomposition using block structure
    best_solution, best_rank = _try_block_decomposition(n, m, p, 60.0)
    
    if best_solution is not None and best_rank < n * m * p:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
