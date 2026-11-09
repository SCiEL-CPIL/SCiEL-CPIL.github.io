import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_reshaped = U_arr.reshape(n, m, r)
    V_reshaped = V_arr.reshape(m, p, r)
    W_reshaped = W_arr.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    """Generate the standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _try_merge_terms(U, V, W, n, m, p, max_iterations=5000, time_limit=5.0):
    """
    Try to reduce rank by merging terms.
    Start from a valid decomposition and try to find linear dependencies
    among the rank-1 terms to eliminate some.
    """
    start_time = time.time()
    r = len(U[0])
    
    # Work with numpy arrays
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    current_rank = r
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if current_rank <= 1:
            break
            
        # Try to find two terms that can be combined
        # A term c contributes U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We want to find c1, c2 such that their combined contribution
        # can be represented with one term
        
        # Strategy: try random pairs and see if we can eliminate one
        c1 = random.randint(0, current_rank - 1)
        c2 = random.randint(0, current_rank - 1)
        if c1 == c2:
            continue
            
        # Try to find coefficients a, b, d such that:
        # a*term_c1 + b*term_c2 = d*new_term
        # This is complex; instead, try a simpler approach:
        # Check if removing one term and adjusting others still works
        
        # Simpler: try setting one term to zero and compensating
        # This is heuristic and may not work well
        
        # Alternative: use a local optimization approach
        # For now, this function is a placeholder for more advanced methods
        pass
    
    return None

def _structured_search(n, m, p, target_rank, time_limit=10.0):
    """
    Try to find a decomposition of target_rank using structured approaches.
    For small matrices, we can try to use known algorithmic patterns.
    """
    start_time = time.time()
    
    # Approach 1: Try to build from scratch with small integer entries
    # For 3x4x5, the schoolbook rank is 60
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    
    # Try random search with small entries, but more intelligently
    # Initialize with sparse random matrices
    
    max_attempts = 2000
    attempts = 0
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        
        # Generate random decomposition with entries in {-1, 0, 1}
        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Make matrices sparse: each entry is -1, 0, or 1 with some probability
        sparsity = 0.3  # 30% chance of non-zero
        
        for c in range(target_rank):
            U[:, c] = np.where(np.random.random(n * m) < sparsity, 
                               np.random.choice([-1, 1], size=n * m), 0)
            V[:, c] = np.where(np.random.random(m * p) < sparsity, 
                               np.random.choice([-1, 1], size=m * p), 0)
            W[:, c] = np.where(np.random.random(p * n) < sparsity, 
                               np.random.choice([-1, 1], size=p * n), 0)
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
        
        # Also try with entries in {-2, -1, 0, 1, 2}
        if attempts % 2 == 0:
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            for c in range(target_rank):
                U[:, c] = np.random.randint(-2, 3, size=n * m).astype(np.float64)
                V[:, c] = np.random.randint(-2, 3, size=m * p).astype(np.float64)
                W[:, c] = np.random.randint(-2, 3, size=p * n).astype(np.float64)
            
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Set a strict deadline: 145s to be safe (evaluator kills at 150s)
    total_deadline = time.time() + 145
    
    # Generate schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Search from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 20
    lower_bound = max(n * m, m * p, p * n)
    
    # We'll try ranks from lower_bound to rank_schoolbook - 1
    # Allocate time based on rank: lower ranks get less time (harder to find)
    # But we need to be strategic
    
    # First, try a few ranks near the lower bound with limited time
    # Then try ranks closer to schoolbook with more time
    
    ranks_to_try = list(range(lower_bound, rank_schoolbook))
    
    # Shuffle to try different ranks, but weight toward middle ranks
    # Actually, let's try in order from low to high, but with time limits
    
    time_per_rank_base = 2.0  # seconds per rank attempt
    
    for rank in ranks_to_try:
        if time.time() > total_deadline - 5:  # Leave 5s safety margin
            break
        
        # Calculate how much time we can spend on this rank
        time_remaining = total_deadline - time.time() - 5
        if time_remaining < 0.5:
            break
        
        # Spend less time on very low ranks (unlikely to succeed)
        # and more on ranks closer to schoolbook
        if rank < 30:
            alloc_time = min(1.0, time_remaining * 0.3)
        elif rank < 45:
            alloc_time = min(3.0, time_remaining * 0.5)
        else:
            alloc_time = min(5.0, time_remaining * 0.7)
        
        if alloc_time < 0.1:
            break
        
        result = _structured_search(n, m, p, rank, time_limit=alloc_time)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            # Found a valid decomposition; continue to see if we can find lower
            # But since we're going from low to high, this is already the lowest
            # So break
            break
    
    return best_result
