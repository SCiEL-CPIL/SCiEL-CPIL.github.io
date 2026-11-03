"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n_, m_, p_ = n, m, p
        # Check shapes
        if len(U) != n_ * m_:
            return False
        if len(V) != m_ * p_:
            return False
        if len(W) != p_ * n_:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        if rank_ <= 0:
            return False
        
        # The tensor M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # should equal 1 for all valid i, j, k
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    expected = 1
                    actual = 0.0
                    for r in range(rank_):
                        actual += U[i * m_ + j][r] * V[j * p_ + k][r] * W[k * n_ + i][r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def make_result(U, V, W):
        """Convert to list-of-lists format with cleaned values."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up near-integer and half-integer values
        def clean_val(x):
            r = round(x, 10)
            if abs(x - r) < 1e-8:
                return r
            rh = round(x * 2) / 2.0
            if abs(x - rh) < 1e-8:
                return rh
            return x
        U_list = [[clean_val(x) for x in row] for row in U_list]
        V_list = [[clean_val(x) for x in row] for row in V_list]
        W_list = [[clean_val(x) for x in row] for row in W_list]
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    # Try structured decompositions based on the rectangular structure
    # For 3x3 * 3x7, we can think of it as 7 separate 3x3 * 3x1 multiplications
    # but with shared structure in A
    
    # Strategy: Use random search with small coefficient sets for various ranks
    random.seed(42)
    
    # Try different coefficient sets and rank targets
    coeff_sets = [
        [-1, 0, 0, 0, 1],           # Very sparse
        [-1, 0, 1],                   # Basic
        [-2, -1, 0, 1, 2],           # Wider
        [-1, 0, 0, 1, 1],            # More 1s
        [-1, 0, 0, 0, 0, 1],         # Very sparse
    ]
    
    # Try ranks from just below schoolbook downward
    # For (3,3,7), known lower bounds suggest rank could be significantly lower
    # Let's try a range of ranks
    
    ranks_to_try = list(range(62, 40, -1))  # Try ranks 62 down to 41
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Choose coefficient set
            coeff_set = coeff_sets[attempt % len(coeff_sets)]
            
            # Random initialization
            U = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W):
                result = make_result(U, V, W)
                best_result = result
                best_rank = target_rank
                break
        
        # If we found a valid decomposition at this rank, try lower
        if best_rank == target_rank:
            continue
    
    # Also try a more targeted approach: exploit the structure
    # For each (i,k) pair, we need sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of 63 equations (3*3*7) in rank*21 variables
    
    # Try a greedy/structured approach: build up the decomposition column by column
    # Each "column" r contributes a rank-1 tensor U_col ⊗ V_col ⊗ W_col
    
    # Try to find a decomposition where each rank-1 term has simple structure
    # e.g., U_col is a single 1, V_col is a single 1, W_col is a single 1 (schoolbook)
    # or U_col has a few 1s, etc.
    
    # Alternative: try to find a decomposition with rank = n*m + (p-1)*something
    # For square case, Strassen gives 7 for 2x2x2
    # For 3x3x3, best known is 23 (Lader)
    # For 3x3x7, we might be able to do better than 63
    
    # Try a specific structured pattern: group the 7 columns
    # Columns 0-2: use one pattern, columns 3-6: use another
    
    # Let's try a random search with more attempts for the most promising ranks
    # Focus on ranks that might be achievable
    
    for target_rank in [60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Mix of sparse and denser coefficients
            if attempt % 3 == 0:
                coeff_set = [-1, 0, 0, 0, 1]
            elif attempt % 3 == 1:
                coeff_set = [-1, 0, 1]
            else:
                coeff_set = [-2, -1, 0, 1, 2]
            
            U = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W):
                result = make_result(U, V, W)
                best_result = result
                best_rank = target_rank
                break
    
    # Final: try a few more targeted searches
    for target_rank in range(best_rank - 1, max(35, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            coeff_set = [-1, 0, 1]
            U = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(coeff_set) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W):
                result = make_result(U, V, W)
                best_result = result
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
