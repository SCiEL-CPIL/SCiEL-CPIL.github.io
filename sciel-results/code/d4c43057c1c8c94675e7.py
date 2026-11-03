"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    """
    terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    return terms

def _strassen_block_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    Split A into two 2x2 blocks, B into 2x6 halves with 3 col blocks of width 2.
    Each (2x2)*(2x2) uses 7 Strassen terms. Total: 2*3*7 = 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2x2_terms()
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for t in range(7):
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=2, m=4, p=6, trials=20):
    """Verify a decomposition with integer arithmetic."""
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    
    for trial in range(trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback: rank-48 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _matmul(A, B):
    """Matrix multiplication using numpy."""
    return np.dot(A, B)

def _verify_decomposition_np(U, V, W, n=2, m=4, p=6, trials=10, seed=None):
    """Verify decomposition using numpy with exact integer arithmetic."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    
    for trial in range(trials):
        A = rng.randint(-5, 6, size=(n, m))
        B = rng.randint(-5, 6, size=(m, p))
        
        C_exp = np.zeros((n, p), dtype=object)
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i, j] * B[j, k]
                C_exp[i, k] = s
        
        C_comp = np.zeros((n, p), dtype=object)
        for r in range(rank):
            u_val = 0
            for i in range(n):
                for j in range(m):
                    u_val += U[i * m + j, r] * A[i, j]
            v_val = 0
            for j in range(m):
                for k in range(p):
                    v_val += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += W[k * n + i, r] * u_val * v_val
        
        if not all(C_exp[i, k] == C_comp[i, k] for i in range(n) for k in range(p)):
            return False
    return True

def _try_random_reduction(U, V, W, target_rank, n=2, m=4, p=6, max_attempts=50, deadline=None):
    """
    Try to reduce the rank by random linear combinations of columns.
    Returns a valid decomposition of rank <= target_rank, or None.
    """
    current_rank = U.shape[1]
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try to reduce by 1 rank
        # Pick a random column to eliminate
        cols_to_keep = list(range(current_rank))
        random.shuffle(cols_to_keep)
        
        # Try eliminating different columns
        for elim_idx in range(min(10, current_rank)):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Try to express column elim_idx as a linear combination of others
            # using small integer coefficients
            # Simple approach: try random small coefficient vectors
            
            # Pick a subset of other columns to combine
            other_cols = [c for c in range(current_rank) if c != elim_idx]
            random.shuffle(other_cols)
            
            # Try using first few other columns
            for n_comb in range(1, min(5, len(other_cols)) + 1):
                if deadline is not None and time.time() > deadline - 5:
                    break
                
                comb_cols = other_cols[:n_comb]
                
                # Try random small integer coefficients
                for _ in range(20):
                    if deadline is not None and time.time() > deadline - 5:
                        break
                    
                    coeffs = [random.randint(-3, 3) for _ in range(n_comb)]
                    if all(c == 0 for c in coeffs):
                        continue
                    
                    # Check if U[:, elim_idx] == sum(coeffs[c] * U[:, comb_cols[c]])
                    u_match = True
                    for i in range(len(U)):
                        lhs = U[i][elim_idx]
                        rhs = sum(coeffs[c] * U[i][comb_cols[c]] for c in range(n_comb))
                        if lhs != rhs:
                            u_match = False
                            break
                    
                    if not u_match:
                        continue
                    
                    v_match = True
                    for i in range(len(V)):
                        lhs = V[i][elim_idx]
                        rhs = sum(coeffs[c] * V[i][comb_cols[c]] for c in range(n_comb))
                        if lhs != rhs:
                            v_match = False
                            break
                    
                    if not v_match:
                        continue
                    
                    w_match = True
                    for i in range(len(W)):
                        lhs = W[i][elim_idx]
                        rhs = sum(coeffs[c] * W[i][comb_cols[c]] for c in range(n_comb))
                        if lhs != rhs:
                            w_match = False
                            break
                    
                    if not w_match:
                        continue
                    
                    # Found a valid elimination!
                    # Remove column elim_idx
                    new_cols = [c for c in range(current_rank) if c != elim_idx]
                    new_U = [[U[i][c] for c in new_cols] for i in range(len(U))]
                    new_V = [[V[i][c] for c in new_cols] for i in range(len(V))]
                    new_W = [[W[i][c] for c in new_cols] for i in range(len(W))]
                    
                    # Verify the result
                    if _verify_decomposition({"U": new_U, "V": new_V, "W": new_W}, n, m, p, trials=10):
                        return {"U": new_U, "V": new_V, "W": new_W}
    
    return None

def _try_random_linear_combination(U, V, W, n=2, m=4, p=6, max_attempts=100, deadline=None):
    """
    Try to find a lower-rank decomposition by random linear combinations of columns.
    The idea: if columns are linearly dependent in a way that preserves the tensor,
    we can reduce the rank.
    """
    current_rank = U.shape[1]
    best = {"U": U, "V": V, "W": W}
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try to eliminate one column by expressing it as a linear combination
        # of others with small integer coefficients
        elim_idx = random.randint(0, current_rank - 1)
        other_cols = [c for c in range(current_rank) if c != elim_idx]
        random.shuffle(other_cols)
        
        # Try using a random subset of other columns
        subset_size = random.randint(1, min(6, len(other_cols)))
        comb_cols = other_cols[:subset_size]
        
        # Generate random small integer coefficients
        coeffs = [random.randint(-3, 3) for _ in range(subset_size)]
        if all(c == 0 for c in coeffs):
            continue
        
        # Check if the linear combination matches
        u_match = True
        for i in range(len(U)):
            lhs = U[i][elim_idx]
            rhs = sum(coeffs[c] * U[i][comb_cols[c]] for c in range(subset_size))
            if lhs != rhs:
                u_match = False
                break
        
        if not u_match:
            continue
        
        v_match = True
        for i in range(len(V)):
            lhs = V[i][elim_idx]
            rhs = sum(coeffs[c] * V[i][comb_cols[c]] for c in range(subset_size))
            if lhs != rhs:
                v_match = False
                break
        
        if not v_match:
            continue
        
        w_match = True
        for i in range(len(W)):
            lhs = W[i][elim_idx]
            rhs = sum(coeffs[c] * W[i][comb_cols[c]] for c in range(subset_size))
            if lhs != rhs:
                w_match = False
                break
        
        if not w_match:
            continue
        
        # Found a valid elimination!
        new_cols = [c for c in range(current_rank) if c != elim_idx]
        new_U = [[U[i][c] for c in new_cols] for i in range(len(U))]
        new_V = [[V[i][c] for c in new_cols] for i in range(len(V))]
        new_W = [[W[i][c] for c in new_cols] for i in range(len(W))]
        
        # Verify
        if _verify_decomposition({"U": new_U, "V": new_V, "W": new_W}, n, m, p, trials=10):
            best = {"U": new_U, "V": new_V, "W": new_W}
            U, V, W = new_U, new_V, new_W
            current_rank = len(U[0])
            # Continue trying to reduce further
            break
    
    return best

def _construct_with_reduction():
    """
    Construct the best decomposition by starting with Strassen and trying to reduce rank.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with Strassen rank-42
    result = _strassen_block_decomposition()
    best_rank = 42
    best_result = result
    
    # Try random reduction
    reduced = _try_random_linear_combination(
        result['U'], result['V'], result['W'],
        n, m, p,
        max_attempts=200,
        deadline=deadline
    )
    
    if reduced is not None:
        new_rank = len(reduced['U'][0])
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = reduced
    
    return best_result

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition, then try to reduce rank
    via random linear combinations of columns.
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 120
    
    # Try the reduction approach
    try:
        result = _construct_with_reduction()
        
        # Verify the decomposition
        if _verify_decomposition(result, 2, 4, 6, trials=20):
            return result
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        return _strassen_block_decomposition()
    
    # Fallback to Strassen (rank 42)
    strassen = _strassen_block_decomposition()
    if _verify_decomposition(strassen, 2, 4, 6, trials=20):
        return strassen
    
    # Last resort: schoolbook (rank 48)
    schoolbook = _schoolbook_decomposition()
    return schoolbook
# EVOLVE-BLOCK-END
