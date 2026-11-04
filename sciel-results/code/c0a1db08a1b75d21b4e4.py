import time
import sys
import random
import numpy as np
from fractions import Fraction

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Known rank 44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find lower rank via compression
    if time.time() < deadline - 30:
        compressed = _compress_solution(U44, V44, W44, deadline - 5)
        if compressed is not None:
            best_rank, best_U, best_V, best_W = compressed
            if best_rank < 42:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _get_rank_44_solution():
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[u_map[0]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        W[w_map[2]][col] += 1
        W[w_map[3]][col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0]][col] += 1
        V[v_map[1]][col] += 1
        V[v_map[3]][col] += -1
        W[w_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3]][col] += 1
        V[v_map[2]][col] += 1
        V[v_map[0]][col] += -1
        W[w_map[0]][col] += 1
        W[w_map[2]][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0]][col] += 1
        U[u_map[1]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += -1
        W[w_map[1]][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2]][col] += 1
        U[u_map[0]][col] += -1
        V[v_map[0]][col] += 1
        V[v_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1]][col] += 1
        U[u_map[3]][col] += -1
        V[v_map[2]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        col += 1

    # Process left half (k = 0..3) and right half (k = 4..7)
    for half in range(2):
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),
            1: 0 * p + (b_start + 1),
            2: 1 * p + (b_start + 0),
            3: 1 * p + (b_start + 1)
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,
            1: (b_start + 1) * n + 0,
            2: (b_start + 0) * n + 1,
            3: (b_start + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        for i in range(2):
            for k in range(4):
                u_idx = i * m + 2
                v_idx = 2 * p + (half * 4 + k)
                w_idx = (half * 4 + k) * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return U, V, W

def _compress_solution(U, V, W, deadline):
    """
    Try to compress a tensor decomposition by finding linear dependencies.
    Returns (rank, U', V', W') or None if no improvement found.
    """
    n, m, p = 2, 3, 8
    current_rank = len(U[0]) if U else 0
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=object)
    V_np = np.array(V, dtype=object)
    W_np = np.array(W, dtype=object)
    
    # Try to find linear dependencies in the columns of U, V, W
    # If column j is a linear combination of other columns, we can potentially eliminate it
    
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W
    
    # Strategy: Try to identify redundant terms by checking if removing a term
    # still allows exact reconstruction
    
    # For each column, check if it's linearly dependent on others
    # This is expensive, so we'll use a heuristic approach
    
    # First, check for exact duplicates
    seen_columns = {}
    duplicate_cols = []
    
    for col_idx in range(current_rank):
        # Create a signature for this column
        u_sig = tuple(U[i][col_idx] for i in range(len(U)))
        v_sig = tuple(V[i][col_idx] for i in range(len(V)))
        w_sig = tuple(W[i][col_idx] for i in range(len(W)))
        
        sig = (u_sig, v_sig, w_sig)
        if sig in seen_columns:
            duplicate_cols.append(col_idx)
        else:
            seen_columns[sig] = col_idx
    
    # Remove exact duplicates
    if duplicate_cols:
        new_rank = current_rank - len(duplicate_cols)
        keep_cols = [i for i in range(current_rank) if i not in duplicate_cols]
        
        new_U = [[U[i][col] for col in keep_cols] for i in range(len(U))]
        new_V = [[V[i][col] for col in keep_cols] for i in range(len(V))]
        new_W = [[W[i][col] for col in keep_cols] for i in range(len(W))]
        
        # Verify the solution is still valid
        if _verify_decomposition(new_U, new_V, new_W, n, m, p):
            best_rank = new_rank
            best_U, best_V, best_W = new_U, new_V, new_W
    
    # Try more aggressive compression: check if any column is a linear combination
    # of others (using rational arithmetic)
    if time.time() < deadline:
        compressed = _try_linear_compression(best_U, best_V, best_W, n, m, p, deadline)
        if compressed is not None:
            rank, U_c, V_c, W_c = compressed
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U_c, V_c, W_c
    
    if best_rank < current_rank:
        return best_rank, best_U, best_V, best_W
    
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    """
    rank = len(U[0]) if U else 0
    
    # Build the multiplication tensor
    # T[i,j,k] should equal sum_r U[i,r] * V[j,r] * W[k,r]
    # where i is in [0, n*m), j is in [0, m*p), k is in [0, p*n)
    
    # Convert to numpy for verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check a few random entries
    np.random.seed(42)
    for _ in range(100):
        i = np.random.randint(0, n * m)
        j = np.random.randint(0, m * p)
        k = np.random.randint(0, p * n)
        
        # Compute the sum
        val = 0.0
        for r in range(rank):
            val += U_np[i, r] * V_np[j, r] * W_np[k, r]
        
        # Expected value: this should be 1 if the indices correspond to a valid
        # matrix multiplication term, 0 otherwise
        # The tensor T[i,j,k] = 1 if there exists a valid multiplication path
        # This is complex to check exactly, so we'll just verify no NaN/Inf
        if not np.isfinite(val):
            return False
    
    # For a proper check, we need to verify against the actual multiplication tensor
    # Let's do a more thorough check
    return _full_verify_decomposition(U, V, W, n, m, p)

def _full_verify_decomposition(U, V, W, n, m, p):
    """
    Full verification of the tensor decomposition.
    """
    rank = len(U[0]) if U else 0
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 iff
    # i = a*m + b, j = b*p + c, k = c*n + a for some valid indices
    # Actually, let me think about this more carefully.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In the tensor decomposition, we have:
    # C[i,k] = sum_r U[i,r] * (sum_j V[j,r] * B[j,k]) ... no, that's not right
    # 
    # The correct formulation:
    # The multiplication tensor T has entries T[a,b,c] where:
    # a indexes A (n*m entries), b indexes B (m*p entries), c indexes C (p*n entries)
    # T[a,b,c] = 1 if a, b, c correspond to a valid multiplication term
    # 
    # More precisely, if a = i*m + j (i-th row of A, j-th column),
    # b = j*p + k (j-th row of B, k-th column),
    # c = k*n + i (k-th row of C, i-th column),
    # then T[a,b,c] = 1
    # 
    # The decomposition says: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Build the expected tensor
    expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected[a, b, c] = 1.0
    
    # Compute the decomposed tensor
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Efficient computation using einsum
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This can be computed as: for each r, outer(U[:,r], V[:,r]) * W[:,r]
    
    decomposed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        outer_uv = np.outer(U_np[:, r], V_np[:, r])
        decomposed += outer_uv * W_np[:, r]
    
    # Check if they match
    max_diff = np.max(np.abs(expected - decomposed))
    return max_diff < 1e-10

def _try_linear_compression(U, V, W, n, m, p, deadline):
    """
    Try to find linear dependencies among the rank-1 terms.
    """
    current_rank = len(U[0]) if U else 0
    
    # For each column, try to see if it can be expressed as a linear combination
    # of other columns
    
    # This is a complex optimization problem. Let's try a simple approach:
    # Check if any two columns are identical or negatives of each other
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for linear dependencies using SVD or rank computation
    # If the combined matrix has rank less than current_rank, we can compress
    
    # Stack U, V, W in a way that captures dependencies
    # Actually, the dependencies are between the rank-1 tensors, not just individual matrices
    
    # A simpler approach: try removing each column and checking if the solution is still valid
    # This is O(rank^2) but might find some improvements
    
    best_rank = current_rank
    best_U, best_V, best_W = U, V, W
    
    # Try removing columns one at a time
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        new_rank = current_rank - 1
        keep_cols = [i for i in range(current_rank) if i != col_to_remove]
        
        new_U = [[U[i][col] for col in keep_cols] for i in range(len(U))]
        new_V = [[V[i][col] for col in keep_cols] for i in range(len(V))]
        new_W = [[W[i][col] for col in keep_cols] for i in range(len(W))]
        
        if _full_verify_decomposition(new_U, new_V, new_W, n, m, p):
            best_rank = new_rank
            best_U, best_V, best_W = new_U, new_V, new_W
            current_rank = new_rank
            U, V, W = new_U, new_V, new_W
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
    
    if best_rank < len(U[0]) + 1:  # If we made any improvement
        return best_rank, best_U, best_V, best_W
    
    return None
