"""Improved program for matrix multiplication 2,3,6 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 6]
import numpy as np
from fractions import Fraction
import time
import itertools

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, r):
    """Fast verification using matrix operations.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Check: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can compute this as: for each c, outer product structure
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Reshape for batch computation
    # U reshaped: (n, m, r)
    # V reshaped: (m, p, r)  
    # W reshaped: (p, n, r)
    
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    # For each (i,j,k): sum over c of U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is a contraction over c
    
    # Compute: result[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = sum_c U_r[i,j,c] * (V_r[j,k,c] * W_r[k,i,c])
    
    # For each c: contribution is U_r[:,:,c] * (V_r * W_r) contracted appropriately
    # V_r[j,k,c] * W_r[k,i,c] for fixed c: sum over k of V_r[j,k,c]*W_r[k,i,c]
    # That gives us a (m, n) matrix for each c
    # Then result[i,j] = sum_c U_r[i,j,c] * (V_r @ W_r)[j,i,c]... 
    
    # Actually let me think more carefully:
    # result[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Let's compute it directly but vectorized:
    # Create arrays of shape (n, m, p, r)
    # A = U_r[:, :, None, :] broadcast to (n, m, p, r) -- no, U_r is (n,m,r)
    # We need: for each (i,j,k,c): U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Let's do it in steps:
    # Step 1: Compute M1[j,k,c] = V_r[j,k,c] * W_r[k,i,c] -- but this depends on i
    # So for each i: M1_i[j,k,c] = V_r[j,k,c] * W_r[k,i,c]
    # Then result[i,j,k] = sum_c U_r[i,j,c] * M1_i[j,k,c]
    
    # Let's use einsum or manual loops for small dimensions
    result = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                result[i,j,k] = val
    
    # Check all are 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if result[i,j,k] != 1:
                    return False
    return True


def _verify_vectorized(U, V, W, n, m, p, r):
    """Vectorized verification."""
    # U: (n*m, r) -> reshape to (n, m, r)
    # V: (m*p, r) -> reshape to (m, p, r)
    # W: (p*n, r) -> reshape to (p, n, r)
    
    U_r = U.reshape(n, m, r)  # (n, m, r)
    V_r = V.reshape(m, p, r)  # (m, p, r)
    W_r = W.reshape(p, n, r)  # (p, n, r)
    
    # We want: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Use einsum: 
    # First, for each c, compute the contribution
    # Let's create arrays:
    # A[i,j,c] = U_r[i,j,c]  shape (n,m,r)
    # B[j,k,c] = V_r[j,k,c]  shape (m,p,r)
    # C[k,i,c] = W_r[k,i,c]  shape (p,n,r)
    
    # T[i,j,k] = sum_c A[i,j,c] * B[j,k,c] * C[k,i,c]
    
    # This can be computed as:
    # For each c: outer product of A[:,:,c], B[:,:,c], C[:,:,c] but with index matching
    # Actually: T = sum_c (A_c ⊗ B_c ⊗ C_c) with proper index alignment
    
    # Let's use a different approach:
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Step 1: Compute M[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c]
    # This is: for each (j,c): U_r[:,j,c] (shape n) * V_r[j,:,c] (shape p)
    # M[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c]
    
    # Using einsum: 'ijc,jkc->ijk'
    M = np.einsum('ijc,jkc->ijk', U_r, V_r)  # shape (n, m, p)
    
    # Step 2: T[i,j,k] = sum_c ... wait, we already summed over c in M
    # No, M[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c]
    # But we need: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # These are different! We can't just multiply M by W.
    
    # Let me redo: 
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # This is a 4-way contraction. Let's do it step by step:
    # For each (i,k): T[:,j,k] involves W_r[k,i,c]
    
    # Alternative: 
    # T[i,j,k] = sum_c U_r[i,j,c] * (V_r[j,k,c] * W_r[k,i,c])
    
    # Let P[j,k,i,c] = V_r[j,k,c] * W_r[k,i,c]  shape (m, p, n, r)
    # Then T[i,j,k] = sum_c U_r[i,j,c] * P[j,k,i,c]
    
    # P[j,k,i,c] = V_r[j,k,c] * W_r[k,i,c]
    # Using einsum: 'jkc,kic->jkic'
    P = np.einsum('jkc,kic->jkic', V_r, W_r)  # shape (m, p, n, r)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * P[j,k,i,c]
    # Using einsum: 'ijc,jkic->ijk'
    T = np.einsum('ijc,jkic->ijk', U_r, P)  # shape (n, m, p)
    
    return np.all(T == 1)


def _is_half_integer(x):
    """Check if x is an integer or half-integer (multiple of 0.5)."""
    return abs(x * 2 - round(x * 2)) < 1e-9


def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2


def _try_rank_structured(n, m, p, r, max_attempts=50):
    """Try to find a rank-r decomposition with structured initialization."""
    deadline = time.time() + 50  # 50 seconds per rank attempt
    
    # Strategy 1: Block structure based on splitting dimensions
    # For 2x3 * 3x6, we can try various block decompositions
    
    # Precompute index mappings
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Try different structured patterns
    patterns = []
    
    # Pattern 1: Identity-like (schoolbook subset)
    # Pattern 2: Strassen-like with +/- 1 entries
    # Pattern 3: Random with structured zeros
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Choose a strategy based on attempt number
        strategy = attempt % 4
        
        if strategy == 0:
            # Dense random with {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            W = np.random.randint(-1, 2, size=(p*n, r))
        elif strategy == 1:
            # Sparse: each row has ~2 nonzeros
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            for i in range(n*m):
                cols = np.random.choice(r, size=min(2, r), replace=False)
                U[i, cols] = np.random.choice([-1, 1], size=len(cols))
            for i in range(m*p):
                cols = np.random.choice(r, size=min(2, r), replace=False)
                V[i, cols] = np.random.choice([-1, 1], size=len(cols))
            for i in range(p*n):
                cols = np.random.choice(r, size=min(2, r), replace=False)
                W[i, cols] = np.random.choice([-1, 1], size=len(cols))
        elif strategy == 2:
            # Structured: try to exploit the 2x3*3x6 structure
            # Split into blocks
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Try a block diagonal structure
            block_size = max(1, r // 3)
            for b in range(min(3, r // block_size)):
                start = b * block_size
                end = min(start + block_size, r)
                if end <= start:
                    continue
                # Random within block
                U[:, start:end] = np.random.randint(-1, 2, size=(n*m, end-start))
                V[:, start:end] = np.random.randint(-1, 2, size=(m*p, end-start))
                W[:, start:end] = np.random.randint(-1, 2, size=(p*n, end-start))
        else:
            # Half-integer random
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
        
        # Verify
        if _verify_vectorized(U, V, W, n, m, p, r):
            # Check all entries are half-integers
            valid = True
            for mat in [U, V, W]:
                for val in mat:
                    if not _is_half_integer(val):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
    
    return None


def _local_search(U, V, W, n, m, p, r, max_iterations=200):
    """Local search to fix near-miss decompositions."""
    deadline = time.time() + 30
    
    def get_errors(U, V, W):
        """Get the error tensor T[i,j,k] - 1."""
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        P = np.einsum('jkc,kic->jkic', V_r, W_r)
        T = np.einsum('ijc,jkic->ijk', U_r, P)
        return T - 1
    
    errors = get_errors(U, V, W)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        if np.all(np.abs(errors) < 1e-9):
            # Check if all are exactly 1 (within floating point)
            if _verify_vectorized(U, V, W, n, m, p, r):
                return U, V, W, True
        
        # Find the largest error
        max_err_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
        i, j, k = max_err_idx
        err = errors[i, j, k]
        
        if abs(err) < 1e-12:
            continue
        
        # Try to fix by modifying one entry
        # The error contribution from column c: U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want to change this sum by -err
        
        # Try flipping or modifying entries in a random column
        c = np.random.randint(r)
        
        u_val = U[i*m+j, c]
        v_val = V[j*p+k, c]
        w_val = W[k*n+i, c]
        
        current_product = u_val * v_val * w_val
        
        # Try changing one of the three values
        options = []
        
        # Change U
        for new_u in [-1, -0.5, 0, 0.5, 1]:
            if new_u != u_val:
                new_product = new_u * v_val * w_val
                delta = new_product - current_product
                options.append(('U', i*m+j, c, new_u, delta))
        
        # Change V
        for new_v in [-1, -0.5, 0, 0.5, 1]:
            if new_v != v_val:
                new_product = u_val * new_v * w_val
                delta = new_product - current_product
                options.append(('V', j*p+k, c, new_v, delta))
        
        # Change W
        for new_w in [-1, -0.5, 0, 0.5, 1]:
            if new_w != w_val:
                new_product = u_val * v_val * new_w
                delta = new_product - current_product
                options.append(('W', k*n+i, c, new_w, delta))
        
        # Pick the option that reduces the error the most
        best_option = None
        best_reduction = 0
        for opt in options:
            mat_name, row, col, new_val, delta = opt
            # New error would be err - delta (approximately, since other errors change too)
            reduction = abs(err) - abs(err - delta)
            if reduction > best_reduction:
                best_reduction = reduction
                best_option = opt
        
        if best_option is None:
            continue
        
        mat_name, row, col, new_val, delta = best_option
        if mat_name == 'U':
            U[row, col] = new_val
        elif mat_name == 'V':
            V[row, col] = new_val
        else:
            W[row, col] = new_val
        
        errors = get_errors(U, V, W)
    
    # Final check
    if _verify_vectorized(U, V, W, n, m, p, r):
        return U, V, W, True
    return U, V, W, False


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    start_time = time.time()
    deadline = start_time + 130  # Leave margin for 150s limit
    
    # Theoretical lower bounds:
    # Rank >= max(n*p, m*p, n*m) is NOT always true
    # But rank >= max(n, m, p) is trivial
    # For M_{2,3,6}: schoolbook is 36
    # Let's try from low ranks up
    
    # Start from a reasonable lower bound
    # The rank must be at least the max dimension product in some sense
    # Let's try: 12, 14, 16, 18, 20, 24, 28, 30, 32, 34, 36
    
    candidate_ranks = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    best_result = None
    best_rank = n * m * p  # 36
    
    # First, get schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    for r in candidate_ranks:
        if time.time() > deadline:
            break
        
        if r >= best_rank:
            continue
        
        # Try structured search
        result = _try_rank_structured(n, m, p, r, max_attempts=30)
        
        if result is not None:
            U, V, W = result
            # Verify once more
            if _verify_vectorized(U, V, W, n, m, p, r):
                best_result = (U, V, W, r)
                best_rank = r
                # Found a lower rank, continue searching for even lower
                continue
        
        # If not found, try local search from a random initialization
        if time.time() < deadline - 20:
            U = np.random.randint(-1, 2, size=(n*m, r))
            V = np.random.randint(-1, 2, size=(m*p, r))
            W = np.random.randint(-1, 2, size=(p*n, r))
            
            U, V, W, success = _local_search(U, V, W, n, m, p, r, max_iterations=100)
            
            if success:
                if _verify_vectorized(U, V, W, n, m, p, r):
                    best_result = (U, V, W, r)
                    best_rank = r
                    continue
    
    if best_result is None:
        best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    U, V, W, rank = best_result
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
