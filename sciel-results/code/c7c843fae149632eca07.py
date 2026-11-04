"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Search for a low-rank exact tensor decomposition of the 4x4 * 4x9 multiplication tensor.
    
    The tensor T has indices (i, j, k) for i in [0,4), j in [0,4), k in [0,9).
    We seek U (shape 16 x r), V (shape 36 x r), W (shape 36 x r) such that:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    
    Strategy: Use a greedy/local search approach with integer coefficients.
    Start from schoolbook and try to reduce rank by finding redundant terms
    or using structured decompositions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p  # 144
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition is exact
    def verify(U, V, W):
        """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        n, m, p = DIMENSIONS
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # Build the tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if val != 1:
                        return False
        return True
    
    # Try to find a better decomposition using structured approaches
    
    # Approach 1: Try small ranks with random integer coefficients and local search
    def try_rank(r, max_iters=5000):
        """Try to find a valid decomposition of rank r using local search."""
        n, m, p = DIMENSIONS
        
        # Initialize with random small integers
        np.random.seed(r)  # Deterministic for reproducibility
        U = np.random.randint(-2, 3, size=(n*m, r))
        V = np.random.randint(-2, 3, size=(m*p, r))
        W = np.random.randint(-2, 3, size=(p*n, r))
        
        # Compute current errors
        def compute_errors(U, V, W):
            errs = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        errs[(i,j,k)] = 1 - val
            return errs
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err_count = sum(1 for e in compute_errors(U, V, W).values() if e != 0)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if best_err_count == 0:
                return best_U, best_V, best_W, r
            
            # Pick a random error entry
            errors = compute_errors(U, V, W)
            err_keys = [k for k, v in errors.items() if v != 0]
            if not err_keys:
                return U, V, W, r
            
            (i, j, k) = err_keys[np.random.randint(len(err_keys))]
            target_err = errors[(i,j,k)]
            
            # Try modifying one random element in one of the three matrices
            mat_choice = np.random.randint(3)
            c = np.random.randint(r)
            
            if mat_choice == 0:
                idx = i * m + j
                old_val = U[idx, c]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val == old_val:
                        continue
                    U[idx, c] = new_val
                    # Quick check: only affects entries (i, j, *)
                    ok = True
                    for k2 in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k2, :] * W[k2*n+i, :])
                        if val != 1:
                            ok = False
                            break
                    if ok:
                        new_errs = compute_errors(U, V, W)
                        new_count = sum(1 for e in new_errs.values() if e != 0)
                        if new_count < best_err_count:
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            best_err_count = new_count
                        else:
                            U[idx, c] = old_val
                    else:
                        U[idx, c] = old_val
            
            elif mat_choice == 1:
                idx = j * p + k
                old_val = V[idx, c]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val == old_val:
                        continue
                    V[idx, c] = new_val
                    ok = True
                    for i2 in range(n):
                        val = np.sum(U[i2*m+j, :] * V[j*p+k, :] * W[k*n+i2, :])
                        if val != 1:
                            ok = False
                            break
                    if ok:
                        new_errs = compute_errors(U, V, W)
                        new_count = sum(1 for e in new_errs.values() if e != 0)
                        if new_count < best_err_count:
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            best_err_count = new_count
                        else:
                            V[idx, c] = old_val
                    else:
                        V[idx, c] = old_val
            
            else:
                idx = k * n + i
                old_val = W[idx, c]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    if new_val == old_val:
                        continue
                    W[idx, c] = new_val
                    ok = True
                    for j2 in range(m):
                        val = np.sum(U[i*m+j2, :] * V[j2*p+k, :] * W[k*n+i, :])
                        if val != 1:
                            ok = False
                            break
                    if ok:
                        new_errs = compute_errors(U, V, W)
                        new_count = sum(1 for e in new_errs.values() if e != 0)
                        if new_count < best_err_count:
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            best_err_count = new_count
                        else:
                            W[idx, c] = old_val
                    else:
                        W[idx, c] = old_val
        
        if best_err_count == 0:
            return best_U, best_V, best_W, r
        return None
    
    # Approach 2: Structured decomposition using Kronecker-like patterns
    def try_structured(r):
        """Try structured decompositions based on known patterns."""
        n, m, p = DIMENSIONS
        
        # For 4x4 * 4x9, try using the fact that 4 = 2*2
        # Split indices into blocks
        # This is a heuristic - may not work but worth trying
        
        # Try rank 64 (half of schoolbook) with a structured pattern
        if r == 64:
            U = np.zeros((n*m, r), dtype=np.int64)
            V = np.zeros((m*p, r), dtype=np.int64)
            W = np.zeros((p*n, r), dtype=np.int64)
            
            # Use a 2x2 block structure
            # i = 2*i1 + i2, j = 2*j1 + j2 where i1,j1 in {0,1}, i2,j2 in {0,1}
            col = 0
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k in range(p):
                                i = 2*i1 + i2
                                j = 2*j1 + j2
                                U[i*m+j, col] = 1
                                V[j*p+k, col] = 1
                                W[k*n+i, col] = 1
                                col += 1
            
            if col == r and verify(U, V, W):
                return U, V, W, r
        
        return None
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    # Try various ranks from low to high
    # Known lower bounds suggest rank >= 16*36/9 = 64 (rough)
    # But exact bounds are hard. Try a range.
    for r in [64, 72, 80, 90, 100, 110, 120, 130]:
        if time.time() > deadline:
            break
        
        # Try structured first
        result = try_structured(r)
        if result:
            if result[3] < best_result[3]:
                best_result = result
            break
        
        # Try local search
        result = try_rank(r, max_iters=2000)
        if result:
            if result[3] < best_result[3]:
                best_result = result
            break
    
    U, V, W, rank = best_result
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
