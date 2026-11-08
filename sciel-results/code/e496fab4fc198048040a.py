"""Matrix multiplication 3x3 by 3x5 - find rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the multiplication tensor."""
    T = _build_tensor(n, m, p)
    # Reconstruct: T_recon[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    # But U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    rank = U.shape[1]
    # Efficient check: for each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that non-tensor entries are zero
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _try_merge_terms(U, V, W, n, m, p, max_tries=500):
    """Try to reduce rank by finding linear dependencies among the rank-1 terms."""
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    
    # Try to eliminate one term at a time by absorbing it into others
    for col_to_remove in range(rank):
        if time.time() > DEADLINE - 5:
            break
        # Try to see if term col_to_remove can be expressed as combination of others
        # This is complex; instead try simple pairwise merges
        for other_col in range(rank):
            if other_col == col_to_remove:
                continue
            # Try: can we adjust U, V, W for these two columns to merge?
            # Simple approach: check if removing one term and adjusting others still works
            # This is a heuristic - try setting one column to zero and see if we can fix
            pass
    
    # Alternative: try to find if two terms are linearly dependent in a useful way
    # For now, return the same decomposition (no improvement found)
    return U, V, W

def _structured_search(n, m, p, target_rank, time_limit=30):
    """Try to find a decomposition of given target_rank using structured patterns."""
    start_time = time.time()
    
    # Try patterns where some terms share structure
    # Pattern: group by i, and try to factor the (j,k) part
    # For each i, we have a 3x5 matrix multiplication slice
    # The full tensor is block-structured
    
    # Try: for each i, use a rank-r_i decomposition of the 3x3x5 sub-tensor
    # But the W index couples i and k
    
    # Let's try: assume W[k*n+i, c] has structure
    # W is indexed by (k,i) where k in 0..4, i in 0..2
    
    # Try a decomposition where for each i, the terms involving i use a common pattern
    # in the (j,k) plane
    
    # Heuristic: try rank = 44, 43, ... and see if we can find one
    # This is very hard in general; use random restarts with constraints
    
    # For now, try a specific structured approach:
    # The 3x3x3 part (k=0,1,2) might have rank 23 (known for 3x3x3 is 23)
    # The remaining k=3,4 adds 2*3*3=18 terms, total 23+18=41?
    # But the coupling through i makes this non-trivial
    
    # Let's just try to verify if a given rank is achievable via random search
    # This is unlikely to succeed quickly for large rank reductions
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to improve by searching for lower rank decompositions
    # Known: for square n x n x n, best known is less than n^3
    # For 3x3x3, best known is 23 (Landsberg)
    # For our case 3x3x3x5, let's try various ranks
    
    # Try ranks from 44 down to 30
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > DEADLINE - 10:
            break
        
        # Try to find a decomposition of target_rank
        # Use a simple approach: start from schoolbook and try to zero out columns
        # while maintaining validity
        
        # Method: try to eliminate one term at a time
        U_curr, V_curr, W_curr = U.copy(), V.copy(), W.copy()
        current_rank = best_rank
        
        for attempt in range(200):
            if time.time() > DEADLINE - 10:
                break
            if current_rank <= target_rank:
                break
            
            # Try to remove a random column
            cols_to_try = list(range(current_rank))
            np.random.shuffle(cols_to_try)
            
            found = False
            for col in cols_to_try:
                # Try removing this column by seeing if the tensor is still satisfied
                # with this column zeroed out (and others adjusted)
                # Simple test: zero out this column and check
                U_test = U_curr.copy()
                V_test = V_curr.copy()
                W_test = W_curr.copy()
                U_test[:, col] = 0
                V_test[:, col] = 0
                W_test[:, col] = 0
                
                if _verify(U_test, V_test, W_test, n, m, p):
                    # Successfully removed a term
                    # Remove the column
                    mask = [i for i in range(current_rank) if i != col]
                    U_curr = U_curr[:, mask]
                    V_curr = V_curr[:, mask]
                    W_curr = W_curr[:, mask]
                    current_rank -= 1
                    found = True
                    break
            
            if not found:
                break
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            U, V, W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    
    # Convert to list-of-lists with exact integer/half-integer values
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                v = round(val * 2) / 2
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    # Final verification
    if not _verify(np.array(U_out, dtype=float), np.array(V_out, dtype=float), 
                   np.array(W_out, dtype=float), n, m, p):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        U_out = to_list(U_sb)
        V_out = to_list(V_sb)
        W_out = to_list(W_sb)
    
    return {"U": U_out, "V": V_out, "W": W_out}
