"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
_start_time = time.time()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > tol:
                    return False
    return True

def _build_base_decomposition():
    """Build the rank-55 base decomposition and return U, V, W as numpy arrays."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00+A10)(B0k0+B1k1) -> C0k0 + C1k1
        U[0*3+0, col] += 1; U[1*3+0, col] += 1
        V[0*10+k0, col] += 1; V[1*10+k1, col] += 1
        W[k0*2+0, col] += 1; W[k1*2+1, col] += 1
        col += 1
        
        # M2: (A10+A11)(B0k0) -> C1k0 - C1k1... let me re-derive Strassen
        # Standard Strassen for C = A*B where A,B are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A01-A10)(B00+B10)
        # M7 = (A10-A00)(B01+B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # But our A is 2x3 and B is 3x2 (for each block). Let me be more careful.
        # In block t, we're computing C[:,k0:k1+1] = A * B[:,k0:k1+1]
        # A is 2x3, B_block is 3x2
        # C_block = A * B_block is 2x2
        # C_block[i,l] = sum_j A[i,j] * B_block[j,l]
        # = A[i,0]*B_block[0,l] + A[i,1]*B_block[1,l] + A[i,2]*B_block[2,l]
        
        # The 7 Strassen terms handle the 2x2 part using A rows 0,1 and B rows 0,1
        # The 4 trivial terms handle A row 2 and B row 2
        
        # Let me just use the standard Strassen for the 2x2 submatrix
        # where A_sub = A[:,0:2] (2x2) and B_sub = B[0:2,:] (2x2)
        # C_sub = A_sub * B_sub
        
        # A_sub indices: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,3,4
        # B_sub indices: B[0,k0], B[0,k1], B[1,k0], B[1,k1] -> V indices 0*10+k0, 0*10+k1, 1*10+k0, 1*10+k1
        # C_sub indices: C[0,k0], C[0,k1], C[1,k0], C[1,k1] -> W indices k0*2+0, k0*2+1, k1*2+0, k1*2+1
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        U[0*3+0, col] += 1; U[1*3+1, col] += 1
        V[0*10+k0, col] += 1; V[1*10+k1, col] += 1
        W[k0*2+0, col] += 1; W[k1*2+1, col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*3+0, col] += 1; U[1*3+1, col] += 1
        V[0*10+k0, col] += 1
        W[k0*2+1, col] += 1; W[k1*2+1, col] -= 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[0*3+0, col] += 1
        V[1*10+k0, col] += 1; V[1*10+k1, col] -= 1
        W[k0*2+0, col] += 1; W[k1*2+0, col] += 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[1*3+1, col] += 1
        V[0*10+k1, col] += 1; V[0*10+k0, col] -= 1
        W[k0*2+0, col] += 1; W[k1*2+0, col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*3+0, col] += 1; U[0*3+1, col] += 1
        V[1*10+k1, col] += 1
        W[k0*2+0, col] -= 1; W[k1*2+0, col] += 1
        col += 1
        
        # M6 = (A01-A10)(B00+B10)
        U[0*3+1, col] += 1; U[1*3+0, col] -= 1
        V[0*10+k0, col] += 1; V[1*10+k0, col] += 1
        W[k1*2+1, col] += 1
        col += 1
        
        # M7 = (A10-A00)(B01+B11)
        U[1*3+0, col] += 1; U[0*3+0, col] -= 1
        V[0*10+k1, col] += 1; V[1*10+k1, col] += 1
        W[k0*2+0, col] += 1
        col += 1
        
        # W00: A02 * B2,k0
        U[0*3+2, col] += 1
        V[2*10+k0, col] += 1
        W[k0*2+0, col] += 1
        col += 1
        
        # W01: A02 * B2,k1
        U[0*3+2, col] += 1
        V[2*10+k1, col] += 1
        W[k1*2+0, col] += 1
        col += 1
        
        # W10: A12 * B2,k0
        U[1*3+2, col] += 1
        V[2*10+k0, col] += 1
        W[k0*2+1, col] += 1
        col += 1
        
        # W11: A12 * B2,k1
        U[1*3+2, col] += 1
        V[2*10+k1, col] += 1
        W[k1*2+1, col] += 1
        col += 1
    
    return U, V, W

def _try_eliminate_term(U, V, W, term_idx, n, m, p):
    """Try to express term term_idx as a linear combination of the other terms.
    If successful, return a new rank-54 decomposition."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    # We want to find coefficients c_r for r != term_idx such that:
    # sum_{r != term_idx} c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = U[:,term_idx] ⊗ V[:,term_idx] ⊗ W[:,term_idx]
    # 
    # This is a linear system in the variables c_r.
    # The tensor space has dimension n*m*p = 60.
    # We have 54 unknowns (c_r for r != term_idx) and 60 equations.
    # This is overdetermined, so a solution may not exist.
    # But if the 55 terms are linearly dependent, there IS a solution.
    
    # Build the 60x54 matrix M where M[entry, r] = (U[:,r] ⊗ V[:,r] ⊗ W[:,r])[entry]
    # and the vector b = U[:,term_idx] ⊗ V[:,term_idx] ⊗ W[:,term_idx]
    
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entries.append((i, j, k))
    
    num_entries = len(entries)
    other_indices = [r for r in range(rank) if r != term_idx]
    num_other = len(other_indices)
    
    M = np.zeros((num_entries, num_other))
    b = np.zeros(num_entries)
    
    for idx, (i, j, k) in enumerate(entries):
        b[idx] = U[i*m+j, term_idx] * V[j*p+k, term_idx] * W[k*n+i, term_idx]
        for oi, r in enumerate(other_indices):
            M[idx, oi] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Solve M * c = b
    try:
        c, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
    except:
        return None
    
    # Check if the solution is valid
    if np.linalg.norm(M @ c - b) > 1e-8:
        return None
    
    # If valid, construct the new decomposition
    # The new decomposition has the same terms except term_idx is removed
    # and all other terms are scaled by their coefficients
    # Wait, that's not right. We need:
    # sum_{r != term_idx} c_r * t_r = t_{term_idx}
    # So: t_{term_idx} - sum_{r != term_idx} c_r * t_r = 0
    # This means the 55 terms are linearly dependent.
    # But we want to REMOVE one term, not rescale.
    
    # Actually, if sum_{r != term_idx} c_r * t_r = t_{term_idx}, then
    # sum_{all r} t_r = sum_{r != term_idx} (1 + c_r) * t_r + (1 - 1) * t_{term_idx}
    # No wait: sum_{all} t_r = t_{term_idx} + sum_{r != term_idx} t_r
    # = sum_{r != term_idx} c_r * t_r + sum_{r != term_idx} t_r
    # = sum_{r != term_idx} (c_r + 1) * t_r
    # So we can replace the 55 terms with 54 terms where term r (r != term_idx) 
    # is replaced by (c_r + 1) * t_r.
    
    # But this only works if c_r + 1 != 0 for all r. If c_r + 1 = 0 for some r,
    # that term disappears entirely, which is even better.
    
    new_U = np.zeros((n * m, num_other), dtype=float)
    new_V = np.zeros((m * p, num_other), dtype=float)
    new_W = np.zeros((p * n, num_other), dtype=float)
    
    for oi, r in enumerate(other_indices):
        scale = c[oi] + 1
        if abs(scale) < 1e-12:
            continue  # This term vanishes
        new_U[:, oi] = U[:, r] * scale
        new_V[:, oi] = V[:, r] * scale
        new_W[:, oi] = W[:, r] * scale
    
    # Check if any term vanished
    non_zero = [oi for oi in range(num_other) if np.any(np.abs(new_U[:, oi]) > 1e-12)]
    if len(non_zero) < num_other:
        # Some terms vanished, even better
        final_U = new_U[:, non_zero]
        final_V = new_V[:, non_zero]
        final_W = new_W[:, non_zero]
    else:
        final_U = new_U
        final_V = new_V
        final_W = new_W
    
    # Verify
    if _verify(final_U, final_V, final_W, n, m, p):
        return final_U, final_V, final_W
    
    return None

def _search_for_lower_rank():
    """Search for a lower-rank decomposition."""
    n, m, p = 2, 3, 10
    U, V, W = _build_base_decomposition()
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Base decomposition is invalid!")
    
    rank = len(U[0])
    print(f"Base rank: {rank}, verified: True")
    
    # Try to eliminate each term
    for term_idx in range(rank):
        if time.time() - _start_time > 100:
            break
        result = _try_eliminate_term(U, V, W, term_idx, n, m, p)
        if result is not None:
            U54, V54, W54 = result
            new_rank = len(U54[0])
            print(f"Found rank {new_rank} by eliminating term {term_idx}")
            return U54, V54, W54
    
    # Try a different approach: look for linear dependencies among subsets
    # If a subset of k terms is linearly dependent, we can reduce the rank by 1
    # (if the dependency allows removing one term)
    
    # Try pairs: can two terms be combined?
    # t_r + t_s = rank-1 tensor?
    # This is hard to check directly.
    
    # Try: perturb the decomposition slightly and see if we can find a lower-rank solution
    # using numerical rank detection
    
    # Alternative: try a completely different decomposition structure
    # What if we don't use Strassen for the 2x2x2 blocks?
    # What if we use a different algorithm for the 2x3x10 problem?
    
    # Let me try: group the 10 columns differently
    # Instead of 5 pairs, try 3 groups of (4,3,3) or other partitions
    
    # For a 2x3 x 3x2 subproblem (2 rows of A, 3 cols of A, 3 rows of B, 2 cols of B):
    # This is a 2x3x2 tensor. What's its minimum rank?
    # The schoolbook rank is 3*2*2 = 12.
    # Can we do better?
    
    # Actually, let me try a different structural approach:
    # Instead of treating each pair of B columns independently, 
    # what if I use a shared structure across all 10 columns?
    
    # The key observation: A is 2x3, so A has 6 entries.
    # B is 3x10, so B has 30 entries.
    # C is 2x10, so C has 20 entries.
    # Each C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need rank-1 tensors u⊗v⊗w where:
    # u is in R^6 (indexed by (i,j) with i in [2], j in [3])
    # v is in R^30 (indexed by (j,k) with j in [3], k in [10])
    # w is in R^20 (indexed by (k,i) with k in [10], i in [2])
    # And sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all (i,j,k)
    
    # This is a 60-dimensional tensor. The minimum rank is at least...
    # The "tensor rank" lower bound from the "slice rank" or other methods.
    
    # Let me just try to verify that rank 55 is indeed minimal by checking
    # if any linear dependency exists.
    
    # Build the full 60x55 matrix
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entries.append((i, j, k))
    
    M_full = np.zeros((60, 55))
    for idx, (i, j, k) in enumerate(entries):
        for r in range(55):
            M_full[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check rank of M_full
    rank_M = np.linalg.matrix_rank(M_full, tol=1e-10)
    print(f"Rank of 60x55 matrix: {rank_M}")
    
    # If rank_M < 55, there's a linear dependency
    if rank_M < 55:
        print("Linear dependency found!")
        # Find the null space
        _, S, Vt = np.linalg.svd(M_full, full_matrices=True)
        null_space = Vt[rank_M:]
        print(f"Null space dimension: {len(null_space)}")
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    if time.time() - _start_time > 100:
        U, V, W = _build_base_decomposition()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition
    result = _search_for_lower_rank()
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return the rank-55 solution
    U, V, W = _build_base_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

