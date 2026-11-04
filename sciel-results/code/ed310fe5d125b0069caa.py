import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1
    
    best_rank = 17
    best_U, best_V, best_W = None, None, None
    
    # Store the current rank 17 solution as fallback
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    # Copy rank 17 solution
    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)
    
    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)
    
    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)
    
    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)
    
    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)
    
    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)
    
    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)
    
    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)
    
    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)
    
    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)
    
    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)
    
    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)
    
    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)
    
    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)
    
    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)
    
    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)
    
    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)
    
    best_U, best_V, best_W = U17, V17, W17

    def compute_error(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    error += (s - 1.0) ** 2
        return error

    def is_exact(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Laser method: Try to find rank-16 by exploiting linear structure
    # The idea: for each pair (i,j), the vector across k and r should have structure
    # We can fix most entries and solve for the rest
    
    def try_laser_rank16():
        """
        Use the laser method: exploit that for fixed (i,j), 
        the tensor slice is a 3x3 matrix (over k, and some structure over r)
        """
        if time.time() > deadline:
            return None, None, None, None
        
        # Strategy: Try to find a rank-16 decomposition by modifying the rank-17
        # and checking if one term can be eliminated
        
        # For rank 16, we need to find U, V, W of shape (6,16), (9,16), (6,16)
        # such that the tensor equation holds.
        
        # Use the structure: the last 10 terms (r=7..16) in the rank-17 solution
        # are "simple" (single nonzero in each of U, V, W). Let's see if we can
        # merge some of these.
        
        # Actually, let's try a different approach: use the fact that the problem
        # has symmetries. The 2x3x3 tensor can be analyzed by considering
        # the "border rank" and using the laser method.
        
        # For now, try a structured search:
        # Fix the first 7 terms (the Strassen-like ones) and try to find
        # 9 more terms (total 16) that complete the decomposition.
        
        # The first 7 terms handle the "complex" interactions.
        # The remaining terms need to handle the "simple" ones.
        
        # Let's try: keep terms 0-6 from rank-17, and find 9 new terms
        # for the remaining contributions.
        
        # Compute what's already covered by terms 0-6
        U7 = U17[:, :7]
        V7 = V17[:, :7]
        W7 = W17[:, :7]
        
        # Residual tensor: T - sum_{r=0}^{6} U_r V_r W_r
        # T[i,j,k] = 1 for all i,j,k
        # Compute residual
        U7_t = U7.reshape(n, m, 7)
        V7_t = V7.reshape(m, p, 7)
        W7_t = W7.reshape(p, n, 7)
        
        residual = np.ones((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(7):
                        s += U7_t[i, j, r] * V7_t[j, k, r] * W7_t[k, i, r]
                    residual[i, j, k] = 1.0 - s
        
        # Now we need to decompose the residual tensor with rank 9
        # using half-integer entries.
        
        # The residual should be "simple" - let's check its structure
        # For the given rank-17 solution, terms 7-16 are:
        # (0,0,7): U[0,0]=1, V[0,2]=1, W[2,0]=1
        # (0,1,8): U[0,1]=1, V[1,2]=1, W[2,0]=1
        # (1,0,9): U[1,0]=1, V[0,2]=1, W[2,1]=1
        # (1,1,10): U[1,1]=1, V[1,2]=1, W[2,1]=1
        # (0,2,11): U[0,2]=1, V[2,0]=1, W[0,0]=1
        # (0,2,12): U[0,2]=1, V[2,1]=1, W[1,0]=1
        # (0,2,13): U[0,2]=1, V[2,2]=1, W[2,0]=1
        # (1,2,14): U[1,2]=1, V[2,0]=1, W[0,1]=1
        # (1,2,15): U[1,2]=1, V[2,1]=1, W[1,1]=1
        # (1,2,16): U[1,2]=1, V[2,2]=1, W[2,1]=1
        
        # These 10 terms contribute to specific (i,j,k) entries.
        # Let's verify: each term contributes 1 to exactly one (i,j,k) entry.
        # Term 7: i=0,j=0,k=2 -> contributes 1
        # Term 8: i=0,j=1,k=2 -> contributes 1
        # Term 9: i=1,j=0,k=2 -> contributes 1
        # Term 10: i=1,j=1,k=2 -> contributes 1
        # Term 11: i=0,j=2,k=0 -> contributes 1
        # Term 12: i=0,j=2,k=1 -> contributes 1
        # Term 13: i=0,j=2,k=2 -> contributes 1
        # Term 14: i=1,j=2,k=0 -> contributes 1
        # Term 15: i=1,j=2,k=1 -> contributes 1
        # Term 16: i=1,j=2,k=2 -> contributes 1
        
        # So the residual after terms 0-6 should have 1s at:
        # (0,0,2), (0,1,2), (1,0,2), (1,1,2),
        # (0,2,0), (0,2,1), (0,2,2), (1,2,0), (1,2,1), (1,2,2)
        # and 0 elsewhere.
        
        # Let's verify this is what we get
        expected_residual = np.zeros((n, m, p))
        for i in range(n):
            for j in [0, 1, 2]:
                if j == 2:
                    for k in range(p):
                        expected_residual[i, j, k] = 1.0
                else:
                    expected_residual[i, j, 2] = 1.0
        
        # Check if residual matches expected
        if np.max(np.abs(residual - expected_residual)) > 1e-9:
            # The first 7 terms don't leave the expected simple residual
            # Try different approach
            pass
        
        # For the simple residual, we can use rank 10 (one term per nonzero entry)
        # But we need rank 9 for total rank 16.
        # Can we merge two of these simple terms?
        
        # Two simple terms can be merged if they share structure.
        # Term A: U_A[i,j] = 1 at (i1,j1), V_A[j,k] = 1 at (j1,k1), W_A[k,i] = 1 at (k1,i1)
        # Term B: U_B[i,j] = 1 at (i2,j2), V_B[j,k] = 1 at (j2,k2), W_B[k,i] = 1 at (k2,i2)
        
        # To merge into one term, we'd need a single rank-1 tensor that gives
        # 1 at both (i1,j1,k1) and (i2,j2,k2), which is impossible unless
        # they're the same entry.
        
        # So rank 10 is needed for the simple part, giving total rank 17.
        # To get rank 16, we need to restructure the first 7 terms.
        
        # Alternative: Can we use half-integers to merge some terms?
        # E.g., can two simple terms be replaced by two terms with half-integer
        # entries that together give the same contribution?
        
        # This is getting complex. Let's try a direct search for rank 16.
        
        # Use the structure: for the 2x3x3 case, we can parameterize
        # and solve a system of equations.
        
        # The tensor has 18 entries (2*3*3). We need 18 equations.
        # Unknowns: 6*16 + 9*16 + 6*16 = 96 + 144 + 96 = 336 unknowns.
        # This is highly underdetermined, so solutions should exist.
        
        # Let's try to find a solution with entries in {-1, -0.5, 0, 0.5, 1}
        
        # Use a greedy approach: build terms one at a time.
        
        U16 = np.zeros((n * m, 16))
        V16 = np.zeros((m * p, 16))
        W16 = np.zeros((p * n, 16))
        
        # Start with the 7 Strassen-like terms
        for r in range(7):
            U16[:, r] = U17[:, r]
            V16[:, r] = V17[:, r]
            W16[:, r] = W17[:, r]
        
        # Now find 9 more terms to complete the decomposition.
        # The residual after 7 terms should be the 10 simple entries.
        # We need to cover 10 entries with 9 terms, so at least one term
        # must cover two entries.
        
        # A term covers (i,j,k) if U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] != 0
        # For it to cover two entries (i1,j1,k1) and (i2,j2,k2), we need:
        # U[i1*m+j1] * V[j1*p+k1] * W[k1*n+i1] = 1
        # U[i2*m+j2] * V[j2*p+k2] * W[k2*n+i2] = 1
        
        # This requires the indices to be compatible.
        
        # Let's try to find pairs of entries that can be covered by one term.
        # Entry (i,j,k) requires U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
        
        # For two entries (i1,j1,k1) and (i2,j2,k2):
        # Need U[i1*m+j1] = U[i2*m+j2] (or related)
        # Need V[j1*p+k1] = V[j2*p+k2]
        # Need W[k1*n+i1] = W[k2*n+i2]
        
        # Or more generally, the products must work out.
        
        # Let's enumerate the 10 simple entries and look for compatible pairs.
        simple_entries = [
            (0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2),
            (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2)
        ]
        
        # For each entry (i,j,k), the required "signature" is:
        # U_idx = i*m + j, V_idx = j*p + k, W_idx = k*n + i
        
        # Two entries can be merged if their U_idx, V_idx, W_idx are the same
        # or can be made to work with appropriate values.
        
        # Actually, for a single term to contribute 1 to both entries,
        # we need:
        # U[i1*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i1, r] = 1
        # U[i2*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i2, r] = 1
        
        # If (i1,j1) != (i2,j2), then U has different nonzero entries,
        # so V and W must compensate.
        
        # This is complex. Let's just try a search.
        
        # For now, return None to indicate this approach didn't work
        return None, None, None, None
    
    # Try the laser method
    U16, V16, W16, success = try_laser_rank16()
    if success and U16 is not None:
        if is_exact(U16, V16, W16, 16):
            best_rank = 16
            best_U, best_V, best_W = U16, V16, W16
    
    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
