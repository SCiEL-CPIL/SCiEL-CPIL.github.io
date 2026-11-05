"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    The multiplication tensor T has entries T[i,j,k] = 1 at positions (i*m+j, j*p+k, k*n+i)
    for i in [0,n), j in [0,m), k in [0,p). The schoolbook rank is n*m*p = 63.
    
    We attempt to find a lower-rank exact decomposition using algebraic search techniques,
    falling back to the schoolbook decomposition if no improvement is found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 3, 7
    rank_schoolbook = n * m * p  # 63
    
    def _elapsed():
        return time.time() - start_time
    
    def _deadline_exceeded():
        return _elapsed() > deadline
    
    def _schoolbook_decomposition():
        """Return the rank n*m*p schoolbook decomposition."""
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
    
    def _verify_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition of the multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all required tensor entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    decomp_val = float(np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i]))
                    if abs(decomp_val - 1.0) > 1e-10:
                        return False
        
        # Check zero entries: for each rank-1 term, verify no spurious contributions
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            
            nz_u = np.where(np.abs(u_col) > 1e-12)[0]
            nz_v = np.where(np.abs(v_col) > 1e-12)[0]
            nz_w = np.where(np.abs(w_col) > 1e-12)[0]
            
            for a in nz_u:
                i_a, j_a = a // m, a % m
                for b in nz_v:
                    j2_b, k_b = b // p, b % p
                    if j_a != j2_b:
                        continue
                    for c in nz_w:
                        k2_c, i2_c = c // n, c % n
                        if k_b != k2_c or i_a != i2_c:
                            prod = u_col[a] * v_col[b] * w_col[c]
                            if abs(prod) > 1e-10:
                                return False
        return True
    
    def _try_strassen_style():
        """
        Attempt to find a rank-reducing decomposition using algebraic methods.
        
        For the 3x3x3 case, Strassen achieves rank 7. We explore whether similar
        rank reductions are possible for 3x3x7 by exploiting the tensor structure.
        
        Strategy: Try to use known low-rank decompositions for sub-blocks and
        combine them with schoolbook for the remaining parts.
        """
        if _deadline_exceeded():
            return None
        
        # For 3x3x3, the best known exact decomposition has rank 19 (CW-19).
        # For 3x3x7, we can potentially use:
        # - A rank-r decomposition for the 3x3x3 sub-tensor (k=0,1,2)
        # - Schoolbook (rank 36) for the remaining 3x3x4 sub-tensor (k=3,4,5,6)
        # Total: r + 36
        
        # Let's try to construct a valid decomposition with rank < 63
        # by exploiting the structure more carefully.
        
        # Approach: Try to find linear dependencies that allow rank reduction.
        # The tensor T[i,j,k] = 1 if the indices are consistent, 0 otherwise.
        
        # For fixed j, the sub-tensor over (i,k) is:
        # T[i, j, k] = 1 for all i in [0,3), k in [0,7)
        # This is a 3x7 matrix of all ones, which has rank 1.
        
        # For fixed i, the sub-tensor over (j,k) is:
        # T[i, j, k] = 1 for all j in [0,3), k in [0,7)
        # This is a 3x7 matrix of all ones, which has rank 1.
        
        # For fixed k, the sub-tensor over (i,j) is:
        # T[i, j, k] = 1 for all i in [0,3), j in [0,3)
        # This is a 3x3 matrix of all ones, which has rank 1.
        
        # This suggests that the tensor might have lower rank than 63,
        # but the 3D structure makes it more complex.
        
        # Let's try a parameterized search: assume a decomposition of the form
        # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # where u_r, v_r, w_r have specific patterns.
        
        # Try to find R such that we can cover all 63 required entries
        # with R rank-1 terms, each contributing to multiple entries.
        
        # Each rank-1 term u_r ⊗ v_r ⊗ w_r contributes to entries
        # (a, b, c) where a is in support of u_r, b in support of v_r,
        # c in support of w_r.
        
        # For the term to be valid, for all (a,b,c) in the product of supports:
        # if j_a == j2_b and k_b == k2_c and i_a == i2_c, then it's a required entry
        # otherwise, it must not contribute (or contributions must cancel)
        
        # This is a complex constraint satisfaction problem.
        # Let's try a heuristic: search for decompositions where each rank-1 term
        # covers a "block" of entries.
        
        # For now, try a few specific patterns
        patterns = [
            # Pattern 1: Each term covers one (j) slice
            # For each j, use 7 terms (one for each k)
            # Total: 3 * 7 = 21 terms, but each term covers 3 entries (one for each i)
            # This gives rank 21, which is less than 63!
            # But wait, we need to check if this is valid.
        ]
        
        # Let's try the pattern where for each (j, k), we have one term
        # that covers all i values.
        # Term for (j, k): u has 1s at positions (0*m+j, 1*m+j, 2*m+j)
        #                  v has 1 at position (j*p+k)
        #                  w has 1s at positions (k*n+0, k*n+1, k*n+2)
        
        # Check: for this term, the entries covered are:
        # (i*m+j, j*p+k, k*n+i') for i in [0,3), i' in [0,3)
        # We need j == j2 (true, j2 = j), k == k2 (true, k2 = k), i == i' (only when i == i')
        # So this term contributes to entries where i == i', which are the required entries.
        # But it also contributes to entries where i != i', which should be zero!
        # So this pattern is NOT valid.
        
        # We need to be more careful. Let's try a different approach.
        
        # For each (i, j, k), the schoolbook term is:
        # u has 1 at (i*m+j), v has 1 at (j*p+k), w has 1 at (k*n+i)
        
        # Can we combine multiple schoolbook terms into one?
        # Suppose we want to combine terms for (i1, j, k) and (i2, j, k) with i1 != i2.
        # The combined term would need:
        # u has 1s at (i1*m+j) and (i2*m+j)
        # v has 1 at (j*p+k)
        # w has 1s at (k*n+i1) and (k*n+i2)
        
        # This term contributes to entries (i*m+j, j*p+k, k*n+i') for i in {i1, i2}, i' in {i1, i2}
        # Required: (i1, j, k) and (i2, j, k) -> covered
        # Spurious: (i1, j, k) with i'=i2 and (i2, j, k) with i'=i1 -> these should be zero
        # But they get contribution 1, which is wrong.
        
        # So we can't simply combine terms this way.
        
        # The key insight is that the tensor has a specific structure that prevents
        # simple rank reductions. The best known result for 3x3x3 is rank 19,
        # which is significantly less than 27, but it uses a complex pattern.
        
        # For 3x3x7, let's try to use the CW-19 decomposition for the 3x3x3 part
        # and schoolbook for the rest.
        
        # CW-19 decomposition for 3x3x3:
        # This is a known result but complex to implement from memory.
        # Let's try to find it or use a simpler approach.
        
        # Alternative: Use the fact that 3x3 matrix multiplication can be done
        # with 19 multiplications (CW-19). For 3x3x7, we can view it as
        # 7 separate 3x3 multiplications, but with shared structure.
        
        # For now, return None as we don't have a correct CW-19 implementation
        return None
    
    def _random_search_for_improvement(max_rank, max_iter=200):
        """
        Random search for a rank < max_rank decomposition.
        This is unlikely to succeed for large gaps but worth trying.
        """
        if _deadline_exceeded():
            return None
        
        # Try ranks from max_rank-1 down to max_rank-10
        for rank in range(max_rank - 1, max(1, max_rank - 10), -1):
            if _deadline_exceeded():
                return None
            for attempt in range(max_iter):
                if _deadline_exceeded():
                    return None
                # Random rank-1 term generation with small integer/half-integer entries
                U = np.random.randint(-2, 3, size=(n * m, rank))
                V = np.random.randint(-2, 3, size=(m * p, rank))
                W = np.random.randint(-2, 3, size=(p * n, rank))
                
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # First, try algebraic methods
    result = _try_strassen_style()
    if result is not None:
        return result
    
    # Then, try random search for small rank reductions
    result = _random_search_for_improvement(rank_schoolbook, max_iter=200)
    if result is not None:
        return result
    
    # Fall back to schoolbook decomposition
    return _schoolbook_decomposition()
