import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    def verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def schoolbook(n, m, p):
        """Standard schoolbook decomposition with rank n*m*p"""
        rank = n * m * p
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
        return U, V, W

    def find_decomp_nm_p(n, m, p, max_rank=None, deadline=None):
        """
        Find rank-minimal exact tensor decomposition for n x m * m x p.
        Uses exhaustive search over small integer entries.
        """
        if max_rank is None:
            max_rank = n * m * p
        
        # For small problems, try ranks from lower bound up
        # Lower bound: max(n*m, m*p, p*n) / something... actually just start reasonable
        # For 2x3x3: schoolbook=18, try from ~10 up
        # For 2x3x2: schoolbook=12, try from ~7 up
        
        sb_U, sb_V, sb_W = schoolbook(n, m, p)
        best_rank = n * m * p
        best_U, best_V, best_W = sb_U, sb_V, sb_W

        # Try to find lower rank
        # Strategy: enumerate possible rank-r decompositions with small integer entries
        # For very small cases (2x3x2, 2x3x3), this is feasible
        
        # Lower bound on rank: at least max(n, m, p) in many cases, but let's be safe
        min_rank_to_try = max(1, (n * m * p) // 4)
        
        for rank in range(min_rank_to_try, best_rank):
            if deadline is not None and time.time() > deadline:
                break
            
            # For this to be feasible, we need small search space
            # Entries in {-1, 0, 1} or {-2, -1, 0, 1, 2}
            # For rank <= 10 and small matrices, try {-1, 0, 1}
            
            # Build the constraint: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a system of n*m*p equations in (n*m + m*p + p*n)*rank variables
            
            # For very small cases, use a smarter approach:
            # Fix W to be identity-like or simple patterns, then solve for U, V
            
            # Approach: try all possible W matrices with entries in {-1, 0, 1}
            # For small rank and small p*n, this is feasible
            
            if p * n * rank > 50:  # Too many possibilities for W
                continue
                
            # Enumerate W: (p*n) x rank matrix with entries in {-1, 0, 1}
            # Total entries: p*n*rank
            # For 2x3x2: p*n=4, rank up to 11 -> 4*11=44 entries, 3^44 too large
            # Need smarter approach for larger cases
            
            # Better approach for small cases: use the structure
            # For 2x3x2: n=2, m=3, p=2
            # We need: for all i in {0,1}, j in {0,1,2}, k in {0,1}:
            #   sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
            
            # Let's use a randomized approach with integer rounding for small cases
            # Or use the known result that small matrix multiplication ranks are known
            
            # For 2x3x2: known rank is at most 12 (schoolbook), likely 12 or less
            # For 2x3x3: known rank is at most 18 (schoolbook)
            
            # Given time constraints, let's use a targeted search:
            # Try rank values and use a simple optimization
            
            # For very small cases, brute force with {-1,0,1} entries
            if p * n * rank <= 20:
                # Feasible to enumerate W
                pn = p * n
                total_w_entries = pn * rank
                if total_w_entries <= 18:  # 3^18 = 387M, too large
                    # Still too large for full enumeration
                    pass
            
            # Use a different strategy: for the specific small cases, 
            # use known constructions or a targeted search
            
            # For now, rely on the tensor product construction for the final answer
            # and just verify the schoolbook is correct
            pass
        
        return best_rank, best_U, best_V, best_W

    # Main approach: use tensor product structure since p=6=2*3
    # R(2,3,6) = R(2,3,2) * R(2,3,3)
    # 
    # For the base cases, use schoolbook (which may be optimal for these small cases):
    # R(2,3,2) <= 12 (schoolbook)
    # R(2,3,3) <= 18 (schoolbook)
    # So R(2,3,6) <= 12 * 18 = 216 ... that's worse than schoolbook 36!
    
    # Wait, that's not right. The tensor product gives:
    # R(n, m1*m2, p) <= R(n, m1, p) * R(n, m2, p)
    # Or R(n1*n2, m, p) <= R(n1, m, p) * R(n2, m, p)
    # Or R(n, m, p1*p2) <= R(n, m, p1) * R(n, m, p2)
    
    # So R(2, 3, 6) = R(2, 3, 2*3) <= R(2, 3, 2) * R(2, 3, 3)
    # If R(2,3,2) = 12 and R(2,3,3) = 18, then R(2,3,6) <= 216. Worse than 36.
    
    # But we can also decompose differently:
    # R(2, 3, 6) <= R(2, 3, 6) directly via schoolbook = 36
    
    # The tensor product only helps if we can find that R(2,3,2) < 12 or R(2,3,3) < 18
    # OR if we use a different factorization.
    
    # Alternative: R(2,3,6) = R(2, 3*2, 3) ... no, m must stay the same.
    
    # Actually, let me reconsider. The standard result is:
    # R(n1*n2, m, p) <= R(n1, m, p) * R(n2, m, p)  [block diagonal in n]
    # R(n, m1*m2, p) <= R(n, m1, p) * R(n, m2, p)  [block diagonal in m] 
    # R(n, m, p1*p2) <= R(n, m, p1) * R(n, m, p2)  [block diagonal in p]
    
    # None of these directly help reduce below 36 unless the base ranks are significantly lower.
    
    # For 2x3x2: is rank < 12 possible?
    # For 2x3x3: is rank < 18 possible?
    
    # Let me try to find lower ranks for the base cases using a targeted search.
    
    # For 2x3x2 (n=2, m=3, p=2):
    # We need U: 6 x r, V: 6 x r, W: 4 x r
    # Equations: for i in {0,1}, j in {0,1,2}, k in {0,1}:
    #   sum_{r'} U[i*3+j, r'] * V[j*2+k, r'] * W[k*2+i, r'] = 1
    # Total: 12 equations, 16r variables
    
    # For 2x3x3 (n=2, m=3, p=3):
    # U: 6 x r, V: 9 x r, W: 6 x r
    # Equations: 18 equations, 21r variables
    
    # Let me try a randomized search with integer constraints for these small cases.
    
    def try_small_decomp(n, m, p, max_rank, time_limit=10):
        """Try to find a low-rank decomposition for small matrix multiplication"""
        start = time.time()
        
        # Schoolbook as baseline
        sb_U, sb_V, sb_W = schoolbook(n, m, p)
        best_rank = n * m * p
        best_U, best_V, best_W = sb_U, sb_V, sb_W
        
        # Try ranks from a lower estimate up
        # Lower bound: at least max(n*m, m*p, p*n) / min(n, m, p) roughly
        # Just try from max(6, n*m*p//3) up
        
        for rank in range(max(4, n * m * p // 4), min(max_rank + 1, n * m * p)):
            if time.time() - start > time_limit:
                break
            
            # Try to find a rank-r decomposition
            # Use a simple approach: random initialization + ALS with integer rounding
            
            np.random.seed(42 + rank)
            
            # Multiple random restarts
            found = False
            for restart in range(5):
                if time.time() - start > time_limit:
                    break
                
                U = np.random.randint(-1, 2, size=(n * m, rank)).astype(np.float64)
                V = np.random.randint(-1, 2, size=(m * p, rank)).astype(np.float64)
                W = np.random.randint(-1, 2, size=(p * n, rank)).astype(np.float64)
                
                # ALS iterations
                for it in range(200):
                    if time.time() - start > time_limit:
                        break
                    
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((p, rank))
                            for k in range(p):
                                for rr in range(rank):
                                    A_mat[k, rr] = V[j * p + k, rr] * W[k * n + i, rr]
                            b_vec = np.ones(p)
                            try:
                                sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                                U[i * m + j, :] = np.round(sol)
                            except:
                                pass
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((n, rank))
                            for i in range(n):
                                for rr in range(rank):
                                    A_mat[i, rr] = U[i * m + j, rr] * W[k * n + i, rr]
                            b_vec = np.ones(n)
                            try:
                                sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                                V[j * p + k, :] = np.round(sol)
                            except:
                                pass
                    
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((m, rank))
                            for j in range(m):
                                for rr in range(rank):
                                    A_mat[j, rr] = U[i * m + j, rr] * V[j * p + k, rr]
                            b_vec = np.ones(m)
                            try:
                                sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                                W[k * n + i, :] = np.round(sol)
                            except:
                                pass
                    
                    # Check if we have a valid solution
                    if verify(U, V, W, 1e-9):
                        # Convert to integer
                        U_int = U.astype(np.int64)
                        V_int = V.astype(np.int64)
                        W_int = W.astype(np.int64)
                        if verify(U_int, V_int, W_int, 1e-9):
                            best_rank = rank
                            best_U, best_V, best_W = U_int, V_int, W_int
                            found = True
                            break
                
                if found:
                    break
            
            if found:
                break
        
        return best_rank, best_U, best_V, best_W

    # Try to find better decompositions for the base cases
    # But given time constraints, let's focus on the main problem
    
    # For 2x3x6, the schoolbook rank is 36.
    # Let's try to directly find a lower-rank decomposition.
    
    # Key insight: maybe we can use the fact that 6 = 2*3 and structure the decomposition
    # as a Kronecker product of two smaller decompositions.
    
    # If we have decompositions for (2,3,2) with rank r1 and (2,3,3) with rank r2,
    # then (2,3,6) has rank at most r1*r2.
    # If r1=6 and r2=6, we'd get rank 36. Same as schoolbook.
    # If r1=5 and r2=7, we'd get rank 35. Better!
    
    # So the question is: what are the true minimal ranks for (2,3,2) and (2,3,3)?
    
    # Let me try to find them with a focused search.
    
    # For (2,3,2): n=2, m=3, p=2
    # Schoolbook rank = 12
    # Let's try to find rank 10 or 11.
    
    if time.time() < deadline - 60:
        r_232, U_232, V_232, W_232 = try_small_decomp(2, 3, 2, max_rank=12, time_limit=30)
    else:
        r_232, U_232, V_232, W_232 = 12, *schoolbook(2, 3, 2)
    
    if time.time() < deadline - 30:
        r_233, U_233, V_233, W_233 = try_small_decomp(2, 3, 3, max_rank=18, time_limit=30)
    else:
        r_233, U_233, V_233, W_233 = 18, *schoolbook(2, 3, 3)
    
    # Now construct (2,3,6) from (2,3,2) and (2,3,3)
    # The tensor product construction:
    # For (n, m, p1*p2) from (n, m, p1) and (n, m, p2):
    # U_combined[i*m+j, (r1, r2)] = U1[i*m+j, r1] * U2[i*m+j, r2]
    # V_combined[j*p1*p2 + k, (r1, r2)] = V1[j*p1 + k1, r1] * V2[j*p2 + k2, r2] where k = k1*p2 + k2
    # W_combined[(k1*p2+k2)*n + i, (r1, r2)] = W1[k1*n + i, r1] * W2[k2*n + i, r2]
    
    r1, r2 = r_232, r_233
    p1, p2 = 2, 3
    p_combined = p1 * p2  # = 6
    
    rank_combined = r1 * r2
    
    U_comb = np.zeros((n * m, rank_combined), dtype=np.int64)
    V_comb = np.zeros((m * p_combined, rank_combined), dtype=np.int64)
    W_comb = np.zeros((p_combined * n, rank_combined), dtype=np.int64)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            col = r1_idx * r2 + r2_idx
            
            # U: same for both, multiply
            U_comb[:, col] = U_232[:, r1_idx] * U_233[:, r2_idx]
            
            # V: V[j*p1*p2 + k, col] where k = k1*p2 + k2
            # V1[j*p1 + k1, r1_idx] * V2[j*p2 + k2, r2_idx]
            for j in range(m):
                for k1 in range(p1):
                    for k2 in range(p2):
                        k = k1 * p2 + k2
                        V_comb[j * p_combined + k, col] = V_232[j * p1 + k1, r1_idx] * V_233[j * p2 + k2, r2_idx]
            
            # W: W[(k1*p2+k2)*n + i, col] = W1[k1*n + i, r1_idx] * W2[k2*n + i, r2_idx]
            for k1 in range(p1):
                for k2 in range(p2):
                    for i in range(n):
                        k = k1 * p2 + k2
                        W_comb[k * n + i, col] = W_232[k1 * n + i, r1_idx] * W_233[k2 * n + i, r2_idx]
    
    # Verify the combined decomposition
    if verify(U_comb, V_comb, W_comb, 1e-9):
        best_rank = rank_combined
        best_U, best_V, best_W = U_comb, V_comb, W_comb
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Also try direct schoolbook for comparison
    sb_rank = n * m * p
    if sb_rank < best_rank:
        best_rank = sb_rank
        best_U, best_V, best_W = schoolbook(n, m, p)
    
    # Final verification
    if not verify(best_U, best_V, best_W, 1e-9):
        # Emergency fallback
        best_U, best_V, best_W = schoolbook(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
