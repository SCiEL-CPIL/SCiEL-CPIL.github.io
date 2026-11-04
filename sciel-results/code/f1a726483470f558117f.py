"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Schoolbook solution (rank 120)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb
    
    def verify_fast(U, V, W, rank):
        """Fast verification using the tensor structure."""
        # Check that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for non-matching: sum = 0
        
        # Build the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # But we only need to check specific entries.
        
        # For efficiency, check all (a,b,c) where a=i*m+j, b=j2*p+k, c=k2*n+i2
        # The tensor should be 1 iff j=j2 and i=i2 and k=k2, else 0.
        
        # For each (i,j,k), compute the sum and check it equals 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        
        # Check zero entries: for (i,j,k) and (i2,j2,k2) with (i,j,k) != (i2,j2,k2)
        # The entry at (a=i*m+j, b=j2*p+k, c=k2*n+i) should be 0 if j!=j2 or i!=i2 or k!=k2
        # But note: a=i*m+j, b=j2*p+k, c=k2*n+i
        # This is non-zero only if j=j2 AND i=i2 AND k=k2 (from the tensor definition)
        # So we need to check that for j!=j2 or i!=i2 or k!=k2, the sum is 0.
        
        # To be thorough but fast, check a sample of non-matching entries
        # Actually, let's check all of them since n,m,p are small
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                b = j2 * p + k
                                c = k2 * n + i
                                if j != j2 or i != i2 or k != k2:
                                    s = 0.0
                                    for r in range(rank):
                                        s += U[a, r] * V[b, r] * W[c, r]
                                    if abs(s) > 1e-6:
                                        return False
        return True
    
    # Verify schoolbook
    if not verify_fast(best_U, best_V, best_W, best_rank):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    def try_rank_reduction(target_rank, max_attempts=20):
        """Try to find a valid decomposition of the given rank using structured search."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Use structured initialization: V and W based on Kronecker-like patterns
            # Strategy: for each j, use a rank-r decomposition of the n x p matrix
            # where the (i,k) entry is 1 if we're in the "right" position
            
            # Better: use random but with scaling
            V_try = np.random.randn(mp, target_rank) * (1.0 / np.sqrt(target_rank))
            W_try = np.random.randn(pn, target_rank) * (1.0 / np.sqrt(target_rank))
            
            # Solve for U: for each (i,j), we have p equations:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0..p-1
            # This is: M_ij @ U_ij = ones(p) where M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            U_try = np.zeros((nm, target_rank))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, target_rank))
                    for k in range(p):
                        M_ij[k, :] = V_try[j * p + k, :] * W_try[k * n + i, :]
                    
                    rhs = np.ones(p)
                    try:
                        # Use least squares
                        U_ij, residuals, rank_m, sv = np.linalg.lstsq(M_ij, rhs, rcond=None)
                        U_try[i * m + j, :] = U_ij
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Quick check before full verification
            # Check the "1" constraints first
            quick_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U_try[i * m + j, r] * V_try[j * p + k, r] * W_try[k * n + i, r]
                        if abs(s - 1.0) > 1e-4:
                            quick_ok = False
                            break
                    if not quick_ok:
                        break
                if not quick_ok:
                    break
            
            if not quick_ok:
                continue
            
            # Full verification
            if verify_fast(U_try, V_try, W_try, target_rank):
                return U_try, V_try, W_try
        
        return None
    
    # Try ranks from best_rank-1 downward
    # Use more attempts for lower ranks
    for target_rank in range(best_rank - 1, max(80, best_rank - 40), -1):
        if time.time() > deadline - 10:
            break
        
        # More attempts for ranks closer to schoolbook
        attempts = 10 if target_rank > best_rank - 10 else 20
        result = try_rank_reduction(target_rank, max_attempts=attempts)
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, target_rank
            break  # Found a lower rank, stop
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
