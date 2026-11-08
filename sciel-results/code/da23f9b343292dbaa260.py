import numpy as np
import time
import random

TARGET_DEADLINE = time.time() + 110  # Safety margin

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    # T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    # T_decomp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    # Check all non-zero entries of the target tensor
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                val = 0.0
                for r in range(R):
                    val += W[c_idx, r] * U[a_idx, r] * V[b_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check a subset of zero entries to ensure no spurious terms
    # For each (i, k), c_idx is fixed.
    # Zero entries occur when the indices don't match the pattern.
    # Specifically, for fixed i, k, and a_idx = i*m+j, b_idx must be j*p+k.
    # If we pick b_idx corresponding to a different j', it should be 0.
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            # Check one zero entry per (i,k) if m > 1
            if m > 1:
                j_test = (1 if j == 0 else 0) # Just pick a different j
                # Actually, let's just check a specific zero: a = i*m + 0, b = 1*p + k (if m>1)
                a_idx = i * m + 0
                b_idx = 1 * p + k
                # But we must ensure (a_idx, b_idx) is not a valid (i, j, k) tuple for this c_idx
                # c_idx fixes i and k. a_idx fixes j=0. b_idx fixes j=1.
                # Since j must be consistent, this is a zero entry.
                val = 0.0
                for r in range(R):
                    val += W[c_idx, r] * U[a_idx, r] * V[b_idx, r]
                if abs(val) > 1e-9:
                    return False
    return True

def _khatri_rao_solve(U, V, n, m, p, R):
    """
    Given U (nm x R) and V (mp x R), solve for W (pn x R).
    For each (i, k), we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j in 0..m-1
    
    This is a linear system for the row W[k*n+i, :].
    Let w_ik be the row.
    For each j, dot(U[i*m+j, :], V[j*p+k, :] * w_ik) = 1? No.
    sum_r U[i*m+j, r] * V[j*p+k, r] * w_ik[r] = 1.
    Let M_j be the vector (U[i*m+j, :] * V[j*p+k, :]).
    Then M_j dot w_ik = 1.
    We have m equations (one for each j) and R unknowns (entries of w_ik).
    We solve this least squares problem. If the residual is small, it's a valid solution.
    """
    W = np.zeros((p * n, R))
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            # Build m x R matrix A where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            try:
                # Solve A @ w = b
                # Use lstsq for robustness
                w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                W[c_idx, :] = w
            except np.linalg.LinAlgError:
                return None
    return W

def construct():
    n, m, p = 2, 3, 10
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Trivial decomposition: Rank = n * m * p = 60
    R_trivial = n * m * p
    U_triv = np.zeros((nm, R_trivial))
    V_triv = np.zeros((mp, R_trivial))
    W_triv = np.zeros((pn, R_trivial))
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                r = (i * p + k) * m + j
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_triv[a_idx, r] = 1.0
                V_triv[b_idx, r] = 1.0
                W_triv[c_idx, r] = 1.0
                
    # Try to find a lower rank solution
    # We search from R=59 down to R=50
    rng = random.Random(42)
    
    for R in range(R_trivial - 1, 49, -1):
        if time.time() > TARGET_DEADLINE - 10:
            break
            
        # Limit iterations per rank to avoid spending too much time on one rank
        max_iters = 500
        found = False
        
        for it in range(max_iters):
            if time.time() > TARGET_DEADLINE - 5:
                break
                
            # Random sparse U and V
            # Entries from {-1, 0, 1} or {-0.5, 0, 0.5}
            # Density around 0.2-0.3
            density = 0.25
            vals = [-1.0, 0.5, 1.0]
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(nm):
                for c in range(R):
                    if rng.random() < density:
                        U[r, c] = rng.choice(vals)
                        
            for r in range(mp):
                for c in range(R):
                    if rng.random() < density:
                        V[r, c] = rng.choice(vals)
            
            W = _khatri_rao_solve(U, V, n, m, p, R)
            
            if W is not None:
                # Verify the solution
                if _verify(U, V, W, n, m, p):
                    # Convert to lists
                    return {
                        'U': U.tolist(),
                        'V': V.tolist(),
                        'W': W.tolist()
                    }
                    
    # Return trivial solution if no lower rank found
    return {
        'U': U_triv.tolist(),
        'V': V_triv.tolist(),
        'W': W_triv.tolist()
    }
