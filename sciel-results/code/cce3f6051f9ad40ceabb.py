import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 6]


def build_target_tensor(n, m, p):
    """T[a*m+b, c*p+d, k*n+i] = 1 iff a==i and b==c and d==k"""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                T[a * m + b, c * p + d, k * n + i] = 1.0
    return T


def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_l U[i,l]*V[j,l]*W[k,l] equals target tensor."""
    rank = U.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            target = 1.0 if (a == i and b == c and d == k) else 0.0
                            s = 0.0
                            idx_u = a * m + b
                            idx_v = c * p + d
                            idx_w = k * n + i
                            for l in range(rank):
                                s += U[idx_u, l] * V[idx_v, l] * W[idx_w, l]
                            if abs(s - target) > tol:
                                return False
    return True


def verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    # For each l, the contribution to the tensor is outer(U[:,l], outer(V[:,l], W[:,l]))
    # We can check by computing the reconstructed tensor
    # T_recon[i,j,k] = sum_l U[i,l]*V[j,l]*W[k,l]
    # This is the tensor product of U, V, W
    # Efficiently: for each l, add U[:,l] outer V[:,l] outer W[:,l]
    # But that's expensive. Instead, check entry by entry but vectorize over l.
    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for l in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, l], V[:, l], W[:, l])
    
    # Build target
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                T_target[a * m + b, c * p + d, k * n + i] = 1.0
    
    return np.allclose(T_recon, T_target, atol=tol)


def try_newton_decomposition(n, m, p, rank, num_trials=100, seed=42, deadline=None):
    """
    Try to find a rank-rank decomposition using Newton's method on coefficient variables.
    
    For each term l, we fix a support pattern (which entries of U[:,l], V[:,l], W[:,l] are nonzero).
    Then we have variables u_{l,i}, v_{l,j}, w_{l,k} for i in support_u(l), etc.
    The equations are: for each (i,j,k) in the tensor support,
        sum_l u_{l,i} * v_{l,j} * w_{l,k} = 1
    And for non-support entries, the sum should be 0.
    
    This is a system of quadratic equations in the variables.
    We use Newton's method to solve it.
    """
    rng = np.random.RandomState(seed)
    
    nU = n * m
    nV = m * p
    nW = p * n
    
    # The tensor has exactly n*m*p nonzero entries (each = 1)
    # All other entries are 0.
    # Total entries: nU * nV * nW
    # Nonzero entries: n * m * p (one for each (a,b,c,d,k,i) with a=i, b=c, d=k)
    # That's n * m * p = 2 * 3 * 6 = 36 nonzero entries.
    
    # For rank r, we have r * (|sup_u| + |sup_v| + |sup_w|) variables.
    # We need at least 36 equations (the nonzero entries) plus the zero entries.
    # But actually all nU*nV*nW entries give equations (most are = 0).
    
    # Strategy: For each term, use a structured support.
    # Term l corresponds to a "path" through the matrix multiplication.
    # A natural support: U has entries for row a, V for column b (of A / row of B), W for output.
    
    # Let's try: for each term l, pick:
    #   1-2 rows for U, 1-2 rows for V, 1-2 rows for W
    # This gives 3-12 variables per term.
    # For rank r, total variables = r * (3 to 12)
    # We need to satisfy 36 nonzero equations + many zero equations.
    
    # Actually, let's be smarter. The tensor is sparse: only 36 nonzero entries.
    # Each nonzero entry T[i,j,k]=1 involves indices (i,j,k) where:
    #   i = a*m+b, j = c*p+d, k = k'*n+i' with a=i', b=c, d=k'
    # So i determines (a,b), j determines (c,d), k determines (k',i')
    # Constraint: a=i', b=c, d=k'
    
    # Let's enumerate the 36 nonzero entries:
    nonzero_entries = []
    for a in range(n):
        for b in range(m):
            for c in range(m):
                if b != c:
                    continue
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and d == k:
                                nonzero_entries.append((a * m + b, c * p + d, k * n + i))
    
    # Should have n * m * p = 36 entries
    assert len(nonzero_entries) == n * m * p, f"Expected {n*m*p}, got {len(nonzero_entries)}"
    
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            return None
        
        # Choose support patterns for each term
        # We'll use a structured approach: each term "covers" some of the 36 entries
        # Term l: pick a subset of U-rows, V-rows, W-rows
        
        supports = []
        for l in range(rank):
            # Pick 1-3 rows for U
            su = rng.choice(nU, size=rng.randint(1, 3), replace=False)
            sv = rng.choice(nV, size=rng.randint(1, 3), replace=False)
            sw = rng.choice(nW, size=rng.randint(1, 3), replace=False)
            supports.append((su, sv, sw))
        
        # Variables: for each term l and each index in its support, a coefficient
        # Total variables: sum_l (|su_l| + |sv_l| + |sw_l|)
        var_count = sum(len(su) + len(sv) + len(sw) for su, sv, sw in supports)
        
        # Equations: for each of the 36 nonzero entries, sum_l prod = 1
        # Also, for entries not in the tensor support, sum_l prod = 0
        # But we only enforce the 36 nonzero ones first; if they're satisfied and the
        # structure is good, zeros might be automatically satisfied.
        # Actually no - we need ALL entries to match. Let's include a subset of zero equations.
        
        # For now, just solve for the 36 nonzero entries.
        # If var_count >= 36, we might have a solution.
        
        if var_count < n * m * p:
            continue
        
        # Initialize variables randomly
        x = rng.uniform(-2, 2, size=var_count)
        
        # Map from (l, type, local_idx) to variable index
        var_map = {}
        var_idx = 0
        for l in range(rank):
            su, sv, sw = supports[l]
            for idx in su:
                var_map[(l, 'U', idx)] = var_idx
                var_idx += 1
            for idx in sv:
                var_map[(l, 'V', idx)] = var_idx
                var_idx += 1
            for idx in sw:
                var_map[(l, 'W', idx)] = var_idx
                var_idx += 1
        
        # Define the residual function and Jacobian
        def residual(x_vec):
            """Compute residuals for the 36 nonzero equations."""
            res = np.zeros(n * m * p)
            for eq_idx, (i, j, k) in enumerate(nonzero_entries):
                s = 0.0
                for l in range(rank):
                    su, sv, sw = supports[l]
                    if i not in su or j not in sv or k not in sw:
                        continue
                    ui = x_vec[var_map[(l, 'U', i)]]
                    vj = x_vec[var_map[(l, 'V', j)]]
                    wk = x_vec[var_map[(l, 'W', k)]]
                    s += ui * vj * wk
                res[eq_idx] = s - 1.0
            return res
        
        def jacobian(x_vec):
            """Compute the Jacobian of the residual w.r.t. variables."""
            J = np.zeros((n * m * p, var_count))
            for eq_idx, (i, j, k) in enumerate(nonzero_entries):
                for l in range(rank):
                    su, sv, sw = supports[l]
                    if i not in su or j not in sv or k not in sw:
                        continue
                    ui = x_vec[var_map[(l, 'U', i)]]
                    vj = x_vec[var_map[(l, 'V', j)]]
                    wk = x_vec[var_map[(l, 'W', k)]]
                    # d/d(ui) of ui*vj*wk = vj*wk
                    J[eq_idx, var_map[(l, 'U', i)]] += vj * wk
                    # d/d(vj) of ui*vj*wk = ui*wk
                    J[eq_idx, var_map[(l, 'V', j)]] += ui * wk
                    # d/d(wk) of ui*vj*wk = ui*vj
                    J[eq_idx, var_map[(l, 'W', k)]] += ui * vj
            return J
        
        # Newton iteration
        converged = False
        for it in range(50):
            if deadline and time.time() > deadline:
                break
            r = residual(x)
            if np.max(np.abs(r)) < 1e-8:
                converged = True
                break
            J = jacobian(x)
            # Solve J * dx = -r
            try:
                dx = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                # Use least squares
                try:
                    dx = np.linalg.lstsq(J, -r, rcond=None)[0]
                except:
                    break
            # Damping
            alpha = 1.0
            for _ in range(10):
                x_new = x + alpha * dx
                r_new = residual(x_new)
                if np.linalg.norm(r_new) < np.linalg.norm(r):
                    x = x_new
                    break
                alpha *= 0.5
            else:
                break
        
        if not converged:
            continue
        
        # Round to nearest integer or half-integer
        x_rounded = np.round(x * 2) / 2.0
        
        # Check if rounded solution works
        r_rounded = residual(x_rounded)
        if np.max(np.abs(r_rounded)) > 1e-6:
            # Try rounding to integers
            x_rounded_int = np.round(x)
            r_int = residual(x_rounded_int)
            if np.max(np.abs(r_int)) > 1e-6:
                continue
        
        # Build U, V, W from the solution
        x_use = x_rounded if np.max(np.abs(r_rounded)) <= 1e-6 else x_rounded_int
        U = np.zeros((nU, rank))
        V = np.zeros((nV, rank))
        W = np.zeros((nW, rank))
        
        for l in range(rank):
            su, sv, sw = supports[l]
            for idx in su:
                U[idx, l] = x_use[var_map[(l, 'U', idx)]]
            for idx in sv:
                V[idx, l] = x_use[var_map[(l, 'V', idx)]]
            for idx in sw:
                W[idx, l] = x_use[var_map[(l, 'W', idx)]]
        
        # Full verification
        if verify_fast(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None


def build_rank11_233():
    """
    Build a rank-11 decomposition for 2x3x3 matrix multiplication.
    A (2x3) * B (3x3) = C (2x3)
    
    This uses Strassen-like ideas extended to the rectangular case.
    """
    n, m, p = 2, 3, 3
    nU, nV, nW = n * m, m * p, p * n  # 6, 9, 6
    
    U = np.zeros((nU, 11))
    V = np.zeros((nV, 11))
    W = np.zeros((nW, 11))
    
    # Strassen for the 2x2x2 core (rows 0,1 of A; cols 0,1 of B; output rows 0,1; cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1  # B[0,0] and B[1,1] -> indices 0 and 5
    W[0, 0] = 1; W[1, 0] = 1; W[3, 0] = 1; W[4, 0] = 1  # C[0,0],C[0,1],C[1,0],C[1,1]
    
    # M2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[3, 2] = 1; V[5, 2] = -1  # B[1,0]=idx3, B[1,1]=idx5
    W[0, 2] = -1; W[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1  # B[0,1]=idx1, B[0,0]=idx0
    W[0, 3] = -1; W[4, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[3, 5] = 1
    W[0, 5] = 1; W[4, 5] = -1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[1, 6] = 1; V[5, 6] = 1
    W[1, 6] = -1; W[3, 6] = 1
    
    # Now handle the third column (b=2) and third output column (k=2)
    # Terms involving A[:,2] or B[2,:] or C[:,2]
    
    # M8 = A02 * (B20 + B21 + B22)  -> contributes to C[0,0],C[0,1],C[0,2]
    U[2, 7] = 1
    V[6, 7] = 1; V[7, 7] = 1; V[8, 7] = 1  # B[2,0]=6, B[2,1]=7, B[2,2]=8
    W[0, 7] = 1; W[1, 7] = 1; W[2, 7] = 1
    
    # M9 = A12 * (B20 + B21 + B22)
    U[5, 8] = 1
    V[6, 8] = 1; V[7, 8] = 1; V[8, 8] = 1
    W[3, 8] = 1; W[4, 8] = 1; W[5, 8] = 1
    
    # But we also need C[0,2] and C[1,2] from the first two columns:
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    # B02 is index 2, B12 is index 5
    # We already have A02*B22 in M8 and A12*B22 in M9.
    # Need: A00*B02 + A01*B12 for C[0,2] and A10*B02 + A11*B12 for C[1,2]
    
    # M10 = (A00+A10) * B02 + ... hmm, this is getting complex.
    # Let's use: M10 = (A00+A10)*B02
    U[0, 9] = 1; U[3, 9] = 1
    V[2, 9] = 1  # B[0,2] = index 2
    W[2, 9] = 1; W[5, 9] = 1  # C[0,2]=idx2, C[1,2]=idx5
    
    # M11 = (A01+A11) * B12
    U[1, 10] = 1; U[4, 10] = 1
    V[5, 10] = 1  # B[1,2] = index 5
    W[2, 10] = 1; W[5, 10] = 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy 1: Try Newton-based search for low rank
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something... 
    # Actually, rank >= max(n, m, p) * min(n, m, p) for some bounds.
    # Schoolbook: n*m*p = 36
    # Strassen-like: try ranks from 18 to 30
    
    # First, try the block approach with rank-11 for 2x3x3
    U_233, V_233, W_233 = build_rank11_233()
    
    # Verify 2x3x3
    if verify_fast(U_233, V_233, W_233, 2, 3, 3, tol=1e-6):
        # Assemble 2x3x6 from two 2x3x3 blocks
        U_full = np.zeros((6, 22))
        V_full = np.zeros((18, 22))
        W_full = np.zeros((12, 22))
        
        for block in range(2):
            offset = block * 11
            U_full[:, offset:offset+11] = U_233
            
            # V: shape (m*p, rank) = (3*6, rank)
            # V index: b*p + d where b in [0,3), d in [0,6)
            # For block j (columns 3j to 3j+2 of B), V[b*6 + 3j + k, l] = V_233[b*3 + k, l]
            for b in range(3):
                for k in range(3):
                    V_full[b * 6 + block * 3 + k, offset:offset+11] = V_233[b * 3 + k]
            
            # W: shape (p*n, rank) = (6*2, rank) = (12, rank)
            # W index: k*n + i where k in [0,6), i in [0,2)
            # For block j, output columns 3j to 3j+2
            for k in range(3):
                for i in range(2):
                    W_full[(block * 3 + k) * 2 + i, offset:offset+11] = W_233[k * 2 + i]
        
        if verify_fast(U_full, V_full, W_full, n, m, p, tol=1e-6):
            # Rank 22 achieved!
            # Try to do better with Newton search
            if time.time() < deadline - 30:
                for target_rank in [20, 21, 19, 18, 17, 16, 15]:
                    if time.time() > deadline - 10:
                        break
                    result = try_newton_decomposition(n, m, p, target_rank, 
                                                       num_trials=20, seed=42,
                                                       deadline=deadline - 5)
                    if result is not None:
                        U, V, W = result
                        if verify_fast(U, V, W, n, m, p, tol=1e-6):
                            return {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
            
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: rank-27 from 2x3x2 blocks (rank 9 each)
    n2, m2, p2 = 2, 3, 2
    U_232 = np.zeros((6, 9))
    V_232 = np.zeros((6, 9))
    W_232 = np.zeros((4, 9))
    
    # Strassen for 2x2x2 part
    U_232[0, 0] = 1; U_232[4, 0] = 1
    V_232[0, 0] = 1; V_232[3, 0] = 1
    W_232[0, 0] = 1; W_232[1, 0] = 1; W_232[2, 0] = 1; W_232[3, 0] = 1
    
    U_232[3, 1] = 1; U_232[4, 1] = 1
    V_232[0, 1] = 1
    W_232[1, 1] = 1; W_232[2, 1] = -1
    
    U_232[0, 2] = 1
    V_232[2, 2] = 1; V_232[3, 2] = -1
    W_232[0, 2] = -1; W_232[2, 2] = 1
    
    U_232[4, 3] = 1
    V_232[1, 3] = 1; V_232[0, 3] = -1
    W_232[0, 3] = -1; W_232[3, 3] = 1
    
    U_232[0, 4] = 1; U_232[1, 4] = 1
    V_232[3, 4] = 1
    W_232[1, 4] = -1; W_232[2, 4] = 1
    
    U_232[3, 5] = 1; U_232[0, 5] = -1
    V_232[0, 5] = 1; V_232[2, 5] = 1
    W_232[0, 5] = 1; W_232[3, 5] = -1
    
    U_232[1, 6] = 1; U_232[4, 6] = -1
    V_232[1, 6] = 1; V_232[3, 6] = 1
    W_232[1, 6] = -1; W_232[2, 6] = 1
    
    U_232[2, 7] = 1
    V_232[4, 7] = 1; V_232[5, 7] = 1
    W_232[0, 7] = 1; W_232[1, 7] = 1
    
    U_232[5, 8] = 1
    V_232[4, 8] = 1; V_232[5, 8] = 1
    W_232[2, 8] = 1; W_232[3, 8] = 1
    
    if verify_fast(U_232, V_232, W_232, 2, 3, 2, tol=1e-6):
        U_full = np.zeros((6, 27))
        V_full = np.zeros((18, 27))
        W_full = np.zeros((12, 27))
        
        for block in range(3):
            offset = block * 9
            U_full[:, offset:offset+9] = U_232
            for b in range(3):
                for k in range(2):
                    V_full[b * 6 + block * 2 + k, offset:offset+9] = V_232[b * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_full[(block * 2 + k) * 2 + i, offset:offset+9] = W_232[k * 2 + i]
        
        if verify_fast(U_full, V_full, W_full, n, m, p, tol=1e-6):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Last resort: schoolbook rank 36
    U_sb = np.zeros((6, 36))
    V_sb = np.zeros((18, 36))
    W_sb = np.zeros((12, 36))
    idx = 0
    for a in range(n):
        for b in range(m):
            for d in range(p):
                U_sb[a * m + b, idx] = 1
                V_sb[b * p + d, idx] = 1
                W_sb[d * n + a, idx] = 1
                idx += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }


if __name__ == "__main__":
    construct()
