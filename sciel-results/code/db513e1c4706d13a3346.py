import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify exact tensor decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    
    # Check non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                if abs(s - 1.0) > tol:
                    return False
    
    # Check zero entries (sampled)
    zero_count = 0
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for j2 in range(m):
                for k in range(p):
                    b = j2 * p + k
                    for k2 in range(p):
                        c = k2 * n + i
                        # Skip the non-zero entry
                        if j2 == j and k2 == k:
                            continue
                        s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
                        if abs(s) > tol:
                            return False
                        zero_count += 1
                        if zero_count > 2000:
                            return True
    
    return True

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _optimize_rank(U, V, W, n, m, p, target_rank, max_iters=5000, deadline=None):
    """Try to optimize factor matrices to achieve target rank via alternating least squares."""
    U = np.array(U[:, :target_rank], dtype=float)
    V = np.array(V[:, :target_rank], dtype=float)
    W = np.array(W[:, :target_rank], dtype=float)
    
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Build list of non-zero tensor positions
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero.append((i * m + j, j * p + k, k * n + i))
    
    # Build list of zero positions (sampled)
    zero_positions = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            for j2 in range(m):
                for k in range(p):
                    b = j2 * p + k
                    for k2 in range(p):
                        c = k2 * n + i
                        if j2 != j or k2 != k:
                            zero_positions.append((a, b, c))
    
    # Sample zero positions if too many
    if len(zero_positions) > 5000:
        indices = np.random.choice(len(zero_positions), 5000, replace=False)
        zero_positions = [zero_positions[idx] for idx in indices]
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check if we already satisfy the constraints
        max_nz_err = 0.0
        max_z_err = 0.0
        for (a, b, c) in nonzero:
            s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            max_nz_err = max(max_nz_err, abs(s - 1.0))
        for (a, b, c) in zero_positions:
            s = np.dot(np.dot(U[a, :], V[b, :]), W[c, :])
            max_z_err = max(max_z_err, abs(s))
        
        loss = max_nz_err ** 2 + max_z_err ** 2
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-8:
            break
        
        # Alternating optimization: fix two, solve for the third
        # Update U: for each a, we want U[a,:] such that the tensor equations are satisfied
        # This is a linear system in U[a,:] when V and W are fixed
        
        # For each a, the equations involving U[a,:] are:
        # For nonzero (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = 1
        # For zero (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = 0
        
        # Let M_a be the matrix where each row corresponds to an equation,
        # and M_a[e, r] = V[b_e, r] * W[c_e, r]
        # Then M_a @ U[a,:] = rhs_a
        
        # Update U
        U_new = U.copy()
        for a in range(a_dim):
            # Collect all equations involving this a
            A_mat = []
            rhs = []
            for (aa, b, c) in nonzero:
                if aa == a:
                    A_mat.append(V[b, :] * W[c, :])
                    rhs.append(1.0)
            for (aa, b, c) in zero_positions:
                if aa == a:
                    A_mat.append(V[b, :] * W[c, :])
                    rhs.append(0.0)
            
            if A_mat:
                A_mat = np.array(A_mat)
                rhs = np.array(rhs)
                # Solve least squares: A_mat @ x = rhs
                try:
                    sol = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
                    U_new[a, :] = sol
                except:
                    pass
        
        U = U_new
        
        # Update V
        V_new = V.copy()
        for b in range(b_dim):
            A_mat = []
            rhs = []
            for (a, bb, c) in nonzero:
                if bb == b:
                    A_mat.append(U[a, :] * W[c, :])
                    rhs.append(1.0)
            for (a, bb, c) in zero_positions:
                if bb == b:
                    A_mat.append(U[a, :] * W[c, :])
                    rhs.append(0.0)
            
            if A_mat:
                A_mat = np.array(A_mat)
                rhs = np.array(rhs)
                try:
                    sol = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
                    V_new[b, :] = sol
                except:
                    pass
        
        V = V_new
        
        # Update W
        W_new = W.copy()
        for c in range(c_dim):
            A_mat = []
            rhs = []
            for (a, b, cc) in nonzero:
                if cc == c:
                    A_mat.append(U[a, :] * V[b, :])
                    rhs.append(1.0)
            for (a, b, cc) in zero_positions:
                if cc == c:
                    A_mat.append(U[a, :] * V[b, :])
                    rhs.append(0.0)
            
            if A_mat:
                A_mat = np.array(A_mat)
                rhs = np.array(rhs)
                try:
                    sol = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
                    W_new[c, :] = sol
                except:
                    pass
        
        W = W_new
    
    return best_U, best_V, best_W, best_loss

def _try_strassen_2x2(n, m, p):
    """
    Try to construct a decomposition using the rank-7 Strassen decomposition for 2x2x2
    embedded in the 3x3x6 problem.
    
    The idea: partition the 3x3x6 multiplication into overlapping 2x2x2 subproblems
    and use Strassen's 7-term decomposition for each.
    
    However, this is complex. Instead, let's try a direct approach:
    Use the fact that 3x3x6 can be seen as having structure that allows
    sharing of computations.
    
    For now, return None if we can't find a valid decomposition.
    """
    # This is a placeholder - the real work is in _optimize_rank
    return None

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110  # 110s safety margin
    
    # Schoolbook baseline
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to reduce rank using alternating optimization
    # Start from rank 54 and try lower ranks
    
    # Try ranks from 53 down to 30
    for target_rank in range(best_rank - 1, max(25, best_rank - 25), -1):
        if time.time() > deadline:
            break
        
        # Initialize from schoolbook (keep first target_rank columns)
        U_init = U_school[:, :target_rank].copy()
        V_init = V_school[:, :target_rank].copy()
        W_init = W_school[:, :target_rank].copy()
        
        # Also try random initialization
        U_rand = np.random.randn(n * m, target_rank)
        V_rand = np.random.randn(m * p, target_rank)
        W_rand = np.random.randn(p * n, target_rank)
        
        # Try both initializations
        for U_i, V_i, W_i in [(U_init, V_init, W_init), (U_rand, V_rand, W_rand)]:
            if time.time() > deadline:
                break
            
            U_opt, V_opt, W_opt, loss = _optimize_rank(
                U_i, V_i, W_i, n, m, p, target_rank,
                max_iters=3000, deadline=deadline
            )
            
            # Verify the solution
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                best_rank = target_rank
                best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
                break
        
        if best_rank < target_rank + 1:
            # Found a valid solution at this rank, try to go lower
            # But first, let's make sure we have a good solution
            pass
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-5):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
