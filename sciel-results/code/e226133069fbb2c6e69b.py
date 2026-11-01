import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    rank = len(U[0])
    # Build the multiplication tensor T[A, B, C]
    # T[i*m+j, j*p+k, k*n+i] = 1
    # All other entries are 0.
    
    # We can check specific non-zero entries and ensure zero entries are 0.
    # It's faster to check all non-zero entries and a sample of zero entries.
    # Or just check all non-zero entries if we trust the structure.
    # Actually, checking all 2*3*3 = 18 non-zero entries is cheap.
    # Checking zero entries: (n*m)*(m*p)*(p*n) - 18 = 18*9*6 - 18 = 972 - 18 = 954 entries.
    # 954 checks is also cheap.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = sum(U[A][r] * V[B][r] * W[C][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
                
                # Check zero entries? 
                # It's safer to check a few random zero entries to ensure no accidental 1s.
                # But if the non-zero entries are exactly 1, and the rank is low, 
                # it's likely the zero entries are 0.
                # However, for "exact" verification, we should check all.
                # 972 iterations is trivial.
    
    # Check zero entries
    # Generate all (A, B, C) not in the non-zero set
    non_zero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                non_zero_set.add((A, B, C))
                
    for A in range(n * m):
        for B in range(m * p):
            for C in range(p * n):
                if (A, B, C) in non_zero_set:
                    continue
                actual = sum(U[A][r] * V[B][r] * W[C][r] for r in range(rank))
                if abs(actual) > tol:
                    return False, (A, B, C, 0, actual)
                    
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _als_decompose(T, n, m, p, rank, max_iters=100, tol=1e-9, rng=np.random.default_rng(42)):
    """
    Alternating Least Squares for tensor decomposition.
    T is a 3D array of shape (n*m, m*p, p*n).
    Returns U, V, W as 2D arrays of shape (dim, rank).
    """
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Initialize
    U = rng.normal(size=(dim_A, rank))
    V = rng.normal(size=(dim_B, rank))
    W = rng.normal(size=(dim_C, rank))
    
    # Normalize to prevent divergence
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    prev_err = np.inf
    for it in range(max_iters):
        # Update U
        # T = sum_r U[:, r] outer V[:, r] outer W[:, r]
        # For each r, U[:, r] = argmin ||T - sum U[:, r] outer V[:, r] outer W[:, r]||
        # This is a least squares problem.
        # T can be reshaped.
        # Let's use the property: 
        # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
        # Fix V, W. Solve for U.
        # For each a, U_{ar} are variables.
        # T_{a, :, :} = sum_r U_{ar} (V[:, r] outer W[:, r])
        # This is a linear system for each a? No, U_{ar} is a vector of length rank.
        # Let M_r = V[:, r] outer W[:, r] (shape dim_B x dim_C)
        # T[:, :, :] (shape dim_A x dim_B x dim_C)
        # We want U such that T approx sum_r U[:, r] outer M_r
        # This is equivalent to:
        # For each a, T[a] approx sum_r U[a, r] M_r
        # Let T_flat[a] = T[a].flatten() (length dim_B * dim_C)
        # Let M_flat[r] = M_r.flatten() (length dim_B * dim_C)
        # Then T_flat[a] approx U[a] dot M_flat[r] for all r? 
        # No, sum_r U[a, r] M_flat[r] is a vector.
        # So T_flat[a] = sum_r U[a, r] M_flat[r]
        # This is a matrix equation: T_flat (dim_A x dim_B*dim_C) = U (dim_A x rank) * M_flat^T (rank x dim_B*dim_C)
        # So U = T_flat * pinv(M_flat^T)
        
        M_flat = np.zeros((rank, dim_B * dim_C))
        for r in range(rank):
            M_r = np.outer(V[:, r], W[:, r])
            M_flat[r] = M_r.flatten()
            
        T_flat = T.reshape(dim_A, -1)
        
        # Solve U * M_flat^T = T_flat
        # U = T_flat * (M_flat^T)^+
        try:
            U_new = T_flat @ np.linalg.pinv(M_flat.T)
        except:
            break
            
        # Update V
        # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
        # Fix U, W.
        # T[:, b, :] = sum_r V[b, r] (U[:, r] outer W[:, r])
        # T_flat[b] = sum_r V[b, r] N_flat[r]
        # T_flat (dim_B x dim_A*dim_C) = V (dim_B x rank) * N_flat^T (rank x dim_A*dim_C)
        # Note: T_flat here is T.reshape(dim_B, -1) but careful with ordering.
        # T is (A, B, C).
        # T.reshape(dim_B, -1) flattens A and C.
        # We need to align with N_r = U[:, r] outer W[:, r] which is (A, C).
        # So we need T to be (B, A, C).
        T_perm_B = np.transpose(T, (1, 0, 2)) # (B, A, C)
        T_flat_B = T_perm_B.reshape(dim_B, -1)
        
        N_flat = np.zeros((rank, dim_A * dim_C))
        for r in range(rank):
            N_r = np.outer(U[:, r], W[:, r])
            N_flat[r] = N_r.flatten()
            
        try:
            V_new = T_flat_B @ np.linalg.pinv(N_flat.T)
        except:
            break
            
        # Update W
        # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
        # Fix U, V.
        # T[:, :, c] = sum_r W[c, r] (U[:, r] outer V[:, r])
        # T_flat[c] = sum_r W[c, r] P_flat[r]
        # T_perm_C (C, A, B)
        T_perm_C = np.transpose(T, (2, 0, 1)) # (C, A, B)
        T_flat_C = T_perm_C.reshape(dim_C, -1)
        
        P_flat = np.zeros((rank, dim_A * dim_B))
        for r in range(rank):
            P_r = np.outer(U[:, r], V[:, r])
            P_flat[r] = P_r.flatten()
            
        try:
            W_new = T_flat_C @ np.linalg.pinv(P_flat.T)
        except:
            break
            
        U, V, W = U_new, V_new, W_new
        
        # Calculate error
        # T_approx = sum_r U[:, r] outer V[:, r] outer W[:, r]
        T_approx = np.zeros_like(T)
        for r in range(rank):
            T_approx += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(T - T_approx)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    return U, V, W

def _round_to_rational(x, max_denom=12, tol=1e-6):
    """Find a rational a/b close to x with b <= max_denom."""
    best_val = 0
    best_err = np.inf
    for b in range(1, max_denom + 1):
        a = round(x * b)
        val = a / b
        err = abs(x - val)
        if err < best_err:
            best_err = err
            best_val = val
    if best_err < tol:
        return best_val
    return None

def _try_rank(n, m, p, rank, time_budget, rng):
    """Try to find a decomposition of given rank."""
    dim_A = n * m
    dim_B = m * p
    dim_C = p * n
    
    # Build T
    T = np.zeros((dim_A, dim_B, dim_C))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                T[A, B, C] = 1
                
    start_time = time.time()
    trials = 10
    
    for trial in range(trials):
        if time.time() - start_time > time_budget:
            break
            
        U, V, W = _als_decompose(T, n, m, p, rank, max_iters=50, tol=1e-12, rng=rng)
        
        # Check if error is small
        T_approx = np.zeros_like(T)
        for r in range(rank):
            T_approx += np.einsum('ar,br,cr->abc', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(T - T_approx)
        
        if err < 1e-6:
            # Try to round
            U_r = np.zeros_like(U)
            V_r = np.zeros_like(V)
            W_r = np.zeros_like(W)
            
            success = True
            for i in range(dim_A):
                for r in range(rank):
                    val = _round_to_rational(U[i, r], max_denom=12)
                    if val is None:
                        success = False
                        break
                    U_r[i, r] = val
                if not success:
                    break
            
            if success:
                for i in range(dim_B):
                    for r in range(rank):
                        val = _round_to_rational(V[i, r], max_denom=12)
                        if val is None:
                            success = False
                            break
                        V_r[i, r] = val
                    if not success:
                        break
            
            if success:
                for i in range(dim_C):
                    for r in range(rank):
                        val = _round_to_rational(W[i, r], max_denom=12)
                        if val is None:
                            success = False
                            break
                        W_r[i, r] = val
                    if not success:
                        break
                        
            if success:
                # Verify exactness
                ok, _ = _verify_factors(U_r, V_r, W_r, n, m, p, tol=1e-9)
                if ok:
                    return U_r, V_r, W_r
                    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = 150 - 10  # 140s safety margin
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Try ranks from 11 to 17
    for rank in range(11, 18):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
            
        # Allocate time budget based on rank
        # Lower ranks are harder, give them more time?
        # Or just split remaining time?
        # Let's give 10s to each rank attempt initially.
        budget = min(10, deadline - elapsed)
        
        result = _try_rank(n, m, p, rank, budget, rng)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
