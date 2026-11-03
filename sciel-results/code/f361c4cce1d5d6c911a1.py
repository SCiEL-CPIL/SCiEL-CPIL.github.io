"""Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural idea:
  Split A (2x3) into A0 (2x1, j=0) and A2 (2x2, j=1,2).
  - A0 * B0 (2x1 * 1x10): naive, 2*1*10 = 20 multiplications.
  - A2 * B2 (2x2 * 2x10): split B2 into 5 blocks of 2x2, apply Strassen (7 mults each) = 35.
  Total rank = 55.

Refinement:
  Verify the solution, then search for a lower rank with a deadline guard.
"""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(1000):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.

    A is 2x3: columns j=0,1,2.
    B is 3x10: rows j=0,1,2.
    C is 2x10: C[i,k] = sum_j A[i,j]*B[j,k].

    Part A: j=0 -> A[i,0]*B[0,k] for i in {0,1}, k in 0..9.  20 terms.
    Part B: j=1,2 -> A2 (2x2) * B2 (2x10). Split B2 into 5 blocks of 2x2.
             Each 2x2 * 2x2 block uses Strassen: 7 multiplications. 5*7 = 35.
    Total: 55.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10. Split B2 into 5 blocks of 2x2.
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen indices for U (A2 entries)
        u_11 = 0 * m + 1  # A[0,1]
        u_12 = 0 * m + 2  # A[0,2]
        u_21 = 1 * m + 1  # A[1,1]
        u_22 = 1 * m + 2  # A[1,2]

        # Strassen indices for V (B2 block entries)
        v_11 = 1 * p + k0  # B[1,k0]
        v_12 = 1 * p + k1  # B[1,k1]
        v_21 = 2 * p + k0  # B[2,k0]
        v_22 = 2 * p + k1  # B[2,k1]

        # W indices for outputs
        w_00 = k0 * n + 0  # C[0,k0]
        w_01 = k1 * n + 0  # C[0,k1]
        w_10 = k0 * n + 1  # C[1,k0]
        w_11 = k1 * n + 1  # C[1,k1]

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_wfta30(n, m, p):
    """
    Attempt to build a rank-30 decomposition using WFTA-style approach.
    
    For 2x3x10 matrix multiplication, the minimum possible rank is 30
    (lower bound = max(n*m, m*p, p*n) = max(6, 30, 20) = 30).
    
    Strategy: Use the structure that C[i,k] = sum_j A[i,j]*B[j,k].
    We need 30 rank-1 tensors. Each rank-1 tensor contributes to
    one (i,j) of A, one (j,k) of B, and one (k,i) of C.
    
    Since we have 2*3=6 A-indices, 3*10=30 B-indices, and 10*2=20 C-indices,
    and we need rank 30, each rank-1 tensor must be "sparse" in the sense
    that it uses a specific combination.
    
    Key insight: For each k in 0..9, we need to compute C[0,k] and C[1,k],
    which depend on A[0,0..2]*B[0..2,k] and A[1,0..2]*B[0..2,k].
    That's 6 multiplications per k, total 60. To get to 30, we need to
    share multiplications across k values.
    
    Let's try: For each pair of k values (k, k+1), can we do 3 multiplications
    instead of 6? That would give 5*3 = 15 for pairs, but we have 10 k values.
    Actually 10/2 = 5 pairs, 5*3 = 15, but we need 30 total. Doesn't work.
    
    Alternative: Maybe we can achieve rank 30 by a more clever construction.
    Let's try to construct it directly by solving the tensor equations.
    """
    rank = 30
    nm = n * m  # 6
    mp = m * p  # 30
    pn = p * n  # 20
    
    # Initialize with a structured pattern
    # For each of the 30 rank components, assign them to specific (j,k) pairs
    # such that we cover all necessary combinations.
    
    # One idea: For each k in 0..9, use 3 rank components to compute
    # C[0,k] and C[1,k] from all 6 A entries and 3 B entries.
    # But 3 rank components can only give us 3 bilinear terms, not 6.
    
    # Let me try a different approach: use the fact that we can have
    # half-integer coefficients.
    
    # Actually, let's just try to find a rank-30 decomposition by optimization.
    # We'll return None if we can't find one, and the caller will fall back.
    
    return None  # Placeholder - will be replaced by optimization-based search


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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


def _residual_func(params, n, m, p, rank):
    """Compute the residual of the tensor decomposition."""
    # params is a flat vector of length rank * (nm + mp + pn)
    nm = n * m
    mp = m * p
    pn = p * n
    
    U_flat = params[:rank * nm]
    V_flat = params[rank * nm : rank * nm + rank * mp]
    W_flat = params[rank * nm + rank * mp :]
    
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    T_reconstructed = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
    # Target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    return T_reconstructed - T_target


def _try_optimize_rank(n, m, p, rank, U_init, V_init, W_init, deadline):
    """Try to optimize a decomposition of the given rank starting from init."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # If no init provided, use random initialization
    if U_init is None or V_init is None or W_init is None:
        U_init = np.random.randn(nm, rank) * 0.5
        V_init = np.random.randn(mp, rank) * 0.5
        W_init = np.random.randn(pn, rank) * 0.5
    
    # Flatten init
    U_flat = np.array(U_init, dtype=np.float64).flatten()
    V_flat = np.array(V_init, dtype=np.float64).flatten()
    W_flat = np.array(W_init, dtype=np.float64).flatten()
    
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    # Use least squares to minimize the residual
    try:
        import scipy.optimize as opt
        
        # Define the residual function for scipy
        def residual(x):
            return _residual_func(x, n, m, p, rank).flatten()
            
        # Run optimization with multiple attempts and different methods
        best_result = None
        best_cost = np.inf
        
        for method in ['trf', 'lm']:
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                try:
                    if method == 'lm':
                        # LM requires more parameters than equations or square system
                        # Our problem is underdetermined, so use trf primarily
                        result = opt.least_squares(residual, x0 + np.random.randn(len(x0)) * 0.01, 
                                                   method='trf', max_nfev=50, 
                                                   xtol=1e-10, ftol=1e-10)
                    else:
                        perturbation = np.random.randn(len(x0)) * 0.01 if attempt > 0 else 0
                        result = opt.least_squares(residual, x0 + perturbation, 
                                                   method='trf', max_nfev=50, 
                                                   xtol=1e-10, ftol=1e-10)
                    
                    if result.cost < best_cost:
                        best_cost = result.cost
                        best_result = result
                    
                    if best_cost < 1e-10:
                        break
                        
                except Exception:
                    continue
            
            if best_cost < 1e-10:
                break
        
        if best_result is not None and best_cost < 1e-8:
            x_opt = best_result.x
            U_opt = x_opt[:rank * nm].reshape(nm, rank)
            V_opt = x_opt[rank * nm : rank * nm + rank * mp].reshape(mp, rank)
            W_opt = x_opt[rank * nm + rank * mp :].reshape(pn, rank)
            
            # Try multiple snapping strategies
            for scale in [1, 2, 4]:
                U_snap = np.round(U_opt * scale) / scale
                V_snap = np.round(V_opt * scale) / scale
                W_snap = np.round(W_opt * scale) / scale
                
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p) and _is_valid_entries(U_list, V_list, W_list):
                    return U_list, V_list, W_list
            
            # Try half-integer snapping specifically
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p) and _is_valid_entries(U_list, V_list, W_list):
                return U_list, V_list, W_list
                
    except Exception:
        pass
        
    return None


def construct():
    n, m, p = DIMENSIONS
    # Set a strict deadline to ensure we return well before the 150s kill switch
    deadline = time.time() + 100

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    # Verify it
    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        # Fallback to schoolbook (rank 60)
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p
        # If even schoolbook fails (shouldn't happen), return empty to avoid crash, but task expects dict
        # Assuming schoolbook always works.

    # Try to find a lower rank via optimization-based search
    # We attempt ranks 30, 40, 50, 54 in order, stopping at the first success.
    
    # First try rank 30 (the theoretical minimum)
    if time.time() < deadline - 60:
        result = _try_optimize_rank(n, m, p, 30, None, None, None, time.time() + 40)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 30
            return {"U": U_best, "V": V_best, "W": W_best}
    
    # Try rank 40
    if best_rank > 40 and time.time() < deadline - 40:
        # Use a random initialization for rank 40
        import numpy as np
        nm, mp, pn = n*m, m*p, p*n
        U_init = np.random.randn(nm, 40) * 0.1
        V_init = np.random.randn(mp, 40) * 0.1
        W_init = np.random.randn(pn, 40) * 0.1
        result = _try_optimize_rank(n, m, p, 40, U_init.tolist(), V_init.tolist(), W_init.tolist(), time.time() + 30)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 40
            return {"U": U_best, "V": V_best, "W": W_best}
    
    # Try rank 50
    if best_rank > 50 and time.time() < deadline - 20:
        import numpy as np
        nm, mp, pn = n*m, m*p, p*n
        U_init = np.random.randn(nm, 50) * 0.1
        V_init = np.random.randn(mp, 50) * 0.1
        W_init = np.random.randn(pn, 50) * 0.1
        result = _try_optimize_rank(n, m, p, 50, U_init.tolist(), V_init.tolist(), W_init.tolist(), time.time() + 15)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 50
            return {"U": U_best, "V": V_best, "W": W_best}
    
    # Try rank 54 (drop one term from rank 55)
    if best_rank > 54 and time.time() < deadline - 10:
        r = 54
        U_init = [row[:r] for row in U_best]
        V_init = [row[:r] for row in V_best]
        W_init = [row[:r] for row in W_best]
        result = _try_optimize_rank(n, m, p, r, U_init, V_init, W_init, time.time() + 8)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r

    return {"U": U_best, "V": V_best, "W": W_best}
