"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _build_tensor_support(n=3, m=3, p=6):
    """Build the list of (a, b, c) indices where the tensor has value 1."""
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    return support


def _compute_tensor_value(U, V, W, a, b, c):
    """Compute the tensor value at (a, b, c) = sum_r U[a,r]*V[b,r]*W[c,r]."""
    return np.dot(U[a, :], V[b, :] * W[c, :])


def _compute_error(U, V, W, support, n=3, m=3, p=6):
    """Compute the total error of the decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    err = 0.0
    # On-support: should be 1
    for (a, b, c) in support:
        val = _compute_tensor_value(U, V, W, a, b, c)
        err += (val - 1.0) ** 2
    # Off-support: should be 0
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = _compute_tensor_value(U, V, W, a, b, c)
                    err += val ** 2
    return err


def _als_step_U(U, V, W, support, n=3, m=3, p=6):
    """One ALS step: update U given V and W."""
    nm = n * m
    r = U.shape[1]
    new_U = np.zeros_like(U)
    for a in range(nm):
        # For each column r, U[a,r] = sum_{(b,c) in support with a fixed} V[b,r]*W[c,r]
        # This is a linear system: for each r, we want sum over support entries
        # Actually, for fixed V and W, U[a,:] is determined by:
        # For each (b,c) in support where a is fixed: U[a,:] must satisfy
        # sum_r U[a,r]*V[b,r]*W[c,r] = 1
        # This is underdetermined if multiple (b,c) share the same a.
        # Let's use least squares: minimize sum_{(b,c): a fixed} (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
        # Let M[a] be the matrix where row (b,c) is V[b,:]*W[c,:]
        # Then U[a,:] = pinv(M[a]) * ones
        rows = []
        for (sa, sb, sc) in support:
            if sa == a:
                rows.append(V[sb, :] * W[sc, :])
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            # Solve M^T M U[a,:] = M^T ones
            try:
                new_U[a, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_U[a, :] = U[a, :]
    return new_U


def _als_step_V(U, V, W, support, n=3, m=3, p=6):
    """One ALS step: update V given U and W."""
    mp = m * p
    r = V.shape[1]
    new_V = np.zeros_like(V)
    for b in range(mp):
        rows = []
        for (sa, sb, sc) in support:
            if sb == b:
                rows.append(U[sa, :] * W[sc, :])
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_V[b, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_V[b, :] = V[b, :]
    return new_V


def _als_step_W(U, V, W, support, n=3, m=3, p=6):
    """One ALS step: update W given U and V."""
    pn = p * n
    r = W.shape[1]
    new_W = np.zeros_like(W)
    for c in range(pn):
        rows = []
        for (sa, sb, sc) in support:
            if sc == c:
                rows.append(U[sa, :] * V[sb, :])
        if rows:
            M = np.array(rows)
            ones = np.ones(M.shape[0])
            try:
                new_W[c, :] = np.linalg.lstsq(M, ones, rcond=None)[0]
            except:
                new_W[c, :] = W[c, :]
    return new_W


def _try_joint_decomposition(n=3, m=3, p=6, deadline=None):
    """
    Try to find a joint decomposition exploiting the shared m=3 dimension.
    Uses ALS with smooth objective.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    nm, mp, pn = n * m, m * p, p * n
    support = _build_tensor_support(n, m, p)
    
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from 20 upward (theoretical lower bound is 18, but tensor rank is higher)
    for r in range(20, 45):
        if time.time() > deadline - 10:
            break
        
        n_restarts = max(3, int((deadline - time.time()) / 10))
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # ALS iterations
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                U = _als_step_U(U, V, W, support, n, m, p)
                V = _als_step_V(U, V, W, support, n, m, p)
                W = _als_step_W(U, V, W, support, n, m, p)
            
            # Check error
            err = _compute_error(U, V, W, support, n, m, p)
            if err < 1e-6:
                # Try to snap to half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                err_half = _compute_error(U_half, V_half, W_half, support, n, m, p)
                if err_half < 1e-6:
                    U, V, W = U_half, V_half, W_half
                else:
                    # Keep the real-valued solution if it's accurate enough
                    pass
                
                if err < 1e-6:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    break
        
        if best_result is not None:
            break
    
    return best_result


def _try_structured_decomposition(n=3, m=3, p=6):
    """
    Try to construct a lower-rank decomposition using the block structure.
    """
    start = time.time()
    deadline = time.time() + 100  # Leave 50s margin
    
    nm, mp, pn = n * m, m * p, p * n
    
    best_result = None
    best_rank = n * m * p
    
    # First, verify schoolbook works
    sb = _schoolbook(n, m, p)
    if _verify(sb["U"], sb["V"], sb["W"], n, m, p):
        best_result = sb
        best_rank = best_rank
    
    # Try joint decomposition
    joint = _try_joint_decomposition(n, m, p, deadline)
    if joint is not None:
        rank_joint = len(joint["U"][0]) if joint["U"] else 0
        if rank_joint < best_rank:
            best_result = joint
            best_rank = rank_joint
    
    return best_result


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix."""
    start = time.time()
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it works
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to find better
        better = _try_structured_decomposition(n, m, p)
        if better is not None:
            rank_better = len(better["U"][0]) if better["U"] else 0
            rank_current = len(result["U"][0]) if result["U"] else 0
            if rank_better < rank_current:
                result = better
    
    # Ensure we return before deadline
    if time.time() > deadline - 5:
        return result
    
    return result
# EVOLVE-BLOCK-END
