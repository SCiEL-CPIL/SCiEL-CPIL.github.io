"""
MILP-style exact tensor decomposition for 4x5x6 matrix multiplication.
Structural change: formulate as a linear feasibility problem over a structured
basis of rank-1 tensors, using column-generation + linear programming.
"""
import time
import numpy as np
from scipy.optimize import linprog

_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0

def _time_left():
    return _DEADLINE - time.time()

def _schoolbook_decomp(n, m, p):
    """Schoolbook decomposition for n x m @ m x p (rank n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-4):
    """Fast vectorized check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_basis_tensors(n, m, p):
    """
    Build a structured basis of rank-1 tensors for the 4x5x6 problem.
    
    Each basis tensor is defined by a pattern of non-zero entries in U, V, W.
    We use a hierarchical basis:
    - Level 0: full rank-1 tensors (schoolbook basis)
    - Level 1: tensors with sparse support (e.g., one index fixed)
    - Level 2: tensors with block structure
    
    For the 4x5x6 case, we generate basis elements that exploit the
    rectangular geometry.
    """
    basis = []
    
    # Schoolbook basis: e_i * e_j * e_k for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
                basis.append((u, v, w))
    
    # Additional sparse basis: for each pair of indices, create tensors
    # that spread across multiple positions
    # Pattern: fix i, vary j and k
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1 + 1, m):
                for k1 in range(p):
                    for k2 in range(k1 + 1, p):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[i * m + j1] = 1.0
                        u[i * m + j2] = 1.0
                        v[j1 * p + k1] = 1.0
                        v[j2 * p + k2] = 1.0
                        w[k1 * n + i] = 1.0
                        w[k2 * n + i] = 1.0
                        basis.append((u, v, w))
    
    # Pattern: fix j, vary i and k
    for j in range(m):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                for k1 in range(p):
                    for k2 in range(k1 + 1, p):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[i1 * m + j] = 1.0
                        u[i2 * m + j] = 1.0
                        v[j * p + k1] = 1.0
                        v[j * p + k2] = 1.0
                        w[k1 * n + i1] = 1.0
                        w[k2 * n + i2] = 1.0
                        basis.append((u, v, w))
    
    # Pattern: fix k, vary i and j
    for k in range(p):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                for j1 in range(m):
                    for j2 in range(j1 + 1, m):
                        u = np.zeros(n * m)
                        v = np.zeros(m * p)
                        w = np.zeros(p * n)
                        u[i1 * m + j1] = 1.0
                        u[i2 * m + j2] = 1.0
                        v[j1 * p + k] = 1.0
                        v[j2 * p + k] = 1.0
                        w[k * n + i1] = 1.0
                        w[k * n + i2] = 1.0
                        basis.append((u, v, w))
    
    return basis

def _greedy_milp_decomp(n, m, p, max_rank=100, max_basis=500):
    """
    Greedy column-generation approach to find a low-rank decomposition.
    
    1. Start with a small set of basis tensors.
    2. Solve LP to find coefficients.
    3. Add most violated basis tensor.
    4. Repeat until convergence or rank limit.
    """
    if _time_left() < 10.0:
        return None
    
    # Build basis
    basis = _build_basis_tensors(n, m, p)
    if len(basis) > max_basis:
        # Subsample for speed
        indices = np.random.choice(len(basis), max_basis, replace=False)
        basis = [basis[i] for i in sorted(indices)]
    
    B = len(basis)
    num_eq = n * m * p  # 120 equations
    
    # Precompute the constraint matrix A
    # A[e, b] = contribution of basis tensor b to equation e
    # Equation e corresponds to (i, j, k)
    def eq_index(i, j, k):
        return i * m * p + j * p + k
    
    A = np.zeros((num_eq, B))
    for b_idx, (u, v, w) in enumerate(basis):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    e = eq_index(i, j, k)
                    # Contribution: u[i*m+j] * v[j*p+k] * w[k*n+i]
                    contrib = u[i * m + j] * v[j * p + k] * w[k * n + i]
                    A[e, b_idx] = contrib
    
    # Start with schoolbook basis (first n*m*p basis tensors)
    initial_basis = list(range(n * m * p))
    
    # Iteratively add basis tensors
    current_basis = list(initial_basis)
    
    for iteration in range(50):
        if _time_left() < 8.0:
            break
        
        # Solve LP: minimize sum of coefficients subject to A[:, current_basis] @ x = 1
        # Actually, we want to find x such that sum_b x[b] * A[:, b] = 1
        # This is a feasibility problem. Use linprog with objective = sum(x)
        
        A_sub = A[:, current_basis]
        B_vec = np.ones(num_eq)
        
        # Minimize sum(x) subject to A_sub @ x = B_vec, x >= 0
        c = np.ones(len(current_basis))
        
        # Equality constraints
        A_eq = A_sub
        b_eq = B_vec
        
        bounds = [(0, None)] * len(current_basis)
        
        try:
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        except Exception:
            break
        
        if not res.success:
            break
        
        x = res.x
        
        # Check if solution is valid
        U = np.zeros((n * m, len(current_basis)))
        V = np.zeros((m * p, len(current_basis)))
        W = np.zeros((p * n, len(current_basis)))
        
        for col_idx, b_idx in enumerate(current_basis):
            u, v, w = basis[b_idx]
            U[:, col_idx] = u * x[col_idx]
            V[:, col_idx] = v * x[col_idx]
            W[:, col_idx] = w * x[col_idx]
        
        # Remove zero columns
        nonzero_mask = np.abs(x) > 1e-10
        if not np.all(nonzero_mask):
            U = U[:, nonzero_mask]
            V = V[:, nonzero_mask]
            W = W[:, nonzero_mask]
            x = x[nonzero_mask]
            current_basis = [current_basis[i] for i in range(len(current_basis)) if nonzero_mask[i]]
        
        # Verify
        if _verify_np(U, V, W, n, m, p, 1e-6):
            if _verify_decomp(U, V, W, n, m, p, 1e-8):
                return U, V, W
        
        # Find most violated basis tensor to add
        # Compute residual
        residual = B_vec - A_sub @ x
        
        # Find basis tensor that best correlates with residual
        # For each basis tensor not in current set, compute correlation
        best_gain = -1
        best_idx = -1
        
        for b_idx in range(B):
            if b_idx in current_basis:
                continue
            # Correlation with residual
            gain = np.dot(A[:, b_idx], residual)
            if gain > best_gain:
                best_gain = gain
                best_idx = b_idx
        
        if best_idx < 0 or best_gain < 1e-6:
            break
        
        current_basis.append(best_idx)
        
        if len(current_basis) > max_rank:
            break
    
    # Final attempt with current basis
    if len(current_basis) > 0:
        A_sub = A[:, current_basis]
        B_vec = np.ones(num_eq)
        c = np.ones(len(current_basis))
        A_eq = A_sub
        b_eq = B_vec
        bounds = [(0, None)] * len(current_basis)
        
        try:
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        except Exception:
            return None
        
        if res.success:
            x = res.x
            U = np.zeros((n * m, len(current_basis)))
            V = np.zeros((m * p, len(current_basis)))
            W = np.zeros((p * n, len(current_basis)))
            
            for col_idx, b_idx in enumerate(current_basis):
                u, v, w = basis[b_idx]
                U[:, col_idx] = u * x[col_idx]
                V[:, col_idx] = v * x[col_idx]
                W[:, col_idx] = w * x[col_idx]
            
            nonzero_mask = np.abs(x) > 1e-10
            if not np.all(nonzero_mask):
                U = U[:, nonzero_mask]
                V = V[:, nonzero_mask]
                W = W[:, nonzero_mask]
            
            if _verify_np(U, V, W, n, m, p, 1e-6):
                if _verify_decomp(U, V, W, n, m, p, 1e-8):
                    return U, V, W
    
    return None

def _iterative_refinement(n, m, p, initial_rank=80, max_iters=100):
    """
    Iterative refinement: start with a random decomposition, then
    alternate between solving for each matrix.
    """
    if _time_left() < 10.0:
        return None
    
    R = initial_rank
    
    for attempt in range(3):
        if _time_left() < 8.0:
            break
        
        np.random.seed(42 + attempt)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for it in range(max_iters):
            if _time_left() < 5.0:
                break
            
            # Solve for W
            A = np.zeros((p, n, R))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        A[k, i, :] += U[i * m + j, :] * V[j * p + k, :]
            
            W_new = np.zeros((p * n, R))
            valid = True
            for k in range(p):
                for i in range(n):
                    a = A[k, i, :]
                    if np.linalg.norm(a) < 1e-10:
                        valid = False
                        break
                    W_new[k * n + i, :] = a / (a @ a)
            if not valid:
                break
            W = W_new
            
            # Solve for U
            B = np.zeros((n, m, R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        B[i, j, :] += V[j * p + k, :] * W[k * n + i, :]
            
            U_new = np.zeros((n * m, R))
            valid = True
            for i in range(n):
                for j in range(m):
                    b = B[i, j, :]
                    if np.linalg.norm(b) < 1e-10:
                        valid = False
                        break
                    U_new[i * m + j, :] = b / (b @ b)
            if not valid:
                break
            U = U_new
            
            # Solve for V
            C = np.zeros((m, p, R))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        C[j, k, :] += U[i * m + j, :] * W[k * n + i, :]
            
            V_new = np.zeros((m * p, R))
            valid = True
            for j in range(m):
                for k in range(p):
                    c = C[j, k, :]
                    if np.linalg.norm(c) < 1e-10:
                        valid = False
                        break
                    V_new[j * p + k, :] = c / (c @ c)
            if not valid:
                break
            V = V_new
            
            if it % 10 == 0:
                if _verify_np(U, V, W, n, m, p, 1e-6):
                    if _verify_decomp(U, V, W, n, m, p, 1e-8):
                        return U, V, W
        
        if _verify_np(U, V, W, n, m, p, 1e-6):
            if _verify_decomp(U, V, W, n, m, p, 1e-8):
                return U, V, W
    
    return None

def _local_reduce(U, V, W, n, m, p, max_iters=10, budget_s=20.0):
    """Local column-removal search."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    for _ in range(max_iters):
        if _time_left() < 3.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        for c in range(R):
            if _time_left() < 3.0:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_np(U2, V2, W2, n, m, p, 1e-4):
                if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                    U, V, W = U2, V2, W2
                    R -= 1
                    found = True
                    break
        if not found:
            break
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    
    # Fallback: schoolbook (always valid, rank 120)
    U, V, W = _schoolbook_decomp(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U, V, W
    
    # Try MILP-style column generation
    try:
        if _time_left() > 30.0:
            cand = _greedy_milp_decomp(n, m, p, max_rank=90, max_basis=300)
            if cand is not None:
                U, V, W = cand
                if _verify_decomp(U, V, W, n, m, p, 1e-9):
                    if U.shape[1] < best_rank:
                        best_rank = U.shape[1]
                        best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Try iterative refinement
    try:
        if _time_left() > 20.0:
            for R in range(60, min(best_rank, 100)):
                if _time_left() < 10.0:
                    break
                cand = _iterative_refinement(n, m, p, initial_rank=R, max_iters=50)
                if cand is not None:
                    U, V, W = cand
                    if _verify_decomp(U, V, W, n, m, p, 1e-9):
                        if U.shape[1] < best_rank:
                            best_rank = U.shape[1]
                            best_U, best_V, best_W = U, V, W
                            break
    except Exception:
        pass
    
    # Local column-removal refinement
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=10, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Final safety check
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _schoolbook_decomp(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
