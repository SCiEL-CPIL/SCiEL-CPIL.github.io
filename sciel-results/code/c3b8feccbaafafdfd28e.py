"""
Rank-minimal exact tensor decomposition for 3x5 x 5x6 matrix multiplication.

Strategy:
  1. Start from a known valid decomposition (block Strassen or schoolbook).
  2. Use numerical optimization (L-BFGS-B) to minimize the squared error of the
     tensor decomposition for a given rank r.
  3. After optimization, snap the factor matrices to the nearest half-integer grid.
  4. Verify exactness of the snapped decomposition.
  5. Iterate downwards in rank until a valid exact decomposition is found or time expires.

The output follows the trace convention: W[k*n + i, r].
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nmp = n * m * p
    Uf = U.reshape(n, m, -1)
    Vf = V.reshape(m, p, -1)
    Wf = W.reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Uf[i, j, :], Vf[j, k, :] * Wf[k, i, :])
                if abs(s - 1.0) > 1e-7:
                    return False
    return True


def _snapped_verify(U, V, W, n, m, p):
    """Verify after snapping to half-integers."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return _verify(U_s, V_s, W_s, n, m, p), U_s, V_s, W_s


def _build_initial(n, m, p, rank):
    """
    Build an initial guess for rank r decomposition.
    Start from schoolbook and truncate/zero-pad to rank r.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill first min(rank, n*m*p) columns with schoolbook pattern
    fill = min(rank, n * m * p)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= fill:
                    break
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
            if col >= fill:
                break
        if col >= fill:
            break
    
    # Add small random noise to break symmetry
    rng = np.random.default_rng(42)
    U += rng.standard_normal((n * m, rank)) * 0.1
    V += rng.standard_normal((m * p, rank)) * 0.1
    W += rng.standard_normal((p * n, rank)) * 0.1
    
    return U, V, W


def _tensor_error_and_grad(U, V, W, n, m, p):
    """
    Compute the squared Frobenius error of the tensor decomposition and its gradient.
    Error = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    """
    rank = U.shape[1]
    Uf = U.reshape(n, m, rank)
    Vf = V.reshape(m, p, rank)
    Wf = W.reshape(p, n, rank)
    
    error = 0.0
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    # For each (i,j,k), compute the inner product and accumulate gradients
    # T[i,j,k] = sum_r Uf[i,j,r] * Vf[j,k,r] * Wf[k,i,r]
    # d/dUf[i,j,r] error = 2*(T[i,j,k]-1) * Vf[j,k,r] * Wf[k,i,r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute T[i,j,k]
                t_val = np.dot(Uf[i, j, :], Vf[j, k, :] * Wf[k, i, :])
                diff = t_val - 1.0
                error += diff * diff
                
                # Gradient contributions
                factor = 2.0 * diff
                # dU
                grad_U[i * m + j, :] += factor * (Vf[j, k, :] * Wf[k, i, :])
                # dV
                grad_V[j * p + k, :] += factor * (Uf[i, j, :] * Wf[k, i, :])
                # dW
                grad_W[k * n + i, :] += factor * (Uf[i, j, :] * Vf[j, k, :])
    
    return error, np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])


def _optimize_rank(n, m, p, rank, initial_U, initial_V, initial_W, max_iter=500, time_limit=20):
    """
    Run L-BFGS-B optimization to minimize tensor error for given rank.
    """
    start_time = time.time()
    
    x0 = np.concatenate([initial_U.ravel(), initial_V.ravel(), initial_W.ravel()])
    
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError("Optimization timeout")
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank :].reshape(p*n, rank)
        return _tensor_error_and_grad(U, V, W, n, m, p)
    
    try:
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 5}
        )
    except TimeoutError:
        return None
    except Exception:
        return None
    
    if time.time() - start_time > time_limit:
        return None
    
    x_opt = result.x
    U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
    V_opt = x_opt[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W_opt = x_opt[n*m*rank + m*p*rank :].reshape(p*n, rank)
    
    return U_opt, V_opt, W_opt


def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    """
    U = np.array([
        [ 1,  0,  1,  0,  1, -1,  0],
        [ 0,  0,  0,  0,  1,  0,  1],
        [ 0,  1,  0,  0,  0,  1,  0],
        [-1,  1,  0,  1,  0,  0, -1]
    ], dtype=float)
    V = np.array([
        [ 0,  1,  1,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 1,  0,  0, -1,  0,  0,  1],
        [ 1,  0, -1,  1,  1,  0,  1]
    ], dtype=float)
    W = np.array([
        [ 1,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0, 0,  0],
        [ 0,  1,  1,  0,  0,  0,  0],
        [ 1,  0,  0,  1,  0,  0,  1]
    ], dtype=float)
    return U, V, W


def _build_block_decomposition(n, m, p):
    """
    Build a decomposition using block Strassen + schoolbook fallback.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    covered = set()
    blocks = []
    for i0 in range(0, n - 1, 2):
        for j0 in range(0, m - 1, 2):
            for k0 in range(0, p - 1, 2):
                blocks.append((i0, j0, k0))
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0 + di, j0 + dj, k0 + dk))
    
    for (i0, j0, k0) in blocks:
        for r in range(7):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            for di in range(2):
                for dj in range(2):
                    u_col[(i0 + di) * m + (j0 + dj)] = strassen_U[di * 2 + dj, r]
            for dj in range(2):
                for dk in range(2):
                    v_col[(j0 + dj) * p + (k0 + dk)] = strassen_V[dj * 2 + dk, r]
            for dk in range(2):
                for di in range(2):
                    w_col[(k0 + dk) * n + (i0 + di)] = strassen_W[dk * 2 + di, r]
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_col = np.zeros(n * m)
                    v_col = np.zeros(m * p)
                    w_col = np.zeros(p * n)
                    u_col[i * m + j] = 1.0
                    v_col[j * p + k] = 1.0
                    w_col[k * n + i] = 1.0
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    if not U_cols:
        return None
    
    U_mat = np.column_stack(U_cols)
    V_mat = np.column_stack(V_cols)
    W_mat = np.column_stack(W_cols)
    
    if _verify(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Step 1: Get a baseline valid decomposition
    base = _build_block_decomposition(n, m, p)
    if base is None:
        base = _schoolbook(n, m, p)
    
    best_result = base
    best_rank = base["U"].shape[1]
    
    # Step 2: Try to reduce rank via numerical optimization
    # Start from best_rank - 1 and go down
    for rank in range(best_rank - 1, 40, -1):
        if time.time() > deadline:
            break
        
        # Build initial guess from schoolbook truncated to this rank
        U_init, V_init, W_init = _build_initial(n, m, p, rank)
        
        # Run optimization
        opt_result = _optimize_rank(n, m, p, rank, U_init, V_init, W_init,
                                     max_iter=300, time_limit=15)
        
        if opt_result is None:
            continue
        
        U_opt, V_opt, W_opt = opt_result
        
        # Snap to half-integers and verify
        valid, U_s, V_s, W_s = _snapped_verify(U_opt, V_opt, W_opt, n, m, p)
        if valid:
            best_result = {"U": U_s, "V": V_s, "W": W_s}
            best_rank = rank
            # Continue trying to go lower
            continue
        else:
            # Check if the un-snapped version is close enough (within 1e-6)
            if _verify(U_opt, V_opt, W_opt, n, m, p):
                # It's valid but not on the half-integer grid; try harder snapping
                # or accept if entries are very close to half-integers
                U_close = np.all(np.abs(U_opt * 2 - np.round(U_opt * 2)) < 1e-4)
                V_close = np.all(np.abs(V_opt * 2 - np.round(V_opt * 2)) < 1e-4)
                W_close = np.all(np.abs(W_opt * 2 - np.round(W_opt * 2)) < 1e-4)
                if U_close and V_close and W_close:
                    U_s = np.round(U_opt * 2) / 2
                    V_s = np.round(V_opt * 2) / 2
                    W_s = np.round(W_opt * 2) / 2
                    if _verify(U_s, V_s, W_s, n, m, p):
                        best_result = {"U": U_s, "V": V_s, "W": W_s}
                        best_rank = rank
                        continue
            # If not valid, break to save time
            break
    
    return {
        "U": best_result["U"].tolist(),
        "V": best_result["V"].tolist(),
        "W": best_result["W"].tolist()
    }
