"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

N, M, P = 3, 4, 6
DEADLINE = 145.0  # safety margin before 150s kill

def verify_fast(U, V, W, rank):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)

def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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
    return {"U": U, "V": V, "W": W}

def round_to_half(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(np.array(U) * 2) / 2
    V_r = np.round(np.array(V) * 2) / 2
    W_r = np.round(np.array(W) * 2) / 2
    return U_r, V_r, W_r

def try_l_bfgs(target_rank, time_budget, seed=42):
    """Try to find a decomposition of rank target_rank using L-BFGS-B."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(seed)

    # Initialize
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5

    # Flatten
    def pack(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unpack(x):
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        U = x[:size_U].reshape(n * m, target_rank)
        V = x[size_U:size_U + size_V].reshape(m * p, target_rank)
        W = x[size_U + size_V:].reshape(p * n, target_rank)
        return U, V, W

    def objective_and_grad(x):
        U, V, W = unpack(x)
        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        res = T - 1.0
        val = np.sum(res * res)
        
        # Gradient
        grad_U_3d = np.einsum('ijk,jkc,kic->ijc', res, Vr, Wr)
        grad_V_3d = np.einsum('ijk,ijc,kic->jkc', res, Ur, Wr)
        grad_W_3d = np.einsum('ijk,ijc,jkc->kic', res, Ur, Vr)
        
        grad = np.concatenate([grad_U_3d.flatten(), grad_V_3d.flatten(), grad_W_3d.flatten()])
        return val, grad

    x0 = pack(U, V, W)
    
    # Run optimization
    try:
        res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e-6:
            U_opt, V_opt, W_opt = unpack(res.x)
            
            # Try rounding to half-integers
            U_r, V_r, W_r = round_to_half(U_opt, V_opt, W_opt)
            if verify_fast(U_r, V_r, W_r, target_rank):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # If rounding fails, check if unrounded is close enough (unlikely to be exact)
            # But the task requires integer/half-integer. 
            # So we must find a solution that is exactly representable.
            # If L-BFGS finds a solution near half-integers, rounding works.
            # If not, we might need to try again.
            
    except Exception as e:
        pass

    return None

def try_search(target_rank, time_budget, seed=42):
    """Try multiple initializations with L-BFGS."""
    start = time.time()
    
    # Number of attempts depends on time budget
    max_attempts = 5
    per_attempt_budget = time_budget / max_attempts
    
    for i in range(max_attempts):
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break
        
        remaining = time_budget - (time.time() - start)
        if remaining < 1:
            break
            
        result = try_l_bfgs(target_rank, min(remaining, 10), seed=seed + i)
        if result is not None:
            return result
            
    return None

def construct():
    start_time = time.time()
    best = schoolbook()

    # Try ranks lower than schoolbook
    # Schoolbook is 72.
    # Known good ranks for similar problems suggest 66-70 might be possible.
    # We try 71 down to 66.
    
    ranks_to_try = [71, 70, 69, 68, 67, 66]
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining < 2:
            break
            
        result = try_search(rank, remaining, seed=rank)
        if result is not None:
            best = result
            # If we find a low rank, we can stop or continue to see if we can go lower
            # Given the time, finding one is a big win.
            # We can break if we find 66, as it's a likely optimal.
            if rank == 66:
                break

    # Verify best
    if not verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        best = schoolbook()

    return best
