import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

def _trivial_decomposition(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol

def _loss_and_grad(x, n, m, p, r):
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    W = x[n*m*r + m*p*r :].reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    loss = np.sum(err ** 2)

    dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
    dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
    dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2

    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _newton_polish(U, V, W, n, m, p, r, max_iter=30, tol=1e-12):
    """High-precision Newton polish for exact decomposition."""
    for _ in range(max_iter):
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T - 1.0
        if np.max(np.abs(err)) < tol:
            break
        
        dU = np.einsum('ijk,jkc,kic->ijc', err, V, W)
        dV = np.einsum('ijk,ijc,kic->jkc', err, U, W)
        dW = np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-18:
            break
            
        # Adaptive step size with backtracking
        step = 0.5
        best_loss = np.sum(err**2)
        for _ in range(20):
            U_new = U - step * dU
            V_new = V - step * dV
            W_new = W - step * dW
            T_new = np.einsum('ijc,jkc,kic->ijk', U_new, V_new, W_new)
            err_new = T_new - 1.0
            new_loss = np.sum(err_new**2)
            if new_loss < best_loss * (1 - 1e-4):
                U, V, W = U_new, V_new, W_new
                best_loss = new_loss
                break
            step *= 0.5
        else:
            break
            
    return U, V, W

def _structured_init(n, m, p, r, seed=0):
    """
    Initialize with a structured pattern that may help convergence.
    Uses a block-diagonal-like structure based on the matrix dimensions.
    """
    rng = np.random.RandomState(seed)
    
    U0 = rng.randn(n * m, r)
    V0 = rng.randn(m * p, r)
    W0 = rng.randn(p * n, r)
    
    # Scale based on rank to help convergence
    scale = 1.0 / np.sqrt(max(r, 1))
    U0 *= scale
    V0 *= scale
    W0 *= scale
    
    # Add a small structured component to break symmetry
    for i in range(min(r, n*m)):
        U0[i % (n*m), i] += 0.1
    for i in range(min(r, m*p)):
        V0[i % (m*p), i] += 0.1
    for i in range(min(r, p*n)):
        W0[i % (p*n), i] += 0.1
    
    return U0, V0, W0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank

    def try_rank(r, max_time=12.0):
        if time.time() > deadline - 18:
            return None
        
        # Try multiple initializations with different strategies
        for seed in range(3):
            if time.time() > deadline - 15:
                break
                
            U0, V0, W0 = _structured_init(n, m, p, r, seed)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use L-BFGS-B with tighter tolerances
                result = opt.minimize(
                    lambda x: _loss_and_grad(x, n, m, p, r),
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                
                if time.time() > deadline - 10:
                    continue
                    
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # High-precision polish
                U_pol, V_pol, W_pol = _newton_polish(U_opt, V_opt, W_opt, n, m, p, r)
                
                if _verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-10):
                    return U_pol, V_pol, W_pol
                    
            except Exception:
                pass
                
        return None

    # Try ranks from trivial_rank - 1 down, but limit the search
    # Focus on a few promising ranks to stay within time budget
    ranks_to_try = list(range(trivial_rank - 1, max(trivial_rank - 8, 55), -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 25:
            break
            
        res = try_rank(r)
        if res is not None:
            U, V, W = res
            best = {'U': U, 'V': V, 'W': W}
            best_rank = r
            break

    if best_rank < trivial_rank:
        return {'U': best['U'].tolist(), 'V': best['V'].tolist(), 'W': best['W'].tolist()}
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
