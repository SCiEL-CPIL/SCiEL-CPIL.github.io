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

def _newton_polish(U, V, W, n, m, p, r, max_iter=20, tol=1e-10):
    """Newton's method to polish the solution for high precision."""
    for _ in range(max_iter):
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T - 1.0
        if np.max(np.abs(err)) < tol:
            break
        
        # Compute Jacobian of T with respect to U, V, W
        # dT/dU_ijc = V_jk * W_ki
        # dT/dV_jkc = U_ij * W_ki
        # dT/dW_kic = U_ij * V_jk
        
        # We want to solve: J * delta = -err
        # J is a 3D tensor (n*m*p, n*m*r + m*p*r + p*n*r)
        # This is too large to form explicitly. We use Gauss-Newton or similar.
        # For simplicity, let's try a few steps of gradient descent with adaptive step size
        # or use the fact that the problem is bilinear.
        
        # Actually, let's just do a few steps of L-BFGS-B with tighter tolerance
        # inside the polish function if needed, but that's redundant.
        # Let's try a simple Newton step for the residuals.
        
        # Since the system is nonlinear, a full Newton step is complex.
        # Instead, we can use the gradient to take a step.
        dU = np.einsum('ijk,jkc,kic->ijc', err, V, W)
        dV = np.einsum('ijk,ijc,kic->jkc', err, U, W)
        dW = np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-15:
            break
            
        # Adaptive step size
        step = 0.1
        for _ in range(10):
            U_new = U - step * dU
            V_new = V - step * dV
            W_new = W - step * dW
            T_new = np.einsum('ijc,jkc,kic->ijk', U_new, V_new, W_new)
            err_new = T_new - 1.0
            if np.sum(err_new**2) < np.sum(err**2):
                U, V, W = U_new, V_new, W_new
                break
            step *= 0.5
        else:
            break
            
    return U, V, W

def _strassen_block_init(n, m, p, r, seed=0):
    """
    Initialize using a block structure inspired by Strassen.
    For 3x4x6, we can try to decompose it into smaller blocks.
    """
    rng = np.random.RandomState(seed)
    
    # Simple random initialization with some structure
    U0 = rng.randn(n * m, r)
    V0 = rng.randn(m * p, r)
    W0 = rng.randn(p * n, r)
    
    # Scale to help convergence
    scale = 1.0 / np.sqrt(r)
    U0 *= scale
    V0 *= scale
    W0 *= scale
    
    return U0, V0, W0

def construct():
    start_time = time.time()
    deadline = 140.0
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank

    def try_rank(r, max_time=15.0):
        if time.time() > deadline - 15:
            return None
        
        # Try multiple initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            U0, V0, W0 = _strassen_block_init(n, m, p, r, seed)
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                result = opt.minimize(
                    lambda x: _loss_and_grad(x, n, m, p, r),
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if time.time() > deadline - 5:
                    continue
                    
                U_opt = result.x[:n*m*r].reshape(n*m, r)
                V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Polish the solution
                U_pol, V_pol, W_pol = _newton_polish(U_opt, V_opt, W_opt, n, m, p, r)
                
                if _verify(U_pol, V_pol, W_pol, n, m, p, tol=1e-8):
                    return U_pol, V_pol, W_pol
                    
            except Exception:
                pass
                
        return None

    # Try ranks from trivial_rank - 1 down to 60
    # But we know that 72 is the trivial rank, and the minimal rank is likely lower.
    # We try a few ranks below 72.
    for r in range(trivial_rank - 1, max(trivial_rank - 10, 60), -1):
        if time.time() > deadline - 20:
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
