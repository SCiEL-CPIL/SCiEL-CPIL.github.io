"""Matrix multiplication 3x4x6 - model-and-optimize approach with structured initialization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol


def _compute_loss_and_grads(Ut, Vt, Wt, n, m, p):
    """Compute loss and gradients for the decomposition."""
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    err = T - 1.0
    loss = np.sum(err ** 2)
    
    dU = np.einsum('ijk,jkc,kic->ijc', err, Vt, Wt) * 2
    dV = np.einsum('ijk,ijc,kic->jkc', err, Ut, Wt) * 2
    dW = np.einsum('ijk,ijc,jkc->kic', err, Ut, Vt) * 2
    
    return loss, dU, dV, dW


def _structured_init(n, m, p, r, seed):
    """
    Structured initialization that leverages the factorization 4=2*2, 6=2*3.
    Uses a block structure inspired by tensor product decompositions.
    """
    np.random.seed(seed)
    
    # Base random initialization with careful scaling
    scale = 1.0 / np.sqrt(r)
    U = np.random.randn(n * m, r) * scale
    V = np.random.randn(m * p, r) * scale
    W = np.random.randn(p * n, r) * scale
    
    # Add structured components: try to align with known low-rank patterns
    # For 3x4x6, we can think of it as combining 3x2x3 and 2x2x2 blocks
    # The key insight: use a Kronecker-like structure where possible
    
    # Create some "anchor" terms that exactly match sub-decompositions
    # First, add the trivial decomposition for a subset to give gradient descent a head start
    n_anchors = min(r // 4, 12)  # Add up to 12 anchor terms
    for a in range(n_anchors):
        # Distribute anchors across the (i,j,k) space
        i = a % n
        j = (a // n) % m
        k = (a // (n * m)) % p
        
        # Add a small perturbation to avoid exact degeneracy
        U[i * m + j, a] += 1.0
        V[j * p + k, a] += 1.0
        W[k * n + i, a] += 1.0
    
    return U, V, W


def _gradient_descent(U, V, W, r, n, m, p, max_iter, lr_init, deadline, 
                      use_adaptive=True, use_normalization=True, use_momentum=False):
    """
    Perform gradient descent to find a rank-r decomposition.
    Returns (U, V, W, converged) where converged is True if loss < 1e-10.
    """
    Ut = U.reshape(n, m, r)
    Vt = V.reshape(m, p, r)
    Wt = W.reshape(p, n, r)
    
    lr = lr_init
    prev_loss = float('inf')
    momentum_U = np.zeros_like(Ut) if use_momentum else None
    momentum_V = np.zeros_like(Vt) if use_momentum else None
    momentum_W = np.zeros_like(Wt) if use_momentum else None
    mu = 0.9  # momentum coefficient
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        loss, dU, dV, dW = _compute_loss_and_grads(Ut, Vt, Wt, n, m, p)
        
        if loss < 1e-10:
            return Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r), True
        
        # Adaptive learning rate
        if use_adaptive:
            if loss > prev_loss:
                lr = max(lr * 0.5, 1e-5)
            else:
                lr = min(lr * 1.02, lr_init)
        
        prev_loss = loss
        
        # Optional gradient normalization
        if use_normalization:
            gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if gnorm > 1e-10:
                dU /= gnorm
                dV /= gnorm
                dW /= gnorm
        
        # Optional momentum
        if use_momentum:
            momentum_U = mu * momentum_U + (1 - mu) * dU
            momentum_V = mu * momentum_V + (1 - mu) * dV
            momentum_W = mu * momentum_W + (1 - mu) * dW
            dU, dV, dW = momentum_U, momentum_V, momentum_W
        
        Ut -= lr * dU
        Vt -= lr * dV
        Wt -= lr * dW
    
    # Final check
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    final_loss = np.sum((T - 1.0) ** 2)
    converged = final_loss < 1e-10
    
    return Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r), converged


def _newton_polish(U, V, W, r, n, m, p, max_iter=50, deadline=None):
    """
    Newton's method polish for high-precision convergence.
    Solves the linearized system for corrections to U, V, W.
    """
    Ut = U.reshape(n, m, r).copy()
    Vt = V.reshape(m, p, r).copy()
    Wt = W.reshape(p, n, r).copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        err = T - 1.0
        loss = np.sum(err ** 2)
        
        if loss < 1e-14:
            break
        
        # Compute the Hessian-like structure for Newton's method
        # We solve for deltas dUt, dVt, dWt such that:
        # sum_c dUt[i,j,c]*Vt[j,k,c]*Wt[k,i,c] + Ut[i,j,c]*dVt[j,k,c]*Wt[k,i,c] + 
        # Ut[i,j,c]*Vt[j,k,c]*dWt[k,i,c] = -err[i,j,k]
        
        # This is a linear system. For small r, we can solve it directly.
        # Let's build the system in a flattened form.
        
        # Actually, for the coupled structure, Newton's method is complex.
        # Instead, use a Gauss-Newton approximation: only use the first-order terms.
        # This is equivalent to doing one step of gradient descent with the 
        # Gauss-Newton Hessian, which is cheaper than full Newton.
        
        # For now, use a damped Newton step via solving a smaller system
        # per "fiber" of the tensor.
        
        # Alternative: use the fact that for fixed V and W, the problem in U is linear,
        # and vice versa. Do coordinate descent with exact solves.
        
        # Solve for U given V, W:
        # For each (i,j): sum_c Ut[i,j,c] * (Vt[j,:,c] * Wt[:,i,c]).sum() ... 
        # This is more complex due to the coupling.
        
        # Let's use a simpler approach: take a few small gradient steps with 
        # very tight tolerance.
        lr = 0.001
        dU = np.einsum('ijk,jkc,kic->ijc', err, Vt, Wt)
        dV = np.einsum('ijk,ijc,kic->jkc', err, Ut, Wt)
        dW = np.einsum('ijk,ijc,jkc->kic', err, Ut, Vt)
        
        # Normalize
        gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if gnorm > 1e-10:
            dU /= gnorm
            dV /= gnorm
            dW /= gnorm
        
        Ut -= lr * dU
        Vt -= lr * dV
        Wt -= lr * dW
    
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    if np.max(np.abs(T - 1.0)) < 1e-8:
        return Ut.reshape(n*m, r), Vt.reshape(m*p, r), Wt.reshape(p*n, r)
    return None


def _try_rank_with_variants(r, n, m, p, deadline, n_restarts=4):
    """
    Try to find a rank-r decomposition using multiple initialization strategies
    and optimization variants.
    """
    best_result = None
    best_loss = float('inf')
    
    strategies = [
        # (init_func, use_adaptive, use_normalization, use_momentum, lr_init)
        (lambda r, seed: _structured_init(n, m, p, r, seed), True, True, False, 0.02),
        (lambda r, seed: _structured_init(n, m, p, r, seed), True, False, False, 0.01),
        (lambda r, seed: _structured_init(n, m, p, r, seed), False, True, True, 0.005),
        (lambda r, seed: _structured_init(n, m, p, r, seed), True, True, True, 0.01),
    ]
    
    for restart in range(n_restarts):
        if time.time() > deadline - 2:
            break
        
        strategy_idx = restart % len(strategies)
        init_func, use_adaptive, use_norm, use_mom, lr_init = strategies[strategy_idx]
        
        seed = 42 + restart * 1000 + r
        U, V, W = init_func(r, seed)
        
        max_iter = 3000 if r > 60 else 5000
        U, V, W, converged = _gradient_descent(
            U, V, W, r, n, m, p, max_iter, lr_init, deadline,
            use_adaptive=use_adaptive, use_normalization=use_norm, use_momentum=use_mom
        )
        
        if converged:
            # Polish for higher precision
            polished = _newton_polish(U, V, W, r, n, m, p, max_iter=30, deadline=deadline)
            if polished is not None:
                U, V, W = polished
            if _verify(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
        
        # Track best non-converged result
        T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
        loss = np.sum((T - 1.0) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_result = (U, V, W)
    
    if best_result is not None and best_loss < 1e-6:
        U, V, W = best_result
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None


def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    # The trivial rank is 72. Try to find lower rank decompositions.
    # Known lower bounds suggest rank >= 54 for 3x4x6.
    # Try ranks from just below 72 down to 54.
    
    # First, try ranks close to 72 where convergence is more likely
    ranks_to_try = list(range(71, 53, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_with_variants(r, n, m, p, deadline, n_restarts=3)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break
    
    if best_rank < n * m * p:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
