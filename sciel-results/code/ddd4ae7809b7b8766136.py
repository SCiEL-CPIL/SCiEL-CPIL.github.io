import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _snap_to_half_integer(U, V, W):
    """Snap values to the nearest half-integer."""
    U_snapped = np.round(U * 2) / 2
    V_snapped = np.round(V * 2) / 2
    W_snapped = np.round(W * 2) / 2
    return U_snapped, V_snapped, W_snapped

def _compute_loss_and_grads(U, V, W, n, m, p):
    """Compute loss and gradients for tensor decomposition."""
    rank = U.shape[1]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    
    E = T - 1.0
    loss = np.sum(E * E)
    
    grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', E, V_tensor, W_tensor)
    grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', E, U_tensor, W_tensor)
    grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', E, U_tensor, V_tensor)
    
    return loss, grad_U.reshape(-1, rank), grad_V.reshape(-1, rank), grad_W.reshape(-1, rank)

def _adam_update(params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer update."""
    for i, (param, grad) in enumerate(zip(params, grads)):
        if i not in state:
            state[i] = {'m': np.zeros_like(param), 'v': np.zeros_like(param), 't': 0}
        
        state[i]['t'] += 1
        t = state[i]['t']
        
        state[i]['m'] = beta1 * state[i]['m'] + (1 - beta1) * grad
        state[i]['v'] = beta2 * state[i]['v'] + (1 - beta2) * (grad * grad)
        
        m_hat = state[i]['m'] / (1 - beta1**t)
        v_hat = state[i]['v'] / (1 - beta2**t)
        
        params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return params

def _gradient_descent_adam(U, V, W, n, m, p, max_iter=500, lr=0.01, tol=1e-12, deadline=None):
    """Perform Adam gradient descent to minimize the decomposition error."""
    rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = np.inf
    
    state = {}
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        loss, grad_U, grad_V, grad_W = _compute_loss_and_grads(U, V, W, n, m, p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        # Adaptive learning rate
        grad_norm = max(np.linalg.norm(grad_U), np.linalg.norm(grad_V), np.linalg.norm(grad_W))
        if grad_norm > 0:
            effective_lr = lr * 50.0 / (grad_norm + 1e-8)
            effective_lr = min(effective_lr, 0.1)
        else:
            break
        
        params = [U, V, W]
        grads = [grad_U, grad_V, grad_W]
        _adam_update(params, grads, state, effective_lr)
        U, V, W = params
    
    return best_U, best_V, best_W, best_loss

def _structured_init_block(n, m, p, target_rank):
    """Initialize with block structure."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Try to fill with structured patterns
    # For each target rank, try different block sizes
    for r in range(target_rank):
        # Random but structured assignment
        i = r % n
        j = (r // n) % m
        k = (r // (n * m)) % p
        
        U[i * m + j, r] = 1.0
        V[j * p + k, r] = 1.0
        W[k * n + i, r] = 1.0
        
        # Add some randomness
        if r % 3 == 0:
            U[i * m + j, r] += 0.5
        if r % 3 == 1:
            V[j * p + k, r] += 0.5
        if r % 3 == 2:
            W[k * n + i, r] += 0.5
    
    return U, V, W

def _structured_init_kronecker(n, m, p, target_rank):
    """Initialize with Kronecker-like structure."""
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Use a pattern based on modular arithmetic
    for r in range(target_rank):
        # Create a pattern
        a = r % n
        b = (r // n) % m
        c = (r // (n * m)) % p
        
        U[a * m + b, r] = 1.0
        V[b * p + c, r] = 1.0
        W[c * n + a, r] = 1.0
        
        # Add perturbations
        if r % 5 == 0:
            U[a * m + b, r] = 0.5
        if r % 5 == 1:
            V[b * p + c, r] = -0.5
        if r % 5 == 2:
            W[c * n + a, r] = 1.5
    
    return U, V, W

def _try_random_search(n, m, p, target_rank, num_trials, deadline, init_type='random'):
    """Try random initializations and gradient descent to find valid decompositions."""
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        if init_type == 'random':
            scale = np.random.uniform(0.1, 1.0)
            U = scale * np.random.randn(n * m, target_rank)
            V = scale * np.random.randn(m * p, target_rank)
            W = scale * np.random.randn(p * n, target_rank)
        elif init_type == 'block':
            U, V, W = _structured_init_block(n, m, p, target_rank)
            # Add small random perturbations
            U += 0.1 * np.random.randn(n * m, target_rank)
            V += 0.1 * np.random.randn(m * p, target_rank)
            W += 0.1 * np.random.randn(p * n, target_rank)
        elif init_type == 'kronecker':
            U, V, W = _structured_init_kronecker(n, m, p, target_rank)
            U += 0.1 * np.random.randn(n * m, target_rank)
            V += 0.1 * np.random.randn(m * p, target_rank)
            W += 0.1 * np.random.randn(p * n, target_rank)
        
        # Run Adam gradient descent
        U_opt, V_opt, W_opt, final_loss = _gradient_descent_adam(
            U, V, W, n, m, p, max_iter=300, lr=0.02, tol=1e-12, deadline=deadline
        )
        
        # Check if the continuous solution is close to an exact one
        if final_loss < 1e-8:
            # Try snapping to half-integers
            U_snap, V_snap, W_snap = _snap_to_half_integer(U_opt, V_opt, W_opt)
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
        
        # Also check the continuous solution directly
        if final_loss < 1e-10:
            return U_opt, V_opt, W_opt

    return None

def _try_systematic_search(n, m, p, deadline, best_rank):
    """Try various structured patterns that might give lower rank."""
    # Try ranks from best_rank down, with multiple initialization strategies
    candidate_ranks = list(range(best_rank - 1, max(20, best_rank - 30), -1))
    
    init_types = ['random', 'block', 'kronecker']
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 20:
            break
        if target_rank >= best_rank:
            continue
        
        for init_type in init_types:
            if time.time() > deadline - 20:
                break
            
            result = _try_random_search(n, m, p, target_rank, 3, deadline, init_type)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    # Try to find better decompositions using systematic search
    result = _try_systematic_search(n, m, p, deadline, best_rank)
    if result is not None:
        U_try, V_try, W_try = result
        if U_try.shape[1] < best_rank:
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                best_rank = U_try.shape[1]
    
    # If time remains, try more random searches with different seeds
    if time.time() < deadline - 25:
        for seed in [42, 123, 456, 789, 111, 222, 333, 444, 555, 666]:
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            result = _try_random_search(n, m, p, best_rank - 1, 2, deadline, 'random')
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = U_try.shape[1]
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
