"""Rewritten program with vectorized QAOA simulation and multi-restart optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 140s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    # Precompute edge bit masks for speed
    edge_masks = []
    for (i, j) in edges:
        mask = (1 << i) | (1 << j)
        edge_masks.append(mask)
    
    for state in range(dim):
        cut = 0
        for mask in edge_masks:
            # Check if bits i and j are different
            # This is equivalent to (state & mask) having exactly one bit set
            # Or simply: ((state >> i) ^ (state >> j)) & 1
            # Using bit count:
            # If we use the mask, we can check parity of (state & mask)
            # If parity is 1, bits are different.
            if np.count_nonzero(np.binary_repr(state & mask, width=n)) % 2 == 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        # For each state, if bit i != bit j, add 1
        # We can do this vectorized
        # Create masks for bit i and bit j
        # Actually, simpler:
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut_sizes[state] += 1.0
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """Apply exp(-i beta sum X) to state vector using tensor operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state(n, edges, cut_sizes, angles):
    """Simulate QAOA circuit and return final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state *= np.exp(-1j * gamma * cut_sizes)
        # X layer
        state = _apply_x_layer(state, beta, n)
        
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, None, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _gradient(params, n, cut_sizes, p, eps=1e-4):
    """Compute gradient via central finite differences."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, n, cut_sizes, p)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        f_minus = _objective(params_minus, n, cut_sizes, p)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # 1. Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # 2. Compute exact MaxCut (for reference, though we maximize expected cut directly)
    # Not strictly needed for optimization, but good for validation if we wanted ratio directly.
    # Since denominator is constant, maximizing numerator is equivalent.
    
    best_angles = None
    best_score = -np.inf
    
    # 3. Optimization with multiple restarts
    # Number of restarts depends on time
    # Let's try 10 restarts. Each should be fast.
    n_restarts = 10
    
    # Initial guesses
    # 1. Standard QAOA angles
    initial_guesses = []
    # gamma ~ 0.35, beta ~ 0.25
    initial_guesses.append(np.array([0.35, 0.25] * p))
    
    # 2. Random guesses
    np.random.seed(42)
    for _ in range(n_restarts - 1):
        gammas = np.random.uniform(0, np.pi, p)
        betas = np.random.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    bounds = [(-np.pi, np.pi)] * (2 * p)
    
    for x0 in initial_guesses:
        if _time_left() < 10.0:
            break
            
        try:
            res = minimize(
                _objective,
                x0,
                args=(n, cut_sizes, p),
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_angles = res.x.reshape(p, 2)
                
        except Exception as e:
            # If optimization fails, ignore this restart
            pass

    # Fallback if no valid optimization result
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
