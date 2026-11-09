"""QAOA angle optimizer for small MaxCut graphs."""
import numpy as np
import time
from itertools import product

# Deadline guard: 150s wall-clock limit, use 145s as safety margin
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _build_hamiltonians(n, edges):
    """
    Build the MaxCut Hamiltonian C = sum_{(i,j) in edges} (1 - Z_i Z_j)/2
    and the transverse field H_B = sum_i X_i.
    
    For n <= 10, we can represent the full 2^n x 2^n matrices.
    C is diagonal, H_B is a sum of Kronecker products.
    """
    dim = 1 << n
    
    # Build C as a diagonal vector of length dim
    # C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    # For a basis state |b>, Z_i = +1 if bit i is 0, -1 if bit i is 1
    # Z_i Z_j = +1 if bits i,j are same, -1 if different
    # (1 - Z_i Z_j)/2 = 0 if same, 1 if different
    # So C = number of cut edges for that state
    
    diag = np.zeros(dim, dtype=np.float64)
    for state in range(dim):
        cut = 0
        for (i, j) in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut += 1
        diag[state] = cut
    
    # Build H_B = sum_i X_i as a matrix
    # X = [[0,1],[1,0]]
    # X_i = I ⊗ ... ⊗ X ⊗ ... ⊗ I
    H_B = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        # Build X_i
        Xi = np.array([[0.0, 1.0], [1.0, 0.0]])
        # Kronecker product with I for other positions
        mat = np.array([[1.0]])
        for k in range(n - 1, -1, -1):
            if k == i:
                mat = np.kron(mat, Xi)
            else:
                mat = np.kron(mat, np.eye(2))
        H_B += mat
    
    return diag, H_B


def _expected_cut_exact(n, edges, gammas, betas):
    """
    Compute the expected cut value for given QAOA angles using exact state evolution.
    For n <= 10, we can do full state vector simulation.
    """
    dim = 1 << n
    
    # Build C diagonal and H_B
    diag, H_B = _build_hamiltonians(n, edges)
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply layers
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i * gamma * C)
        # C is diagonal, so this multiplies each component by exp(-i * gamma * diag[k])
        psi = psi * np.exp(-1j * gamma * diag)
        
        # Apply exp(-i * beta * H_B)
        # Use matrix exponential via eigendecomposition (H_B is Hermitian)
        # For small n, compute directly
        eigvals, eigvecs = np.linalg.eigh(H_B)
        exp_HB = eigvecs @ np.diag(np.exp(-1j * beta * eigvals)) @ eigvecs.conj().T
        psi = exp_HB @ psi
    
    # Expected cut = sum_k |psi_k|^2 * diag[k]
    probs = np.abs(psi) ** 2
    return np.real(np.sum(probs * diag))


def _expected_cut_fast(n, edges, gammas, betas):
    """
    Faster computation using the fact that C is diagonal.
    We only need the probabilities, and we can avoid full matrix exponentiation
    by noting that for n <= 10, the eigendecomposition of H_B is computed once.
    """
    dim = 1 << n
    
    # Build C diagonal
    diag = np.zeros(dim, dtype=np.float64)
    for state in range(dim):
        cut = 0
        for (i, j) in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut += 1
        diag[state] = cut
    
    # Build H_B and precompute eigendecomposition
    H_B = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n):
        Xi = np.array([[0.0, 1.0], [1.0, 0.0]])
        mat = np.array([[1.0]])
        for k in range(n - 1, -1, -1):
            if k == i:
                mat = np.kron(mat, Xi)
            else:
                mat = np.kron(mat, np.eye(2))
        H_B += mat
    
    eigvals, eigvecs = np.linalg.eigh(H_B)
    V = eigvecs  # unitary
    Vdag = eigvecs.conj().T
    
    # Initial state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute exp(-i*gamma*diag) for each gamma
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i * gamma * C)
        phase = np.exp(-1j * gamma * diag)
        psi = psi * phase
        
        # Apply exp(-i * beta * H_B) = V diag(exp(-i*beta*eigvals)) Vdag
        psi = V @ (np.exp(-1j * beta * eigvals) * (Vdag @ psi))
    
    probs = np.abs(psi) ** 2
    return np.real(np.sum(probs * diag))


def _exact_optimum(n, edges):
    """Compute the exact MaxCut optimum by brute force."""
    best = 0
    for state in range(1 << n):
        cut = 0
        for (i, j) in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut += 1
        if cut > best:
            best = cut
    return best


def _gradient_finite_diff(n, edges, gammas, betas, h=1e-6):
    """Compute gradient of expected cut w.r.t. (gammas, betas) via finite differences."""
    p = len(gammas)
    params = np.concatenate([gammas, betas])
    grad = np.zeros(2 * p)
    
    f0 = _expected_cut_fast(n, edges, gammas, betas)
    
    for k in range(2 * p):
        params_p = params.copy()
        params_m = params.copy()
        params_p[k] += h
        params_m[k] -= h
        
        if k < p:
            gp, bp = params_p[:p], params_p[p:]
            gm, bm = params_m[:p], params_m[p:]
        else:
            gp, bp = params_p[:p], params_p[p:]
            gm, bm = params_m[:p], params_m[p:]
        
        fp = _expected_cut_fast(n, edges, gp, bp)
        fm = _expected_cut_fast(n, edges, gm, bm)
        grad[k] = (fp - fm) / (2 * h)
    
    return grad


def _optimize_angles(n, edges, p, initial_guess):
    """
    Optimize QAOA angles using gradient descent with momentum, then polish with Nelder-Mead.
    """
    gamma_init, beta_init = initial_guess
    params = np.concatenate([gamma_init, beta_init])
    
    best_params = params.copy()
    best_val = _expected_cut_fast(n, edges, gamma_init, beta_init)
    
    # Gradient descent phase
    lr = 0.1
    momentum = 0.9
    vel = np.zeros_like(params)
    
    max_iter = 200
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        gammas = params[:p]
        betas = params[p:]
        
        grad = _gradient_finite_diff(n, edges, gammas, betas)
        
        vel = momentum * vel - lr * grad
        params = params + vel
        
        # Keep in reasonable range
        params = np.clip(params, -np.pi, np.pi)
        
        val = _expected_cut_fast(n, edges, params[:p], params[p:])
        if val > best_val:
            best_val = val
            best_params = params.copy()
    
    # Nelder-Mead polish
    from scipy.optimize import minimize
    
    def objective(x):
        g = x[:p]
        b = x[p:]
        return -_expected_cut_fast(n, edges, g, b)
    
    if _remaining() > 10.0:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < -best_val:
                best_params = result.x
                best_val = -result.fun
        except Exception:
            pass
    
    return best_params, best_val


def _graph_theoretic_init(n, edges, p):
    """
    Generate initial guess based on graph properties.
    Use heuristic: gamma ~ 1/(2*d_avg), beta ~ pi/4
    """
    if len(edges) == 0:
        return np.zeros(p), np.zeros(p)
    
    d_avg = 2 * len(edges) / n
    
    # Heuristic initial values
    gamma_init = np.full(p, 1.0 / (2 * d_avg + 0.5))
    beta_init = np.full(p, np.pi / 4.0)
    
    # For p=2, use slightly different values for the two layers
    if p >= 2:
        gamma_init[0] = 1.0 / (d_avg + 0.3)
        gamma_init[1] = 0.5 / (d_avg + 0.3)
        beta_init[0] = np.pi / 3.0
        beta_init[1] = np.pi / 6.0
    
    return gamma_init, beta_init


def _multi_start_optimize(n, edges, p):
    """Try multiple initializations and return the best."""
    best_params = None
    best_val = -np.inf
    
    # Start 1: graph-theoretic
    g0, b0 = _graph_theoretic_init(n, edges, p)
    params, val = _optimize_angles(n, edges, p, (g0, b0))
    if val > best_val:
        best_val = val
        best_params = params
    
    if _remaining() < 10.0:
        return best_params, best_val
    
    # Start 2: random
    np.random.seed(42)
    for trial in range(5):
        if _remaining() < 5.0:
            break
        g_rand = np.random.uniform(0.1, 1.5, p)
        b_rand = np.random.uniform(0.1, 1.5, p)
        params, val = _optimize_angles(n, edges, p, (g_rand, b_rand))
        if val > best_val:
            best_val = val
            best_params = params
    
    return best_params, best_val


# Cache for precomputed values
_cache = {}


def angles(n, edges, p):
    """
    Compute optimal QAOA angles for the given graph.
    
    Args:
        n: number of nodes
        edges: list of (i, j) edges
        p: number of QAOA layers
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    global _start_time
    _start_time = time.time()  # Reset for each call
    
    # Check cache
    key = (n, tuple(sorted(edges)), p)
    if key in _cache:
        return _cache[key]
    
    if _remaining() < 5.0:
        # Not enough time, return heuristic
        g0, b0 = _graph_theoretic_init(n, edges, p)
        result = np.column_stack([g0, b0])
        _cache[key] = result
        return result
    
    # Multi-start optimization
    best_params, best_val = _multi_start_optimize(n, edges, p)
    
    if best_params is None:
        g0, b0 = _graph_theoretic_init(n, edges, p)
        result = np.column_stack([g0, b0])
    else:
        result = np.column_stack([best_params[:p], best_params[p:]])
    
    # Clip to valid range
    result = np.clip(result, -np.pi, np.pi)
    
    _cache[key] = result
    return result
