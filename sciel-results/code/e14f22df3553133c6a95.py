"""Optimized twelve-stage product formula for Heisenberg-chain time evolution."""
import time
import numpy as np
from scipy.linalg import expm

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_DEADLINE = 150.0
_START_TIME = time.time()

def _remaining_time():
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)  # 10s safety margin

def _build_matrices():
    """Build A (odd bonds) and B (even bonds) for a 6-site open XXZ Heisenberg chain.
    
    Anisotropy gamma = 0.8, field h = 0.3.
    The Hamiltonian is H = sum_{j=1}^{5} [gamma/2 (S_j^z S_{j+1}^z) + (S_j^x S_{j+1}^x + S_j^y S_{j+1}^y)/2] + h * sum_j S_j^z
    
    Split into odd bonds (1-2, 3-4, 5-6) -> A, and even bonds (2-3, 4-5) -> B.
    Each half includes half of the uniform field.
    
    In the computational basis of 6 spin-1/2 sites (2^6 = 64 dim), we use:
    S^x = 1/2 [[0,1],[1,0]], S^y = 1/2 [[0,-i],[i,0]], S^z = 1/2 [[1,0],[0,-1]]
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    # Single-spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def tensor_op(op, site):
        """Apply op on 'site' (0-indexed) and identity elsewhere."""
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        jx = tensor_op(sx, j) @ tensor_op(sx, j+1)
        jy = tensor_op(sy, j) @ tensor_op(sy, j+1)
        jz = tensor_op(sz, j) @ tensor_op(sz, j+1)
        A += 0.5 * (jx + jy) + 0.5 * gamma * jz
    # Half field on all sites
    for j in range(N):
        A += 0.5 * h * tensor_op(sz, j)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        jx = tensor_op(sx, j) @ tensor_op(sx, j+1)
        jy = tensor_op(sy, j) @ tensor_op(sy, j+1)
        jz = tensor_op(sz, j) @ tensor_op(sz, j+1)
        B += 0.5 * (jx + jy) + 0.5 * gamma * jz
    # Half field on all sites
    for j in range(N):
        B += 0.5 * h * tensor_op(sz, j)
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error at given times."""
    dim = A.shape[0]
    U_exact_base = expm(-1j * (A + B))
    total_err = 0.0
    for t in times:
        U_exact = expm(np.logm(U_exact_base) * t) if False else expm(-1j * t * (A + B))
        U_approx = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U_approx = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U_approx
        # Spectral norm of (U_approx - U_exact)
        err = np.linalg.norm(U_approx - U_exact, ord=2)
        total_err += err
    return total_err / len(times)

def coefficients(stages):
    """Return optimal (a, b) coefficient lists for the product formula."""
    t_start = time.time()
    
    # Build matrices
    A, B = _build_matrices()
    dim = A.shape[0]
    
    times = [0.25, 0.5, 1.0]
    
    # Precompute expm for various coefficients to speed up evaluation
    # Strategy: use a multi-start local optimization with Nelder-Mead or similar
    # But given the 60s CPU budget, we need to be efficient.
    
    # First, try a few structured initial guesses
    # The seed uses Strang splitting: a_k = 1/(2*substeps), b alternates 1/substeps, 0
    
    # Let's try to optimize using a simple gradient-free method with limited iterations
    
    def _eval(a_list, b_list):
        """Evaluate the error for given coefficients."""
        # Check constraints
        if abs(sum(a_list) - 1.0) > 1e-8 or abs(sum(b_list) - 1.0) > 1e-8:
            return 1e10
        if any(abs(x) > 50 for x in a_list + b_list):
            return 1e10
        if any(not np.isfinite(x) for x in a_list + b_list):
            return 1e10
        return _compute_error(a_list, b_list, A, B, times)
    
    # Precompute exponentials for efficiency
    # For a given set of (a_k, b_k), we need expm(-i * a_k * t * A) and expm(-i * b_k * t * B)
    # This is expensive. Let's use a smarter approach.
    
    # Approach: Use the fact that for small t, we can use series expansion or
    # precompute the matrix logarithms.
    
    # Actually, let's use a different strategy:
    # 1. Start with the Strang splitting baseline
    # 2. Try a few structured perturbations
    # 3. Use local optimization with a limited number of function evaluations
    
    # Baseline: Strang splitting with 6 substeps (stages/2 = 6)
    substeps = stages // 2
    a_base = [0.5 / substeps] * (2 * substeps)
    b_base = []
    for _ in range(substeps):
        b_base += [1.0 / substeps, 0.0]
    while len(a_base) < stages:
        a_base.append(0.0)
        b_base.append(0.0)
    
    best_a = list(a_base)
    best_b = list(b_base)
    best_err = _eval(best_a, best_b)
    
    # Try symmetric structures: a_k = a_{stages+1-k}, b_k = b_{stages+1-k}
    # This reduces the effective number of free parameters.
    
    # For a symmetric 12-stage formula:
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # with sum(a) = 2*(a1+a2+a3+a4+a5+a6) = 1, so a1+...+a6 = 0.5
    # and sum(b) = 2*(b1+...+b6) = 1, so b1+...+b6 = 0.5
    
    # Let's try a few heuristic structures:
    
    # Structure 1: Equal weights (Strang)
    # Already have this.
    
    # Structure 2: Trapezoidal rule on the coefficients
    # a_k proportional to sin^2(pi*k/(2*(substeps+1))) or similar
    
    # Structure 3: Use known optimal 4th order coefficients for Strang-type splitting
    # For a 2-stage Strang: a = [1/2, 1/2], b = [1, 0]
    # For higher order, we can use composition methods.
    
    # Let's try a simple local search with a limited budget.
    # Use a coordinate descent approach.
    
    n_evals = 0
    max_evals = 500  # Keep within budget
    
    def _try_update(a_try, b_try):
        nonlocal best_a, best_b, best_err, n_evals
        if n_evals >= max_evals:
            return
        if time.time() - t_start > 40:  # Safety margin
            return
        n_evals += 1
        err = _eval(a_try, b_try)
        if err < best_err:
            best_err = err
            best_a = list(a_try)
            best_b = list(b_try)
    
    # Try small perturbations to the baseline
    rng = np.random.default_rng(42)
    
    for iteration in range(100):
        if time.time() - t_start > 40 or n_evals >= max_evals:
            break
        
        # Perturb a and b
        a_try = list(best_a)
        b_try = list(best_b)
        
        # Random perturbation
        scale = 0.01 / (iteration + 1)
        for k in range(stages):
            a_try[k] += rng.normal(0, scale)
            b_try[k] += rng.normal(0, scale)
        
        # Enforce sum constraints
        a_sum = sum(a_try)
        b_sum = sum(b_try)
        if a_sum != 0:
            for k in range(stages):
                a_try[k] += (1.0 - a_sum) / stages
        if b_sum != 0:
            for k in range(stages):
                b_try[k] += (1.0 - b_sum) / stages
        
        _try_update(a_try, b_try)
    
    # Try structured symmetric forms
    # Symmetric 6-parameter family
    for trial in range(20):
        if time.time() - t_start > 45 or n_evals >= max_evals:
            break
        
        # Generate random symmetric coefficients
        a_half = rng.uniform(-0.1, 0.6, size=6)
        a_half_sum = sum(a_half)
        if a_half_sum > 0:
            a_half = [x / a_half_sum * 0.5 for x in a_half]
        a_sym = list(a_half) + list(reversed(a_half))
        
        b_half = rng.uniform(-0.1, 0.6, size=6)
        b_half_sum = sum(b_half)
        if b_half_sum > 0:
            b_half = [x / b_half_sum * 0.5 for x in b_half]
        b_sym = list(b_half) + list(reversed(b_half))
        
        _try_update(a_sym, b_sym)
    
    # Return the best found
    return best_a, best_b
