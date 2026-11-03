import numpy as np
import time
import warnings
from scipy.optimize import minimize

N = 50
_TIME_LIMIT = 145.0  # safety margin below 150s kill
_start_time = time.time()
warnings.filterwarnings('ignore')

# Precompute the autoconvolution kernel for step functions
# The autoconvolution of two step functions on [-1/4, 1/4] with N equal intervals
# can be computed exactly using the overlap of intervals.

def _precompute_conv_kernel():
    """
    Precompute the kernel K such that autoconv = h_outer @ K_flat
    where autoconv is evaluated at a fine grid of points.
    
    For step functions, the autoconvolution at point t is:
    (f * f)(t) = integral f(s) f(t-s) ds
    
    If f is constant h_i on [x_i, x_{i+1}], then:
    (f*f)(t) = sum_{i,j} h_i h_j * overlap_length([x_i, x_{i+1}], [t-x_{j+1}, t-x_j])
    
    We evaluate this at a fine grid of t values to approximate the supremum.
    """
    dw = 1.0 / N
    x = np.linspace(-0.25, 0.25, N + 1)
    
    # Use a fine grid for the autoconvolution evaluation
    n_pts = 8 * N
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0
    
    # K[i, j, k] = overlap of interval [x_i, x_{i+1}] with [t_k - x_{j+1}, t_k - x_j]
    # = max(0, min(x_{i+1}, t_k - x_j) - max(x_i, t_k - x_{j+1}))
    
    # Vectorized computation
    i_idx = np.arange(N)[:, None, None]  # (N, 1, 1)
    j_idx = np.arange(N)[None, :, None]  # (1, N, 1)
    k_idx = np.arange(n_pts)[None, None, :]  # (1, 1, n_pts)
    
    x_i = x[i_idx]  # (N, 1, 1)
    x_i1 = x[i_idx + 1]  # (N, 1, 1)
    x_j = x[j_idx]  # (1, N, 1)
    x_j1 = x[j_idx + 1]  # (1, N, 1)
    t_k = midpoints[k_idx]  # (1, 1, n_pts)
    
    # Compute lo and hi of the overlap
    lo = np.maximum(x_i, t_k - x_j1)  # (N, N, n_pts)
    hi = np.minimum(x_i1, t_k - x_j)  # (N, N, n_pts)
    
    K = np.maximum(0.0, hi - lo)  # (N, N, n_pts)
    
    # Flatten for matrix multiplication: h_outer (N*N,) @ K_flat (N*N, n_pts)
    K_flat = K.reshape(N * N, -1)
    
    return K_flat, midpoints

_K_flat, _t_mid = _precompute_conv_kernel()

def _compute_score(h):
    """Compute the ratio L2^2 / (L1 * sup|autoconv|) for step heights h."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    
    # L1 norm (integral)
    L1 = np.sum(h) * dw
    
    # L2 norm squared (integral of h^2)
    L2_sq = np.sum(h**2) * dw
    
    # Compute autoconvolution using precomputed kernel
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ _K_flat
    
    sup_conv = np.max(np.abs(autoconv))
    
    if sup_conv < 1e-15:
        return 0.0
    
    return L2_sq / (L1 * sup_conv)

def _score_neg(h):
    """Negative score for minimization."""
    return -_compute_score(h)

def _score_neg_smooth(h, beta=100.0):
    """Negative smooth surrogate score using log-sum-exp for the supremum."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 0.0
    
    dw = 1.0 / N
    L1 = np.sum(h) * dw
    L2_sq = np.sum(h**2) * dw
    
    h_outer = np.outer(h, h).flatten()
    autoconv = h_outer @ _K_flat
    
    # Smooth max using log-sum-exp
    abs_ac = np.abs(autoconv)
    max_val = np.max(abs_ac)
    if max_val < 1e-15:
        return 0.0
    
    shifted = beta * (abs_ac - max_val)
    log_sum = max_val + np.log(np.sum(np.exp(shifted))) / beta
    
    if log_sum < 1e-15:
        return 0.0
    
    return -(L2_sq / (L1 * log_sum))

def _gradient_ascent(h, max_iters=2000, lr_init=0.01, lr_decay=0.999, tol=1e-12):
    """
    Gradient ascent on the score function with nonnegativity constraint.
    Uses numerical gradient since the max operation makes analytic gradient complex.
    """
    h = h.copy()
    best_score = _compute_score(h)
    best_h = h.copy()
    lr = lr_init
    
    for it in range(max_iters):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        
        score = _compute_score(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_minus = h.copy()
            h_plus[i] = max(0, h[i] + eps)
            h_minus[i] = max(0, h[i] - eps)
            
            score_plus = _compute_score(h_plus)
            score_minus = _compute_score(h_minus)
            
            if h[i] > eps:
                grad[i] = (score_plus - score_minus) / (2 * eps)
            else:
                grad[i] = (score_plus - score_minus) / eps
        
        # Gradient ascent step
        h_new = h + lr * grad
        h_new = np.maximum(h_new, 0.0)
        
        # Keep scale reasonable
        if np.sum(h_new) > 1e10:
            h_new = h_new / np.sum(h_new) * N
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < tol:
            break
        
        h = h_new
        lr *= lr_decay
        
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            current_score = _compute_score(h)
            if current_score < best_score - 1e-8:
                lr *= 0.5
    
    final_score = _compute_score(h)
    if final_score > best_score:
        best_score = final_score
        best_h = h.copy()
    
    return best_h, best_score

def _coordinate_descent(h, max_rounds=100, tol=1e-10):
    """
    Coordinate descent on nonnegative vectors.
    """
    h = h.copy()
    best_score = _compute_score(h)
    
    for rnd in range(max_rounds):
        if time.time() - _start_time > _TIME_LIMIT:
            break
        improved = False
        for i in range(N):
            if time.time() - _start_time > _TIME_LIMIT:
                break
            old_val = h[i]
            mean_val = np.sum(h) / N
            candidates = [0.0, old_val]
            if old_val > 0:
                candidates += [old_val * 0.5, old_val * 0.75, old_val * 1.25, old_val * 1.5]
            candidates += [mean_val * 0.5, mean_val, mean_val * 1.5, mean_val * 2.0]
            candidates = [max(0.0, c) for c in candidates]
            candidates += [1e-4, 1e-3, 1e-2]
            
            best_cand = old_val
            best_cand_score = best_score
            
            for c in candidates:
                if abs(c - old_val) < 1e-12:
                    continue
                h[i] = c
                s = _compute_score(h)
                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c
            
            h[i] = best_cand
            if best_cand != old_val:
                improved = True
                if best_cand_score > best_score:
                    best_score = best_cand_score
        
        if not improved:
            break
    
    return h, best_score

def _structured_inits():
    """Generate diverse structured initializations."""
    inits = []
    
    # Sparse: few nonzero with equal weights
    for k in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        h = np.zeros(N)
        h[:k] = 1.0 / k
        inits.append(h)
    
    # Spread: all nonzero
    h = np.ones(N) / N
    inits.append(h)
    
    # Gaussian-like
    idx = np.arange(N)
    center = N / 2
    h = np.exp(-0.5 * ((idx - center) / (N / 4)) ** 2)
    h /= h.sum()
    inits.append(h)
    
    # Two bumps
    h = np.zeros(N)
    h[5:10] = 0.2
    h[30:35] = 0.2
    h /= h.sum()
    inits.append(h)
    
    # Alternating
    h = np.zeros(N)
    h[::2] = 1.0 / (N // 2 + 1)
    inits.append(h)
    
    # Triangular
    h = np.abs(np.sin(np.pi * np.arange(N) / N))
    h /= h.sum()
    inits.append(h)
    
    # Exponential decay
    h = np.exp(-0.1 * np.arange(N))
    h /= h.sum()
    inits.append(h)
    
    # Power law
    h = 1.0 / (np.arange(N) + 1) ** 0.5
    h /= h.sum()
    inits.append(h)
    
    return inits

def _random_init(seed):
    """Generate a random nonnegative vector."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.1, 2.0, N)
    h = rng.dirichlet(alpha) * rng.uniform(0.5, 2.0)
    return h

def _sparse_random_init(seed, n_nonzero=None):
    """Generate a random sparse nonnegative vector."""
    rng = np.random.default_rng(seed)
    if n_nonzero is None:
        n_nonzero = rng.integers(3, 15)
    positions = rng.choice(N, size=n_nonzero, replace=False)
    h = np.zeros(N)
    weights = rng.dirichlet(np.ones(n_nonzero))
    h[positions] = weights
    return h

def _optimize_from_init(init, method='lbgfs', max_iters=None):
    """Run optimization from an initialization."""
    h = np.array(init, dtype=float)
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-12:
        return h, 0.0
    
    if method == 'lbgfs':
        # Use L-BFGS-B with smooth surrogate
        try:
            result = minimize(_score_neg_smooth, h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             args=(50.0,),
                             options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            return h_trial, score_trial
        except Exception:
            pass
        # Fallback to gradient ascent
        return _gradient_ascent(h, max_iters=2000)
    elif method == 'grad':
        if max_iters is None:
            max_iters = 3000
        return _gradient_ascent(h, max_iters=max_iters)
    else:
        # Fallback: coordinate descent
        return _coordinate_descent(h, max_rounds=100)

def _genetic_search(pop_size=20, generations=30, mutation_rate=0.1, crossover_rate=0.7):
    """
    Genetic algorithm that maximizes spectral flatness (minimizes spectral peaks).
    The autoconvolution peak is related to the inverse of the minimal magnitude
    of the Fourier transform. We optimize the support mask and weights.
    """
    rng = np.random.default_rng(42)
    
    # Initialize population
    population = []
    for i in range(pop_size):
        if i < 5:
            # Structured inits
            inits = _structured_inits()
            h = inits[i % len(inits)].copy()
        else:
            # Random inits
            h = _sparse_random_init(i, n_nonzero=rng.integers(3, 12))
        population.append(h)
    
    # Evaluate initial population
    scores = [_compute_score(h) for h in population]
    
    best_h = population[np.argmax(scores)].copy()
    best_score = max(scores)
    
    for gen in range(generations):
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_h = population[0].copy()
        
        # Selection: keep top half
        elite = population[:pop_size // 2]
        
        # Generate new population
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            # Crossover
            if rng.random() < crossover_rate and len(elite) >= 2:
                parent1 = elite[rng.integers(0, len(elite))]
                parent2 = elite[rng.integers(0, len(elite))]
                # Uniform crossover
                mask = rng.random(N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                parent = elite[rng.integers(0, len(elite))]
                child = parent.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                # Sparse mutation: add/remove a few nonzeros
                n_mut = rng.integers(1, 4)
                for _ in range(n_mut):
                    idx = rng.integers(0, N)
                    if rng.random() < 0.5:
                        child[idx] = 0.0
                    else:
                        child[idx] = rng.uniform(0.1, 1.0)
            else:
                # Gaussian mutation
                child = child + rng.normal(0, 0.05, N)
                child = np.maximum(child, 0.0)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
        scores = [_compute_score(h) for h in population]
    
    return best_h, best_score

def construct():
    best_h = None
    best_score = 0.0
    
    # Generate initializations
    inits = _structured_inits()
    
    # Add random inits
    for seed in range(10):
        inits.append(_random_init(seed))
    
    # Add sparse random inits
    for seed in range(15):
        inits.append(_sparse_random_init(seed))
    
    # Phase 1: L-BFGS-B optimization from each init with smooth surrogate
    for init in inits:
        if time.time() - _start_time > _TIME_LIMIT - 30:
            break
        h_opt, s_opt = _optimize_from_init(init, method='lbgfs')
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Phase 2: Increase beta for sharper approximation
    if best_h is not None:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() - _start_time > _TIME_LIMIT - 20:
                break
            try:
                result = minimize(_score_neg_smooth, best_h, method='L-BFGS-B',
                                 bounds=[(0, 10)] * N,
                                 args=(beta,),
                                 options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10})
                h_trial = np.maximum(result.x, 0.0)
                score_trial = _compute_score(h_trial)
                if score_trial > best_score:
                    best_score = score_trial
                    best_h = h_trial
            except Exception:
                pass
    
    # Phase 3: Direct optimization with exact (non-smooth) objective
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 15:
        try:
            result = minimize(_score_neg, best_h, method='L-BFGS-B',
                             bounds=[(0, 10)] * N,
                             options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-10})
            h_trial = np.maximum(result.x, 0.0)
            score_trial = _compute_score(h_trial)
            if score_trial > best_score:
                best_score = score_trial
                best_h = h_trial
        except Exception:
            pass
    
    # Phase 4: Gradient ascent refinement
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 10:
        h_opt, s_opt = _gradient_ascent(best_h, max_iters=2000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Phase 5: Coordinate descent refinement
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 5:
        h_opt, s_opt = _coordinate_descent(best_h, max_rounds=100)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    # Phase 6: Genetic search
    if time.time() - _start_time < _TIME_LIMIT - 5:
        h_gen, s_gen = _genetic_search(pop_size=15, generations=20)
        if s_gen > best_score:
            best_score = s_gen
            best_h = h_gen.copy()
    
    # Final polish
    if best_h is not None and time.time() - _start_time < _TIME_LIMIT - 2:
        h_opt, s_opt = _gradient_ascent(best_h, max_iters=1000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.zeros(N)
        best_h[0] = 2.0 / 3.0
        best_h[1] = 1.0 / 3.0
    
    return [float(x) for x in best_h]
