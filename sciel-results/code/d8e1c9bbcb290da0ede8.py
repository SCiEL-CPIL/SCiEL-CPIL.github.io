import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _neg_ratio(x):
    """Negative ratio for minimization. x is log-parameterized."""
    h = np.exp(x)
    return -_compute_ratio(h)

def _get_sidon_set(n, size):
    """
    Generate a Sidon-like set of 'size' elements from [0, n).
    Uses a greedy approach to maximize minimum pairwise sum difference.
    """
    if size == 0:
        return np.array([], dtype=int)
    if size == 1:
        return np.array([0], dtype=int)
    
    # Start with a simple construction: use quadratic residues or a greedy method
    # For small sizes, we can use a known Sidon set construction
    # B^2 sequence: a_k = k^2 mod p for prime p, or use the classical construction
    
    # Simple greedy: pick elements that maximize minimum pairwise sum separation
    chosen = [0]
    for _ in range(1, size):
        best_pos = -1
        best_min_sep = -1
        for pos in range(n):
            if pos in chosen:
                continue
            # Check all pairwise sums with existing elements
            sums = [chosen[i] + pos for i in range(len(chosen))]
            # Also include pos+pos
            all_sums = sums + [2 * pos]
            # Check separation from existing pairwise sums
            existing_sums = set()
            for i in range(len(chosen)):
                for j in range(i, len(chosen)):
                    existing_sums.add(chosen[i] + chosen[j])
            
            min_sep = min(abs(s - es) for s in all_sums for es in existing_sums) if existing_sums else 0
            
            if min_sep > best_min_sep or (min_sep == best_min_sep and pos > best_pos):
                best_min_sep = min_sep
                best_pos = pos
        
        if best_pos >= 0:
            chosen.append(best_pos)
    
    return np.array(sorted(chosen), dtype=int)

def _sidon_initialization(size, n=N):
    """Create initial h with 1s at Sidon set positions, 0 elsewhere."""
    sidon = _get_sidon_set(n, size)
    h = np.zeros(n)
    for pos in sidon:
        if 0 <= pos < n:
            h[pos] = 1.0
    # Ensure at least one nonzero
    if np.sum(h) == 0:
        h[0] = 1.0
    return h

def _optimize_from(h0, time_budget=5.0):
    """
    Optimize starting from h0 using L-BFGS-B with log-parameterization.
    Returns (h_opt, score).
    """
    if np.max(h0) <= 0:
        h0 = h0.copy()
        h0[0] = 1.0
    
    # Log-parameterization: x = log(h + eps) to keep h > 0
    eps = 1e-10
    x0 = np.log(h0 + eps)
    
    start_time = time.time()
    
    result = minimize(
        _neg_ratio,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    
    h_opt = np.exp(result.x)
    # Clip to ensure nonnegativity
    h_opt = np.maximum(h_opt, 0.0)
    
    score = _compute_ratio(h_opt)
    
    return h_opt, score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_h = None
    best_score = 0.0
    
    # Try different Sidon set sizes
    # For N=50, good Sidon sets have size around 7-12
    sidon_sizes = [5, 6, 7, 8, 9, 10, 11, 12]
    
    # Also try some random initializations for diversity
    np.random.seed(42)
    
    for size in sidon_sizes:
        if time.time() > deadline - 15:
            break
        
        h0 = _sidon_initialization(size)
        h_opt, score = _optimize_from(h0, time_budget=8.0)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Add a few random restarts if time permits
    for r in range(5):
        if time.time() > deadline - 10:
            break
        
        # Random sparse initialization
        n_nonzero = np.random.randint(5, 15)
        positions = np.random.choice(N, size=n_nonzero, replace=False)
        h0 = np.zeros(N)
        h0[positions] = 1.0
        
        h_opt, score = _optimize_from(h0, time_budget=5.0)
        
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    if best_h is None:
        # Fallback: all ones
        return [1.0] * N
    
    return list(best_h)
