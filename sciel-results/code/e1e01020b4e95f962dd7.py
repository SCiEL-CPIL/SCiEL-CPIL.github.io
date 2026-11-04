"""Optimized program for the second autocorrelation inequality task using sparse Sidon-set representation."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.maximum(h, 0.0)
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if peak == 0:
        return 0.0
    return l2sq / (l1 * peak)

def _smooth_max(x, sigma=0.01):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    return x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma

def _generate_sidon_sets(K, num_sets=10, seed=42):
    """
    Generate multiple Sidon sets of size K from {0, 1, ..., N-1}.
    A Sidon set has all pairwise differences distinct.
    """
    rng = np.random.RandomState(seed)
    sidon_sets = []
    
    for _ in range(num_sets):
        attempts = 0
        while attempts < 100:
            S = np.sort(rng.choice(N, size=K, replace=False))
            # Check if it's a Sidon set
            diffs = []
            valid = True
            for i in range(K):
                for j in range(i+1, K):
                    d = S[j] - S[i]
                    if d in diffs:
                        valid = False
                        break
                    diffs.append(d)
                if not valid:
                    break
            if valid:
                sidon_sets.append(S)
                break
            attempts += 1
        
        if not valid:
            # Fallback: use a known Sidon construction
            base = 3 * (K - 1)
            S = np.array([i * (K - 1) + (i % 2) for i in range(K)])
            if S[-1] < N:
                sidon_sets.append(S)
    
    return sidon_sets

def _evaluate_sparse_h(amplitudes, sidon_set):
    """
    Construct h from sparse polynomial: h[n] = sum_k |a_k|^2 * delta(n - s_k)
    where a_k are complex amplitudes and s_k are Sidon set elements.
    """
    h = np.zeros(N)
    for k, s in enumerate(sidon_set):
        h[s] = np.abs(amplitudes[k])**2
    return h

def _objective_sparse(params, sidon_set):
    """
    Objective to minimize: negative of the ratio for sparse representation.
    params: real vector of length 2*K representing complex amplitudes.
    """
    K = len(sidon_set)
    amp_real = params[:K]
    amp_imag = params[K:]
    amplitudes = amp_real + 1j * amp_imag
    
    h = _evaluate_sparse_h(amplitudes, sidon_set)
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    
    l2sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    peak = _smooth_max(conv)
    
    if peak == 0:
        return 0.0
    
    return -l2sq / peak

def _optimize_sparse(sidon_set, K, time_limit):
    """
    Optimize complex amplitudes for a given Sidon set.
    Returns optimal h.
    """
    start_time = time.time()
    
    # Multiple random initializations
    best_h = np.zeros(N)
    best_obj = np.inf
    
    for init_seed in range(5):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        rng = np.random.RandomState(init_seed * 123 + 7)
        amp_real = rng.randn(K)
        amp_imag = rng.randn(K)
        params0 = np.concatenate([amp_real, amp_imag])
        
        try:
            result = minimize(
                _objective_sparse,
                params0,
                args=(sidon_set,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 1000}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                amp_opt = result.x[:K] + 1j * result.x[K:]
                best_h = _evaluate_sparse_h(amp_opt, sidon_set)
        except:
            continue
    
    return best_h

def _optimize_full_from_sparse(h_sparse):
    """
    Given a sparse h, optimize all 50 heights starting from this initialization.
    """
    # Convert to g-space
    g_real = np.sqrt(h_sparse)
    g_imag = np.zeros(N)
    g_flat = np.concatenate([g_real, g_imag])
    
    def _objective_full(g_flat):
        g = g_flat.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        h = h / l1
        
        l2sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        peak = _smooth_max(conv)
        
        if peak == 0:
            return 0.0
        
        return -l2sq / peak
    
    try:
        result = minimize(
            _objective_full,
            g_flat,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
        return h_opt
    except:
        return h_sparse

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try different sparsity levels
    K_values = [3, 4, 5, 6, 7]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 20:
            break
        
        # Generate Sidon sets
        sidon_sets = _generate_sidon_sets(K, num_sets=5, seed=K * 100)
        
        for sidon_set in sidon_sets:
            elapsed = time.time() - start_time
            if elapsed > _DEADLINE - 15:
                break
            
            # Optimize sparse amplitudes
            time_per_sparse = 5.0
            if elapsed + time_per_sparse > _DEADLINE - 10:
                break
            
            h_sparse = _optimize_sparse(sidon_set, K, time_per_sparse)
            
            if np.sum(h_sparse) == 0:
                continue
            
            score_sparse = _ratio(h_sparse)
            if score_sparse > best_score:
                best_score = score_sparse
                best_h = h_sparse.copy()
            
            # Refine with full optimization
            elapsed = time.time() - start_time
            time_per_full = 8.0
            if elapsed + time_per_full > _DEADLINE - 5:
                break
            
            h_full = _optimize_full_from_sparse(h_sparse)
            
            if np.sum(h_full) > 0:
                score_full = _ratio(h_full)
                if score_full > best_score:
                    best_score = score_full
                    best_h = h_full
    
    # Final polish with perturbations
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        for pert_seed in [100, 200, 300]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            
            rng = np.random.RandomState(pert_seed)
            g_best_real = np.sqrt(best_h)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng.randn(N) * 0.03
            pert_imag = rng.randn(N) * 0.03
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            def _objective_pert(g_flat):
                g = g_flat.reshape(N, 2)
                h = g[:, 0]**2 + g[:, 1]**2
                
                l1 = np.sum(h)
                if l1 == 0:
                    return 0.0
                h = h / l1
                
                l2sq = np.sum(h**2)
                conv = np.convolve(h, h, mode='full')
                peak = _smooth_max(conv)
                
                if peak == 0:
                    return 0.0
                
                return -l2sq / peak
            
            try:
                result = minimize(
                    _objective_pert,
                    g_pert_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
                )
                
                g_opt = result.x.reshape(N, 2)
                h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
                
                score = _ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt
            except:
                pass
    
    return list(best_h)
