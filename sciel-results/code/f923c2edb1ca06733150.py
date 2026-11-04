"""Optimized program for second autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    The autoconvolution here is circular convolution of h with itself.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    l2_sq = np.sum(h**2)
    # Circular autoconvolution: h * h
    autoconv = np.fft.ifft(np.fft.fft(h)**2).real
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm < 1e-15:
        return -np.inf
    return l2_sq / (l1 * sup_norm)

def _negative_score(h):
    """Negative of the score for minimization."""
    s = _score(h)
    if s == -np.inf:
        return 1e10
    return -s

def _sidon_set(n, target_size):
    """Find a Sidon set in Z_n via greedy with randomization."""
    best_sidon = [0]
    best_score_val = 0
    for _ in range(20):
        sidon = []
        order = list(range(n))
        np.random.shuffle(order)
        for cand in order:
            if len(sidon) >= target_size:
                break
            ok = True
            for i in range(len(sidon)):
                for j in range(i+1, len(sidon)):
                    s1 = (sidon[i] + sidon[j]) % n
                    s2 = (sidon[i] + cand) % n
                    s3 = (sidon[j] + cand) % n
                    if s2 == s3 or s2 == s1 or s3 == s1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                sidon.append(cand)
        if len(sidon) > len(best_sidon):
            best_sidon = sidon
    return best_sidon

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    deadline = time.time() + 110.0
    rng = np.random.RandomState(42)
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Phase 1: Sidon-set based initializations
    for target_size in [6, 8, 10, 12, 15]:
        if time.time() > deadline - 30:
            break
        sidon = _sidon_set(N, target_size)
        if len(sidon) < 3:
            continue
        for trial in range(8):
            if time.time() > deadline - 30:
                break
            h_init = np.zeros(N)
            for s in sidon:
                h_init[s] = rng.uniform(0.8, 1.5)
            # Add small noise to other positions
            noise_mask = rng.random(N) < 0.1
            h_init[noise_mask] = rng.uniform(0, 0.1, size=np.sum(noise_mask))
            try:
                result = minimize(_negative_score, h_init, method='L-BFGS-B',
                                  bounds=[(0, None)]*N,
                                  options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10})
                h_opt = np.maximum(result.x, 0)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 2: Sparse random initializations
    for sparsity in [0.15, 0.25, 0.35, 0.5]:
        if time.time() > deadline - 20:
            break
        n_nz = max(3, int(N * sparsity))
        for trial in range(6):
            if time.time() > deadline - 20:
                break
            h_init = np.zeros(N)
            nz = rng.choice(N, size=n_nz, replace=False)
            h_init[nz] = rng.uniform(0.5, 2.0, size=n_nz)
            try:
                result = minimize(_negative_score, h_init, method='L-BFGS-B',
                                  bounds=[(0, None)]*N,
                                  options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10})
                h_opt = np.maximum(result.x, 0)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Spectral mask initializations
    for K in [3, 5, 7, 10]:
        if time.time() > deadline - 15:
            break
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            mask = np.zeros(N, dtype=complex)
            active = rng.choice(N, size=K, replace=False)
            for idx in active:
                mag = rng.uniform(0.5, 2.0)
                phase = rng.uniform(0, 2*np.pi)
                mask[idx] = mag * np.exp(1j * phase)
            h_init = np.abs(np.fft.ifft(mask))**2
            h_init = np.maximum(h_init, 0)
            if np.sum(h_init) > 1e-10:
                h_init = h_init / np.sum(h_init) * N
            try:
                result = minimize(_negative_score, h_init, method='L-BFGS-B',
                                  bounds=[(0, None)]*N,
                                  options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10})
                h_opt = np.maximum(result.x, 0)
                s = _score(h_opt)
                if s > best_score:
                    best_score = s
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 4: Local refinement
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        for _ in range(1000):
            if time.time() > deadline - 3:
                break
            idx = rng.randint(N)
            scale = max(0.01, h_curr[idx])
            delta = rng.uniform(-0.15, 0.15) * scale
            h_new = h_curr.copy()
            h_new[idx] = max(0.0, h_curr[idx] + delta)
            s_new = _score(h_new)
            if s_new > best_score:
                best_score = s_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                h_curr = best_h.copy()
    
    best_h = np.maximum(best_h, 0)
    return list(best_h)
