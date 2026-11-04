"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_max_abs(f):
    """Compute max |f*f| over all lags using FFT."""
    n = len(f)
    conv = np.fft.ifft(np.fft.fft(f, 2*n) ** 2).real
    return np.max(np.abs(conv))


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    best_f = None
    best_val = np.inf
    
    np.random.seed(42)
    
    # Strategy: combine sparse combinatorial design (from inspiration 2)
    # with multi-start local search (from current program).
    # The key structural change: start with sparse patterns (many zeros)
    # rather than dense ±1 patterns. Sparse patterns can have much lower
    # autoconvolution peaks because most lags have few or zero overlapping
    # non-zero terms.
    
    # Try different sparsity levels
    sparsity_levels = [20, 30, 40, 50, 60, 80, 100, 120, 150]
    
    for K in sparsity_levels:
        if time.time() > deadline - 20:
            break
        
        # Multiple random starts for each sparsity level
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization: pick K random indices
            np.random.seed(42 + trial)
            active_indices = np.random.choice(N, size=K, replace=False)
            signs = np.random.choice([-1, 1], size=K)
            
            f = np.zeros(N)
            f[active_indices] = signs
            
            current_score = _autoconvolution_max_abs(f)
            
            if current_score < best_val:
                best_val = current_score
                best_f = f.copy()
            
            # Local search: flip individual signs
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for i in range(K):
                    if time.time() > deadline - 10:
                        break
                    idx = active_indices[i]
                    old_sign = f[idx]
                    f[idx] = -old_sign
                    new_score = _autoconvolution_max_abs(f)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                        if new_score < best_val:
                            best_val = new_score
                            best_f = f.copy()
                    else:
                        f[idx] = old_sign
            
            # Also try swapping active indices to change the support
            if time.time() < deadline - 10:
                for _ in range(100):
                    if time.time() > deadline - 10:
                        break
                    # Pick a random active index to remove
                    i_remove = np.random.randint(K)
                    # Pick a random inactive index to add
                    inactive = np.setdiff1d(np.arange(N), active_indices)
                    if len(inactive) == 0:
                        break
                    j_add = inactive[np.random.randint(len(inactive))]
                    
                    idx_remove = active_indices[i_remove]
                    f[idx_remove] = 0
                    f[j_add] = 1  # Start with +1
                    
                    new_active = np.delete(active_indices, i_remove)
                    new_active = np.append(new_active, j_add)
                    
                    new_score = _autoconvolution_max_abs(f)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        if new_score < best_val:
                            best_val = new_score
                            best_f = f.copy()
                        # Refine sign of the new element
                        f[j_add] = -1
                        alt_score = _autoconvolution_max_abs(f)
                        if alt_score < current_score:
                            current_score = alt_score
                            if alt_score < best_val:
                                best_val = alt_score
                                best_f = f.copy()
                        else:
                            f[j_add] = 1
                        active_indices = new_active
                    else:
                        # Revert
                        f[j_add] = 0
                        f[idx_remove] = 1
                        active_indices = np.delete(active_indices, len(active_indices) - 1)
                        active_indices = np.insert(active_indices, i_remove, idx_remove)
    
    # Final polish: try all sign flips one more time on best solution
    if best_f is not None and time.time() < deadline - 5:
        active = np.nonzero(best_f)[0]
        for idx in active:
            if time.time() > deadline - 5:
                break
            old_val = best_f[idx]
            best_f[idx] = -old_val
            s = _autoconvolution_max_abs(best_f)
            if s < best_val - 1e-12:
                best_val = s
            else:
                best_f[idx] = old_val
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
