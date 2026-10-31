import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    C = np.fft.irfft(F * F, n=fft_size)
    return C[:size]


def _score(f):
    """Compute max(|autoconv(f)|) / (sum(f))^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)


def _make_sparse_sequence(support, signs):
    """Create a sparse sequence with given support positions and signs."""
    f = np.zeros(N)
    for pos, sign in zip(support, signs):
        f[pos] = sign
    return f


def _simulated_annealing_sparse(rng, initial_support, initial_signs, time_budget=30.0):
    """
    Simulated annealing on sparse signed sequences.
    State: (support_set, signs)
    Moves: add/remove/flip sign of an element
    """
    support = list(initial_support)
    signs = list(initial_signs)
    f = _make_sparse_sequence(support, signs)
    current_score = _score(f)
    
    best_support = support.copy()
    best_signs = signs.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 1.0  # Temperature
    cooling = 0.999
    
    while time.time() - start_time < time_budget and T > 0.01:
        if time.time() > DEADLINE:
            break
        
        move_type = rng.random()
        
        if move_type < 0.3 and len(support) < 30:
            # Add a new element
            pos = rng.integers(0, N)
            if pos not in support:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                support.append(pos)
                signs.append(sign)
        elif move_type < 0.6 and len(support) > 5:
            # Remove an element
            idx = rng.integers(0, len(support))
            support.pop(idx)
            signs.pop(idx)
        elif move_type < 0.9 and len(support) > 0:
            # Flip sign of an element
            idx = rng.integers(0, len(support))
            signs[idx] = -signs[idx]
        else:
            # Move an element to a new position
            if len(support) == 0:
                continue
            idx = rng.integers(0, len(support))
            old_pos = support[idx]
            new_pos = rng.integers(0, N)
            if new_pos not in support:
                support[idx] = new_pos
            else:
                # Revert
                support[idx] = old_pos
                continue
        
        new_f = _make_sparse_sequence(support, signs)
        new_score = _score(new_f)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_support = support.copy()
                best_signs = signs.copy()
        else:
            # Revert
            if move_type < 0.3 and len(support) > len(initial_support):
                support.pop()
                signs.pop()
            elif move_type < 0.6:
                # Can't easily revert removal without storing state
                pass
            elif move_type < 0.9:
                signs[idx] = -signs[idx]
            else:
                support[idx] = old_pos
        
        T *= cooling
    
    best_f = _make_sparse_sequence(best_support, best_signs)
    return best_f, best_score


def _local_search_full(h, deadline, max_passes=20):
    """
    Local optimization with single-flip moves on full-length sequences.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    max_abs_ac = np.max(np.abs(ac))
    s = np.sum(h)
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        
        # Single flip moves
        for i in range(n):
            if time.time() > deadline:
                break
            
            v = h[i]
            delta = -2.0 * v
            
            # Try flipping position i
            h[i] = v + delta
            H_new = np.fft.rfft(h, fsize)
            ac_new = np.fft.irfft(H_new * H_new, fsize)[:size]
            max_abs_new = np.max(np.abs(ac_new))
            s_new = s + delta
            
            if abs(s_new) < 1e-6:
                h[i] = v
                continue
                
            val_new = max_abs_new / (s_new * s_new)
            
            if val_new < best_val - 1e-12:
                best_val = val_new
                H = H_new
                ac = ac_new
                max_abs_ac = max_abs_new
                s = s_new
                improved = True
            else:
                h[i] = v
        
        if not improved:
            break
    
    return h, best_val


def _generate_chirp_sequence(freq, N):
    """Generate a signed sequence from a quadratic phase chirp."""
    x = np.arange(N, dtype=np.float64)
    phase = np.pi * freq * x * x / N
    h = np.sign(np.cos(phase))
    h[h == 0] = 1.0
    return h


def _optimize_chirp_freq(freq_init, N, deadline):
    """Optimize chirp frequency using local search."""
    best_h = _generate_chirp_sequence(freq_init, N)
    best_val = _score(best_h)
    
    # Try perturbations around the initial frequency
    for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02, 0.001, -0.001]:
        if time.time() > deadline:
            break
        freq = freq_init + delta
        h = _generate_chirp_sequence(freq, N)
        val = _score(h)
        if val < best_val:
            best_val = val
            best_h = h
    
    return best_h, best_val


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Baseline: all ones
    best_f = np.ones(N)
    best_score = _score(best_f)
    
    # Strategy 1: Chirp sequences (full-length) with local search
    for freq in [1.0, 2.0, 0.5, 3.0, 1.5, 2.5]:
        if time.time() > DEADLINE:
            break
        
        h_chirp, val_chirp = _optimize_chirp_freq(freq, N, DEADLINE)
        
        if val_chirp < best_score:
            best_score = val_chirp
            best_f = h_chirp.copy()
        
        # Local search on chirp
        remaining = DEADLINE - time.time()
        if remaining > 10:
            h_opt, val_opt = _local_search_full(h_chirp, DEADLINE, max_passes=10)
            if val_opt < best_score:
                best_score = val_opt
                best_f = h_opt.copy()
    
    # Strategy 2: Sparse sequences with simulated annealing
    for restart in range(5):
        if time.time() > DEADLINE:
            break
        
        # Start with a random sparse support
        K = rng.integers(8, 25)
        support = list(rng.choice(N, size=K, replace=False))
        signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(K)]
        
        # Ensure non-zero sum
        S = sum(signs)
        if abs(S) < 3:
            # Adjust signs to increase |S|
            for i in range(K):
                if signs[i] * S < 0:
                    signs[i] = -signs[i]
                    S = sum(signs)
            if abs(S) < 3:
                # Make all same sign
                signs = [1.0] * K
                S = K
        
        remaining_time = DEADLINE - time.time()
        time_budget = min(15.0, remaining_time * 0.7 / (5 - restart))
        
        if time_budget < 1.0:
            break
        
        f_sa, score_sa = _simulated_annealing_sparse(rng, support, signs, time_budget=time_budget)
        
        if score_sa < best_score:
            best_score = score_sa
            best_f = f_sa.copy()
    
    # Strategy 3: Local improvement on best solution
    if time.time() < DEADLINE - 5:
        # Convert to full-length for local search
        f_local = best_f.copy()
        h_opt, val_opt = _local_search_full(f_local, DEADLINE, max_passes=15)
        if val_opt < best_score:
            best_score = val_opt
            best_f = h_opt.copy()
    
    return best_f
