"""Optimal program for the suite task."""
import numpy as np
import time

N = 400
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n
    fx = np.fft.rfft(x, size)
    ac = np.fft.irfft(fx * np.conj(fx), size)
    return ac[:n]  # lags 0..n-1

def _metric(x):
    """max |autoconv| / (sum x)^2"""
    ac = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _try_constant():
    x = np.ones(N)
    return x, _metric(x)

def _try_m_sequence():
    """Generate an M-sequence (maximal length LFSR sequence) of length 2^m - 1, pad/truncate to N."""
    # Try several LFSR configurations
    best_x, best_m = None, np.inf
    for m in range(7, 13):
        # Use a primitive polynomial
        polys = {
            7: 0x60, 8: 0x49, 9: 0x11D, 10: 0x131, 11: 0x171, 12: 0x1D9
        }
        if m not in polys:
            continue
        poly = polys[m]
        # Generate M-sequence
        seq = []
        state = 1
        for i in range(2**m - 1):
            seq.append(1 if state & 1 else -1)
            # LFSR step
            bit = 0
            for bit_pos in range(m):
                if poly & (1 << bit_pos):
                    bit ^= (state >> bit_pos) & 1
            state = (state >> 1) | (bit << (m - 1))
        seq = np.array(seq, dtype=np.float64)
        # Pad or truncate to N
        if len(seq) >= N:
            x = seq[:N]
        else:
            reps = (N // len(seq)) + 1
            x = np.tile(seq, reps)[:N]
        m_val = _metric(x)
        if m_val < best_m:
            best_m = m_val
            best_x = x.copy()
    return best_x, best_m

def _try_alternating_blocks():
    """Try sequences with alternating block patterns."""
    best_x, best_m = None, np.inf
    for block_len in range(2, 50):
        for num_blocks in range(2, N // block_len + 1):
            if time.time() > _DEADLINE:
                break
            # Create pattern: +1 for block_len, -1 for block_len, repeat
            pattern = np.tile([1.0, -1.0], block_len)
            x = np.tile(pattern, N // len(pattern) + 1)[:N]
            m_val = _metric(x)
            if m_val < best_m:
                best_m = m_val
                best_x = x.copy()
    return best_x, best_m

def _try_gaussian_weighted():
    """Try sequences with Gaussian-like weighting."""
    best_x, best_m = None, np.inf
    for sigma in [5, 10, 20, 50, 100, 200]:
        t = np.arange(N)
        center = N / 2
        w = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        x = np.sign(w - 0.1)  # threshold to create signed steps
        x[x == 0] = 1.0
        m_val = _metric(x)
        if m_val < best_m:
            best_m = m_val
            best_x = x.copy()
    return best_x, best_m

def _try_local_search(x0, max_iter=5000):
    """Local search: flip random elements and accept if metric improves."""
    x = x0.copy()
    current_metric = _metric(x)
    best_x = x.copy()
    best_metric = current_metric
    
    for i in range(max_iter):
        if time.time() > _DEADLINE:
            break
        idx = np.random.randint(0, N)
        x[idx] *= -1.0  # flip sign
        new_metric = _metric(x)
        if new_metric < current_metric:
            current_metric = new_metric
            if new_metric < best_metric:
                best_metric = new_metric
                best_x = x.copy()
        else:
            x[idx] *= -1.0  # revert
    return best_x, best_metric

def construct():
    start_time = time.time()
    
    # Try multiple approaches
    candidates = []
    
    # 1. Constant sequence
    x_const, m_const = _try_constant()
    candidates.append((x_const, m_const))
    
    # 2. M-sequences
    x_m, m_m = _try_m_sequence()
    if x_m is not None:
        candidates.append((x_m, m_m))
    
    # 3. Alternating blocks
    x_block, m_block = _try_alternating_blocks()
    if x_block is not None:
        candidates.append((x_block, m_block))
    
    # 4. Gaussian weighted
    x_gauss, m_gauss = _try_gaussian_weighted()
    if x_gauss is not None:
        candidates.append((x_gauss, m_gauss))
    
    # 5. Local search from best candidate
    if candidates:
        candidates.sort(key=lambda t: t[1])
        best_x, best_m = candidates[0]
        
        # Run local search from a few starting points
        for x_start, m_start in candidates[:3]:
            if time.time() > _DEADLINE - 5:
                break
            x_ls, m_ls = _try_local_search(x_start, max_iter=2000)
            if m_ls < best_m:
                best_m = m_ls
                best_x = x_ls
        
        return best_x
    
    return np.ones(N)
