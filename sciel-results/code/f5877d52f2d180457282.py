import numpy as np
import time

N = 400

def _compute_autocorr(h):
    """Compute linear aperiodic autocorrelation using FFT."""
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """Compute max absolute value of sidelobes."""
    return np.max(np.abs(ac[1:N]))

def _evaluate(h):
    """Evaluate the peak sidelobe of the autoconvolution."""
    ac = _compute_autocorr(h)
    return _max_sidelobe(ac)

def _simulated_annealing(h, time_limit, deadline=None, T0=None, verbose=False):
    """
    Simulated annealing with multiple move operators.
    Returns (best_h, best_val).
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    if T0 is None:
        T0 = current_val * 0.1 if current_val > 1e-10 else 1.0
    
    T = T0
    best_h = h.copy()
    best_val = current_val
    
    # Track acceptance for adaptive cooling
    accepted = 0
    total = 0
    window_size = 1000
    accept_history = []
    
    iteration = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if deadline is not None and time.time() >= deadline:
            break
        
        # Choose move operator
        move_type = np.random.randint(3)
        old_h = h.copy()
        
        if move_type == 0:
            # Single-bit flip
            i = np.random.randint(N)
            h[i] = -h[i]
        elif move_type == 1:
            # Block flip
            block_len = np.random.randint(1, 11)
            start = np.random.randint(0, N - block_len + 1)
            h[start:start+block_len] = -h[start:start+block_len]
        else:
            # Two-bit swap (if different)
            i, j = np.random.choice(N, size=2, replace=False)
            if h[i] != h[j]:
                h[i], h[j] = h[j], h[i]
            else:
                # Just flip one
                h[i] = -h[i]
        
        # Evaluate
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        delta = new_val - current_val
        
        total += 1
        if delta < 0:
            # Improvement
            current_val = new_val
            ac = ac_new
            accepted += 1
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            # Deterioration
            if T > 1e-12:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_val = new_val
                    ac = ac_new
                    accepted += 1
                else:
                    h = old_h
            else:
                h = old_h
        
        # Adaptive cooling
        if total % window_size == 0:
            accept_rate = accepted / total
            # Target ~50% acceptance
            if accept_rate > 0.5:
                T *= 0.9
            elif accept_rate < 0.1:
                T *= 1.1
            else:
                T *= 0.95
            T = max(T, 1e-12)
            accepted = 0
            total = 0
        
        iteration += 1
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Multiple restarts with different strategies
    num_restarts = 15
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        # Different initialization strategies
        if r == 0:
            # Start with all ones, then randomize
            h = np.ones(N)
            np.random.shuffle(h)
            h = np.sign(h)
        elif r == 1:
            # Random
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r == 2:
            # Alternating
            h = np.ones(N) * (2 * (np.arange(N) % 2) - 1)
        elif r == 3:
            # Block pattern
            h = np.ones(N)
            for i in range(0, N, 8):
                h[i:i+4] = -1
        else:
            # Random
            h = np.random.choice([-1.0, 1.0], size=N)
        
        # Run simulated annealing
        T0 = best_val * 0.1 if best_val < float('inf') else 10.0
        h_new, val_new = _simulated_annealing(
            h, time_limit=time_per_restart, deadline=deadline, T0=T0
        )
        
        if val_new < best_val:
            best_val = val_new
            best_h = h_new
    
    # Final polish: run SA on the best solution with more time
    if best_h is not None and time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 1.0:
            T0 = best_val * 0.05
            h_final, val_final = _simulated_annealing(
                best_h, time_limit=remaining, deadline=deadline, T0=T0
            )
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
