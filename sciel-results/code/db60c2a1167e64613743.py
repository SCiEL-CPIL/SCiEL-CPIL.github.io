import numpy as np
import time
import math

N = 400

def _compute_autocorr(x):
    """Compute periodic autocorrelation R[k] for all k using FFT."""
    X = np.fft.fft(x.astype(np.float64))
    R = np.real(np.fft.ifft(X * np.conj(X)))
    return R

def _compute_c3_bound(R):
    """Compute C3 upper bound from autocorrelation array."""
    # Peak sidelobe level: max absolute value of R[k] for k != 0
    # R[0] is the energy, which is N for +/-1 sequences.
    return np.max(np.abs(R[1:]))

def _fast_flip_autocorr_update(R, x, pos):
    """
    Fast update of autocorrelation when flipping x[pos].
    Returns the new autocorrelation array.
    """
    v = x[pos]
    new_R = R.copy()
    
    # For k=0, R[0] = sum x[i]^2 = N. Flipping x[pos] keeps x[pos]^2 = 1.
    # So R[0] should remain N.
    # However, the derivation in the prompt suggested -2 or -4.
    # Let's re-verify:
    # R[k] = sum_i x[i] x[(i+k)%N]
    # Flip x[pos] to -x[pos].
    # Term i=pos: x[pos]x[(pos+k)%N] -> -x[pos]x[(pos+k)%N]. Delta: -2 x[pos]x[(pos+k)%N]
    # Term i=(pos-k)%N: x[(pos-k)%N]x[pos] -> x[(pos-k)%N](-x[pos]). Delta: -2 x[(pos-k)%N]x[pos]
    
    # If k=0:
    # Term i=pos: x[pos]^2 -> -x[pos]^2. Delta: -2 x[pos]^2 = -2.
    # Term i=pos: Same term?
    # The sum is over i. The term is x[i]x[i].
    # Only i=pos involves x[pos].
    # So only one term changes.
    # Delta R[0] = -2 x[pos]^2 = -2.
    # But R[0] = N. If we subtract 2, R[0] becomes N-2. This is wrong because sum of squares is invariant.
    # Why?
    # Ah, the formula R[k] = sum x[i]x[i+k] for k=0 is sum x[i]^2.
    # If we flip x[pos], x[pos]^2 is still 1.
    # So R[0] should NOT change.
    # Where is the error in the delta derivation?
    # R_new[k] - R_old[k] = sum (x'_i x'_{i+k} - x_i x_{i+k})
    # Only terms involving x[pos] change.
    # For k=0:
    # Only i=pos involves x[pos].
    # x'_pos x'_pos - x_pos x_pos = (-x_pos)(-x_pos) - x_pos^2 = x_pos^2 - x_pos^2 = 0.
    # So Delta R[0] = 0.
    
    # For k != 0:
    # Terms involving x[pos]:
    # 1. i=pos: x_pos x_{pos+k} -> (-x_pos) x_{pos+k}. Delta: -2 x_pos x_{pos+k}
    # 2. i=pos-k: x_{pos-k} x_pos -> x_{pos-k} (-x_pos). Delta: -2 x_{pos-k} x_pos
    # Are these distinct?
    # i=pos and i=pos-k are distinct if k != 0 mod N.
    # So Delta R[k] = -2 x_pos (x_{pos+k} + x_{pos-k})
    
    # What if k = N/2?
    # pos+k = pos-k mod N.
    # So x_{pos+k} = x_{pos-k}.
    # Delta R[k] = -2 x_pos (2 x_{pos+k}) = -4 x_pos x_{pos+k}.
    # But wait, are the terms distinct in the sum?
    # Sum is over i.
    # Term 1: i=pos.
    # Term 2: i=pos-N/2.
    # These are distinct indices i (since N>2).
    # So both terms change.
    # So the formula holds.
    
    # So:
    # new_R[0] = R[0]  (No change)
    # new_R[k] = R[k] - 2 * v * (x[(pos+k)%N] + x[(pos-k)%N]) for k=1..N-1
    
    if N > 1:
        ks = np.arange(1, N)
        idx1 = (pos + ks) % N
        idx2 = (pos - ks) % N
        new_R[1:] = R[1:] - 2 * v * (x[idx1] + x[idx2])
        
    return new_R

def _generate_m_sequence(m):
    """Generate an m-sequence of length 2^m - 1 using LFSR."""
    if m == 9:
        taps = (1 << 4)
        length = (1 << m) - 1
    else:
        m = 9
        taps = (1 << 4)
        length = (1 << m) - 1
    
    state = (1 << m) - 1
    seq = []
    for i in range(length):
        bit = state & 1
        seq.append(1 if bit == 1 else -1)
        fb = 0
        for t in range(m):
            if (taps >> t) & 1:
                fb ^= (state >> t) & 1
        state = (state >> 1) | (fb << (m - 1))
    
    return np.array(seq, dtype=np.float64)

def construct():
    """Construct a signed step function (sequence of ±1) to minimize C3 upper bound."""
    deadline = time.time() + 115.0  # Safety margin
    
    # Initialize with m-sequence
    mseq = _generate_m_sequence(9)
    x = mseq[:N].copy()
    
    current_R = _compute_autocorr(x)
    current_metric = _compute_c3_bound(current_R)
    best_metric = current_metric
    best_x = x.copy()
    best_R = current_R.copy()
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    last_improve_time = time.time()
    no_improve_limit = 5.0 # seconds
    
    while time.time() < deadline:
        pos = int(rng.integers(0, N))
        old_val = x[pos]
        x[pos] = -old_val
        
        R_candidate = _fast_flip_autocorr_update(current_R, x, pos)
        metric_candidate = _compute_c3_bound(R_candidate)
        
        delta_E = metric_candidate - current_metric
        
        accept = False
        if delta_E < 0:
            accept = True
        else:
            if T > T_min:
                # Avoid overflow/underflow issues with exp
                prob = math.exp(-delta_E / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            current_R = R_candidate
            current_metric = metric_candidate
            if current_metric < best_metric:
                best_metric = current_metric
                best_x = x.copy()
                best_R = current_R.copy()
                last_improve_time = time.time()
        else:
            # Revert
            x[pos] = old_val
        
        # Cooling
        T *= cooling_rate
        
        # Reheating strategy
        if time.time() - last_improve_time > no_improve_limit:
            T = max(T, 2.0)
            
    return best_x
