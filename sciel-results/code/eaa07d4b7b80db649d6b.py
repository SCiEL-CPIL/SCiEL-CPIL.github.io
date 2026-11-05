"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_value(h):
    """Compute the objective value using FFT-based convolution."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _eval_value_binary(b, pos_sum, neg_sum):
    """
    Evaluate the objective for a binary sequence b (0/1).
    h[i] = pos_sum if b[i]==1 else neg_sum
    """
    n_pos = np.sum(b)
    n_neg = N - n_pos
    h = np.where(b == 1, pos_sum, neg_sum)
    s = n_pos * pos_sum + n_neg * neg_sum
    if abs(s) < 1e-12:
        return np.inf
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_binary(b, pos_sum, neg_sum):
    """
    Objective for simulated annealing: minimize max autoconvolution.
    We work with the unnormalized peak, then normalize at the end.
    Actually, let's just compute the full value.
    """
    n_pos = np.sum(b)
    n_neg = N - n_pos
    # Fast computation of sum
    s = n_pos * pos_sum + n_neg * neg_sum
    if abs(s) < 1e-12:
        return 1e10
    
    # Fast computation of max conv
    # conv[k] = sum_i h[i] h[i+k]
    # For binary h with values p and n:
    # h[i] = p * b[i] + n * (1-b[i]) = (p-n) * b[i] + n
    # Let d = p - n, c = n
    # h[i] = d * b[i] + c
    # conv[k] = sum_i (d*b[i]+c)(d*b[i+k]+c)
    #          = d^2 * sum_i b[i]b[i+k] + d*c * (sum_i b[i] + sum_i b[i+k]) + c^2 * N
    # Note: sum_i b[i+k] is just sum_i b[i] for the valid range, but we need to be careful with boundaries.
    # Actually, np.convolve does full convolution.
    # Let's stick to FFT for correctness, but we can optimize the check.
    h = np.where(b == 1, pos_sum, neg_sum)
    conv = _convolve_fft(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Simulated Annealing on a binary sequence to find low-peak autocorrelation.
    """
    deadline = time.time() + 145.0  # Safety margin
    
    # Initial best: all ones
    best_h = np.ones(N)
    best_val = 2.0
    
    # We will search over binary sequences b in {0,1}^N
    # h[i] = A if b[i]=1 else -B (or similar)
    # Let's fix the values of h to be symmetric for simplicity? 
    # Or let's allow h[i] to be from a small set of values.
    # The prompt suggests "binary sequence optimized via Simulated Annealing".
    # This implies h[i] takes two values.
    # Let h[i] = u if b[i]=1 else v.
    # We need sum(h) != 0.
    
    # Let's try a few fixed pairs (u, v) and optimize b.
    # Or, we can let u and v be part of the search?
    # The directive says "Represent the step heights as a binary sequence".
    # This implies the values are fixed or derived from the binary choice.
    # Let's assume h[i] = 1 if b[i]=1 else -1.
    # Then sum(h) = n_pos - n_neg = 2*n_pos - N.
    # If N=400, sum(h) can be 0 if n_pos=200.
    # We must avoid sum(h)=0.
    
    # Let's try h[i] in {1, -1}.
    # Objective: minimize |2 * N * max(conv) / sum(h)^2|
    
    # Simulated Annealing on b
    # State: b (binary array)
    # Energy: value(b)
    
    # Initialize b
    # A good starting point for low peak autocorrelation is a random sequence or a specific pattern.
    # Random is a good start for SA.
    
    rng = np.random.RandomState(42)
    
    # We will try a few different value pairs (u, v) to see which yields better results.
    # Pair 1: (1, -1)
    # Pair 2: (1, -0.5) -> sum = n_pos - 0.5*n_neg
    # Pair 3: (1, -2)
    
    value_pairs = [(1.0, -1.0), (1.0, -0.5), (1.0, -2.0), (2.0, -1.0), (1.0, -0.1)]
    
    for u, v in value_pairs:
        if time.time() > deadline - 20:
            break
            
        # Initialize b randomly
        b = rng.randint(0, 2, size=N)
        
        # Ensure sum(h) != 0
        # sum(h) = n_pos * u + n_neg * v
        # If this is close to 0, flip a bit.
        
        current_val = _eval_value_binary(b, u, v)
        
        if np.isinf(current_val):
            # Flip a bit to make sum non-zero
            b[0] = 1 - b[0]
            current_val = _eval_value_binary(b, u, v)
        
        if current_val < best_val:
            best_val = current_val
            best_h = np.where(b == 1, u, v)
            
        # Simulated Annealing
        T_start = 1.0
        T_end = 0.01
        # Number of steps depends on time available
        # Each step is fast (just one FFT? No, we can update conv incrementally? 
        # No, FFT is O(N log N), N=400 is very fast. ~10us.
        # We can do ~100k steps in 1 second.
        # Let's do as many as possible within the time budget.
        
        max_time_per_pair = 20.0
        start_time = time.time()
        
        while time.time() < start_time + max_time_per_pair:
            # Pick a random index to flip
            idx = rng.randint(N)
            
            old_bit = b[idx]
            new_bit = 1 - old_bit
            
            # Calculate new value
            # We can do this incrementally?
            # conv_new = conv_old + delta
            # delta depends on the change in h[idx].
            # h[idx] changes by (new_val - old_val).
            # This affects conv[k] for all k.
            # It's easier to just recompute if N is small.
            # Let's try recomputing.
            
            b[idx] = new_bit
            new_val = _eval_value_binary(b, u, v)
            
            if np.isinf(new_val):
                # Revert
                b[idx] = old_bit
                continue
                
            delta_E = new_val - current_val
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = np.where(b == 1, u, v)
            else:
                # Revert
                b[idx] = old_bit
            
            # Update temperature
            # Linear cooling? Geometric?
            # Let's use geometric cooling.
            T = T_start * (T_end / T_start) ** ((time.time() - start_time) / max_time_per_pair)
            if T < T_end:
                T = T_end

    # Final polish: try to improve best_h with a few random flips
    if time.time() < deadline - 5:
        # Identify the u, v used for best_h
        # We can just try flipping bits in best_h directly, assuming it's binary {u,v}
        # But best_h might not be strictly binary if we mixed pairs? 
        # In the loop above, best_h is always from a specific pair.
        # Let's assume the last best_h is from the last pair that improved it, or we can just
        # re-run SA on the best found configuration.
        
        # Let's just do a few random perturbations on best_h
        # But best_h values are fixed to {u,v}. 
        # We need to know u and v.
        # Let's just skip this if we don't track u,v for best_h.
        # Or, we can just return best_h.
        
        pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h
# EVOLVE-BLOCK-END
