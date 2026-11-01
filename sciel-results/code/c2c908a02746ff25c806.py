import numpy as np
import time

N = 400

def _compute_energy(R):
    """Compute the L3 norm of the autocorrelation vector."""
    return np.sum(np.abs(R)**3)

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing with incremental autocorrelation updates.
    """
    # Start with a random sequence
    s = np.random.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation
    # R[k] = sum_i s[i] s[(i+k)%N]
    # Using FFT is O(N log N), but for initialization it's fine.
    # Or just use the property that for binary, we can compute it directly or via FFT.
    # Let's use FFT for initial computation to ensure correctness, then switch to incremental.
    # Actually, incremental requires initial R.
    # R = ifft(fft(s) * conj(fft(s))).real
    fft_s = np.fft.fft(s)
    R = np.fft.ifft(fft_s * np.conj(fft_s)).real
    
    current_energy = _compute_energy(R)
    best_energy = current_energy
    best_s = s.copy()
    best_R = R.copy()
    
    # Simulated Annealing parameters
    start_temp = 10.0
    end_temp = 0.01
    # We have ~110 seconds.
    # Each step is O(N) for update + O(N) for energy diff?
    # Energy diff: We only need to update the terms in the sum that changed.
    # The terms that change are R[k] for all k?
    # Delta R[k] is non-zero for all k?
    # Delta R[k] = -2 s_i (s_{i+k} + s_{i-k}).
    # This is non-zero for all k generally.
    # So we have to update all R[k] and recompute the sum?
    # Sum of cubes: E = sum R[k]^3.
    # New E = sum (R[k] + dR[k])^3.
    # This is O(N) to compute the new sum if we iterate.
    # So one step is O(N).
    # N=400. 110s / 110s = 1 step per sec? No, Python is slow.
    # Let's estimate. 100,000 steps in 10s?
    # Let's just run as many steps as possible within the time limit.
    
    start_time = time.time()
    max_time = 110.0
    
    # Cooling schedule
    # Linear cooling? Exponential?
    # Let's use exponential cooling.
    # T = start_temp * (end_temp/start_temp)^(t/total_steps)
    # We don't know total steps. Let's use a fixed number of steps or time-based.
    # Let's aim for ~200,000 steps if possible.
    # If it's too slow, we adjust.
    
    # Optimization: 
    # Instead of recalculating sum(R^3) from scratch every time, we can maintain it.
    # E_new = E_old - sum(R_old[k]^3) + sum(R_new[k]^3)
    # This is still O(N) to compute the sums.
    
    # Let's try to be efficient.
    # Precompute indices?
    
    step = 0
    while time.time() - start_time < max_time:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Calculate change in R
        # dR[k] = -2 * s[i] * (s[(i+k)%N] + s[(i-k)%N])
        # Note: s[i] is the current value.
        si = s[i]
        
        # Vectorized calculation of dR
        # k ranges from 0 to N-1
        k = np.arange(N)
        idx_plus = (i + k) % N
        idx_minus = (i - k) % N
        
        dR = -2 * si * (s[idx_plus] + s[idx_minus])
        
        # Calculate change in energy
        # E = sum |R|^3. Since R is real, |R|^3 = R^3 if R>0, -R^3 if R<0? No, |R|^3 = (abs(R))^3.
        # Actually R can be negative.
        # Let's use np.abs(R)**3.
        
        R_new = R + dR
        
        # To save time, we can just compute the energy difference.
        # delta_E = sum(|R_new|^3) - sum(|R|^3)
        # This requires two O(N) operations.
        
        # Optimization: 
        # We can compute sum(|R_new|^3) directly.
        new_energy = np.sum(np.abs(R_new)**3)
        delta_E = new_energy - current_energy
        
        # Acceptance probability
        if delta_E < 0:
            accept = True
        else:
            # T needs to be defined.
            # Let's use a simple cooling schedule based on step count.
            # Assume ~100,000 steps.
            # T = start_temp * exp(-step / 1000)
            T = start_temp * np.exp(-step / 5000.0)
            if T > 1e-9:
                accept_prob = np.exp(-delta_E / T)
                accept = (np.random.random() < accept_prob)
            else:
                accept = False
        
        if accept:
            # Apply flip
            s[i] = -si
            R = R_new
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_s = s.copy()
                best_R = R.copy()
                
        step += 1
        
    return best_s
