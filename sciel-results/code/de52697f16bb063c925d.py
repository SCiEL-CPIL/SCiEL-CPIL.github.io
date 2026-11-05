import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def h_from_spectrum(mags):
        """
        Reconstruct h from spectral magnitudes.
        mags: array of length N//2 + 1 (DC to Nyquist)
        We assign random phases (fixed for the duration of the optimization 
        to maintain smoothness in the search space, or re-randomized occasionally).
        Actually, for SA, we fix the phases to a random set at the start 
        and only optimize magnitudes. This reduces dimensionality and 
        maintains the real-valued constraint naturally.
        """
        # mags[0] is DC, mags[N//2] is Nyquist
        # We need to construct the full FFT H
        H = np.zeros(N, dtype=complex)
        
        # DC component
        H[0] = mags[0] + 0j
        
        # Nyquist component (if N is even)
        if N % 2 == 0:
            H[N//2] = mags[N//2] + 0j
        
        # Positive frequencies
        # mags[1 : N//2] correspond to k = 1 .. N//2 - 1
        # We use fixed random phases for these
        phases = np.random.default_rng(42).uniform(0, 2*np.pi, size=N//2 - 1)
        H[1:N//2] = mags[1:N//2] * np.exp(1j * phases)
        
        # Negative frequencies (Hermitian symmetry)
        # H[N-k] = conj(H[k])
        # k goes from 1 to N//2 - 1
        # Indices: N-1 down to N-(N//2-1)
        if N % 2 == 0:
            # H[N-1 : N//2+1 : -1] corresponds to k=1..N//2-1
            H[N-1 : N//2 : -1] = np.conj(H[1:N//2])
        else:
            # N is even here (400), so this block is technically unreachable 
            # but good for robustness if N changes.
            H[N-1 : (N+1)//2 : -1] = np.conj(H[1:(N+1)//2])
            
        h = np.fft.ifft(H).real
        return h

    # Initialize with a "flat" spectrum (white noise in time) 
    # or a "smooth" spectrum (sinc-like in time).
    # A flat spectrum (mags all 1) gives h[0]=1, others 0 (impulse).
    # An impulse has conv[0]=1, max=1, sum=1, val = 2*N/1 = 800. Bad.
    # We want low peak in conv. Conv is related to |FFT|^2.
    # max(conv) is roughly the peak of the inverse FFT of |H|^2.
    # To minimize peak of ifft(|H|^2), we want |H|^2 to be as flat as possible.
    # So we start with mags all equal.
    
    mags = np.ones(N//2 + 1)
    # Scale doesn't matter, but let's keep it normalized
    h_curr = h_from_spectrum(mags)
    # Ensure sum is not zero
    if abs(np.sum(h_curr)) < 1e-6:
         h_curr = np.ones(N)
         mags = np.abs(np.fft.fft(h_curr))[:N//2+1]
    
    curr_val = compute_value(h_curr)
    best_h = h_curr.copy()
    best_val = curr_val
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.999
    
    # We optimize the log-magnitudes to ensure positivity
    log_mags = np.log(mags + 1e-10)
    
    steps = 0
    max_steps = 50000 # Limit to ensure we don't run out of time
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Propose a change: pick a random frequency bin and perturb its log-magnitude
        k = np.random.randint(0, len(log_mags))
        delta = np.random.normal(0, 0.1)
        
        old_log_mag = log_mags[k]
        log_mags[k] = old_log_mag + delta
        
        new_mags = np.exp(log_mags)
        new_h = h_from_spectrum(new_mags)
        
        new_sum = np.sum(new_h)
        if abs(new_sum) < 1e-6:
            # Reject if sum is too small
            log_mags[k] = old_log_mag
            continue
            
        new_val = compute_value(new_h)
        
        # Acceptance criterion
        if new_val < curr_val:
            # Better, accept
            curr_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = new_h.copy()
                # Optional: print progress
                if best_val < 1.5:
                    print(f"SA Improvement: {best_val:.6f} at step {steps}", file=sys.stderr)
        else:
            # Worse, accept with probability exp(-delta_E / T)
            delta_E = new_val - curr_val
            if np.random.rand() < np.exp(-delta_E / T):
                curr_val = new_val
            else:
                # Reject
                log_mags[k] = old_log_mag
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If stuck at T_min, we might want to restart with a higher T or break
            # For simplicity, just break if we've cooled enough and no recent improvement
            # But let's just keep running until time limit or max steps.
            # Actually, if T is very small, we are doing greedy hill climbing.
            # Let's add a check: if no improvement in last 1000 steps and T < 0.1, restart T?
            # Simpler: just let it run. The random restarts of phases are not done, 
            # so we are stuck in a local min of this phase configuration.
            # To explore more, we could re-randomize phases occasionally.
            if steps % 500 == 0 and T < 0.1:
                 # Re-randomize phases to jump to a new basin
                 # This is a "kick"
                 pass # Skipping complex kick logic to keep code simple and robust.
                 # The initial random phase set is fixed. This is a limitation.
                 # A better strategy would be to optimize phases too, but that's 200 more vars.
                 # Given the time budget, optimizing 200 log-mags is faster per step.

    # Final Polish: Local gradient descent on log-mags?
    # SA is usually good enough if run long. 
    # Let's try a quick local refinement using Nelder-Mead on a subset of mags? 
    # No, let's just use the best found.
    
    # Safety check
    if time.time() > deadline - 1.0:
        return best_h

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)

if __name__ == "__main__":
    # Quick self-test
    h = construct()
    print(f"Resulting value: {compute_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Max conv: {np.max(np.convolve(h, h))}")
