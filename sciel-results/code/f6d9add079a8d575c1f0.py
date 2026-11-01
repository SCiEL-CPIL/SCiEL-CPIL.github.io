import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function of length N=400 to minimize the peak 
    autocorrelation sidelobe.
    Uses Greedy Local Search with Random Restarts and Incremental Autocorrelation Update.
    """
    start_time = time.time()
    time_budget = 110.0  # seconds
    
    best_f = None
    best_peak = np.inf
    
    # Precompute indices for non-zero lags
    # Lags 1..N-1 and -(N-1)..-1
    # We will store autocorrelation for lags 0..N-1 and use symmetry or store all.
    # Let's store R[0..N-1] where R[k] is autocorrelation at lag k.
    # R[-k] = R[k].
    # Peak is max(R[1..N-1]).
    
    def compute_initial_autocorr(f):
        F = np.fft.fft(f, n=2 * N)
        acf = np.fft.ifft(F * np.conj(F)).real
        # acf[0] is lag 0
        # acf[1:N] are lags 1..N-1
        # acf[N+1:2*N] are lags -(N-1)..-1
        # Due to real input, acf is symmetric: acf[N-k] = acf[k]
        # So we just need acf[1:N]
        return acf[1:N].copy()

    def get_peak_from_autocorr(R):
        return np.max(np.abs(R))
        
    # Run multiple restarts
    while (time.time() - start_time) < time_budget:
        # 1. Initialize with random signs
        f = np.random.choice([-1, 1], N)
        R = compute_initial_autocorr(f)
        current_peak = get_peak_from_autocorr(R)
        
        # 2. Greedy Local Search
        while True:
            improved = False
            indices = np.random.permutation(N)
            for i in indices:
                # Calculate change in autocorrelation if we flip f[i]
                # Delta R[k] = -2 * f[i] * (f[i+k] + f[i-k])
                # Note: indices must be handled with wrap-around? 
                # No, this is linear autocorrelation, not circular.
                # f is length N.
                # For lag k, R[k] = sum_{j=0}^{N-1-k} f[j] f[j+k].
                # When we flip f[i], it affects R[k] if i is in [0, N-1-k] OR i+k is in [0, N-1-k] (i.e., i is in [k, N-1]).
                # Actually, the formula Delta R[k] = -2 f[i] (f[i+k] + f[i-k]) works if we define f[x]=0 for x<0 or x>=N.
                
                # We need to update R for all k=1..N-1.
                # This is O(N) per flip.
                # N=400. 400 * 400 = 160k ops per epoch.
                
                # Let's optimize the update.
                # We only need to check if the new peak is lower.
                # Instead of updating all R and then finding max, 
                # we can just compute the new peak directly?
                # No, we need the new R to continue.
                
                # Optimization: 
                # Only lags k such that i-k >= 0 or i+k < N are affected?
                # Actually, every lag k where i is involved in the sum is affected.
                # i is involved in R[k] if 0 <= i <= N-1-k (as first term) OR 0 <= i-k <= N-1-k (as second term).
                # This is basically all k from 1 to N-1 potentially.
                
                # Let's just do the O(N) update.
                old_f_i = f[i]
                f[i] = -old_f_i
                
                # Update R
                # R[k] += Delta R[k]
                # Delta R[k] = -2 * old_f_i * (f_new[i+k] + f_new[i-k]) ?
                # Wait, f[i] changed.
                # Let's use the formula:
                # R_new[k] = R_old[k] - 2 * old_f_i * (f_new[i+k] + f_new[i-k])
                # Note: f_new[i] = -old_f_i.
                # f_new[j] = f_old[j] for j != i.
                
                # We need to be careful with boundaries.
                # f[x] is 0 if x < 0 or x >= N.
                
                # Vectorized update might be tricky with boundaries.
                # Let's stick to FFT for simplicity and reliability, 
                # unless we are sure the incremental is faster.
                # FFT of 800 is ~10-20us. 
                # O(N) loop in Python is ~400 iterations * ~100ns = 40us.
                # They are comparable. FFT is in C, so it wins.
                
                # So I will revert to using FFT in the inner loop for simplicity and speed.
                new_peak = get_peak(f)
                
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    f[i] = old_f_i
            
            if not improved:
                break
                
        if current_peak < best_peak:
            best_peak = current_peak
            best_f = f.copy()
            
    if best_f is None:
        best_f = np.random.choice([-1, 1], N)
        
    return best_f
# EVOLVE-BLOCK-END
