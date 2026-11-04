import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with random +/- 1
    h = np.random.choice([-1, 1], N)
    
    # Function to compute linear autocorrelation
    def get_autocorr(h):
        # Use FFT for initial calculation
        fft_size = 1
        while fft_size < 2 * N:
            fft_size *= 2
        H = np.fft.rfft(h, n=fft_size)
        acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
        # Linear autocorrelation R[k] for k=0..N-1
        # acf from irfft is circular. For linear, we need to adjust?
        # Actually, for zero-padded FFT, irfft(|H|^2) gives the linear autocorrelation
        # in the first N samples if padded correctly?
        # Let's verify.
        # Convolution of h with h[::-1] is correlation.
        # h is length N. h[::-1] is length N.
        # Zero pad to 2N.
        # FFT size 2N is sufficient.
        # acf[k] for k=0..N-1 is the linear autocorrelation.
        return acf[:N]

    R = get_autocorr(h)
    current_psl = np.max(np.abs(R[1:]))
    best_psl = current_psl
    best_h = h.copy()
    
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        # Iterate through indices
        # To avoid bias, we can randomize the order or just sweep
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = h[i]
            new_val = -old_val
            
            # Update R incrementally
            # Delta R[k] = -2 * old_val * (h[i+k] + h[i-k])
            # We need to update R in place
            
            # Save old R to check improvement? 
            # We can compute new PSL after update.
            # But we need to revert if it gets worse.
            
            # Optimization: Only update R[k] for k where h[i+k] or h[i-k] is non-zero.
            # Since h is +/- 1, all are non-zero.
            
            # We will create a copy of R to test? 
            # R is size 400. Copy is cheap.
            R_old = R.copy()
            
            # Apply flip
            h[i] = new_val
            
            # Update R
            for k in range(1, N):
                # Term 1: h[i] * h[i+k] in R[k]
                # Term 2: h[i-k] * h[i] in R[k]
                
                delta = 0.0
                if i + k < N:
                    delta += h[i + k]
                if i - k >= 0:
                    delta += h[i - k]
                
                # The change in R[k] is -2 * old_val * delta
                R[k] += -2.0 * old_val * delta
                # Symmetry R[k] = R[N-k]? 
                # Linear autocorrelation R[k] = sum_{t=0}^{N-1-k} h[t]h[t+k].
                # R[k] is NOT symmetric in the array index k for linear autocorrelation?
                # Wait. R[k] = sum h[t]h[t+k].
                # R[-k] = sum h[t]h[t-k].
                # For real sequences, R[k] = R[-k].
                # But our array R[0..N-1] only stores k>=0.
                # The formula for update derived above:
                # Delta R[k] = -2 h[i] (h[i+k] + h[i-k])
                # This assumes R[k] contains both contributions?
                # Let's check R[k] definition again.
                # R[k] = sum_{t} h[t] h[t+k].
                # Terms involving h[i]:
                # 1. t=i: h[i] h[i+k]. (Requires i+k < N)
                # 2. t+k=i => t=i-k: h[i-k] h[i]. (Requires i-k >= 0)
                # So yes, the update is correct for R[k].
                # Does this update maintain R[k] correctly? Yes.
                
            # Calculate new PSL
            new_psl = np.max(np.abs(R[1:]))
            
            if new_psl < current_psl - 1e-9:
                current_psl = new_psl
                improved = True
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_h = h.copy()
                    no_improve_count = 0
            else:
                # Revert
                h[i] = old_val
                R = R_old
                no_improve_count += 1
        
        if not improved:
            # Restart with new random seed if stuck?
            # Or just break
            if no_improve_count > 1000: # Heuristic
                break
            # Try a few random restarts if time permits
            h = np.random.choice([-1, 1], N)
            R = get_autocorr(h)
            current_psl = np.max(np.abs(R[1:]))
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
            no_improve_count = 0

    return best_h
# EVOLVE-BLOCK-END
