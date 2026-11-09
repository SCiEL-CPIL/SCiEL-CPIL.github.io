import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initialize with random signs
    x = np.random.choice([-1, 1], N)
    
    # Function to compute objective J = sum_{k=1}^{N-1} |R(k)|
    def compute_J(x):
        # Use FFT for initial calculation or full recalculation
        # Zero-pad to 2*N to avoid circular convolution issues
        f = np.fft.fft(x, 2*N)
        r = np.fft.ifft(f * np.conj(f)).real
        # r[0] is R(0), r[1..N-1] are R(1)..R(N-1)
        # r[N..] are negative lags
        return np.sum(np.abs(r[1:N]))

    # Current best
    J = compute_J(x)
    best_x = x.copy()
    best_J = J
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.999
    
    # Precompute indices for efficiency? 
    # The delta calculation:
    # For flip at i:
    # delta_R[k] = -2 * x[i] * (x[i+k] + x[i-k])
    # We need sum_{k=1}^{N-1} |R[k] + delta_R[k]| - |R[k]|
    
    # To speed up, we can maintain R explicitly.
    # Let's recompute R initially using correlation
    R = np.correlate(x, x, mode='full')
    # R is length 2N-1. 
    # R[N-1] is lag 0.
    # R[N] is lag 1 ... R[2N-2] is lag N-1.
    # R[N-2] is lag -1 ... R[0] is lag -(N-1).
    # We want lags 1 to N-1.
    # So R_lags = np.concatenate([R[N-2:0:-1], R[N:]]) ?
    # Actually, autocorrelation is symmetric. R(k) = R(-k).
    # So we just need R[1:N] from the one-sided correlation?
    # np.correlate(x, x, 'full') returns lags from -(N-1) to N-1.
    # Index N-1 is lag 0.
    # Indices N to 2N-2 are lags 1 to N-1.
    # Indices N-2 down to 0 are lags -1 to -(N-1).
    # Since we sum absolute values, and R is symmetric, 
    # sum_{k=1}^{N-1} |R(k)| is half of sum_{k != 0} |R(k)|.
    # Or we can just use the positive lags.
    
    # Let's store R_pos = R[N : 2*N-1] (lags 1..N-1)
    R_pos = R[N:2*N-1].copy()
    J = np.sum(np.abs(R_pos))
    
    # Annealing loop
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Calculate delta R
        # delta_R[k] for k=1..N-1
        # term1: x[i+k] for k=1..N-1-i (if i+k < N)
        # term2: x[i-k] for k=1..N-1 (if i-k >= 0)
        
        # Create a vector of zeros for delta_R
        # It's faster to compute the contribution directly
        # delta_R = -2 * x[i] * (shifted_x_forward + shifted_x_backward)
        
        # Forward part: x[i+1 : N] corresponds to k=1..N-1-i
        # Backward part: x[i-1 : -1 : -1] corresponds to k=1..i
        
        # Let's construct the full vector of (x[i+k] + x[i-k]) for k=1..N-1
        # This is tricky with slices.
        # Alternative: Just recompute J? 
        # N=400. Recomputing J via FFT is ~10us. 
        # Looping 100,000 times takes 1s.
        # Let's try simple recomputation first, it's safer and less bug-prone.
        
        x_old = x[i]
        x[i] = -x[i]
        
        # Recompute R_pos
        # Using FFT is fast
        f = np.fft.fft(x, 2*N)
        r = np.fft.ifft(f * np.conj(f)).real
        new_R_pos = r[1:N]
        new_J = np.sum(np.abs(new_R_pos))
        
        delta_J = new_J - J
        
        if delta_J < 0 or np.random.rand() < np.exp(-delta_J / T):
            J = new_J
            R_pos = new_R_pos
            if J < best_J:
                best_J = J
                best_x = x.copy()
        else:
            x[i] = x_old
            
        T *= cooling
        if T < 1e-6:
            break
            
    return best_x
