import numpy as np
import time

N = 400

def _compute_max_autocorr(s):
    """
    Computes the maximum absolute linear autocorrelation for k != 0.
    s: array of 1s and -1s of length N.
    """
    # Pad to 2N for linear convolution via FFT
    s_padded = np.zeros(2 * N)
    s_padded[:N] = s
    
    # FFT
    S = np.fft.fft(s_padded)
    # Power spectrum
    P = S * np.conj(S)
    # Inverse FFT to get autocorrelation
    # The result is real, but FFT returns complex
    R = np.fft.ifft(P).real
    
    # R[0] is N. R[1:N] are R(1) to R(N-1).
    # R[N:2N] are negative lags, symmetric.
    # We only need max absolute value for k in 1..N-1
    # Note: R is circularly shifted in some FFT implementations?
    # np.fft.ifft returns the sequence in natural order for linear convolution 
    # if we padded correctly?
    # Let's verify.
    # Convolution of [1,0,0] and [1,0,0] is [1,0,0,0].
    # FFT [1,0,0] -> [1,1,1]. P=[1,1,1]. IFFT [1,0,0,0]? No, IFFT of [1,1,1] is [1,0,0] (scaled).
    # Actually, np.fft.ifft(np.fft.fft(x)*np.conj(np.fft.fft(x))) gives the circular autocorrelation.
    # If we zero-pad to 2N, the circular autocorrelation of the padded signal 
    # corresponds to the linear autocorrelation of the original signal in the first N samples.
    
    # R[0] = sum s_i^2 = N
    # R[k] = sum s_i s_{i+k} for k=1..N-1
    # R[N-k] = sum s_i s_{i-N+k} ... which is the negative lag.
    # Due to symmetry R[k] = R[-k].
    # So we just take max(abs(R[1:N]))
    
    if N <= 1:
        return 0.0
        
    max_val = np.max(np.abs(R[1:N]))
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_seq = None
    best_score = float('inf')
    
    # Seed program returns np.ones(N)
    # Score of seed is based on its autocorrelation.
    # We don't need to calculate the exact seed score for the return value, 
    # just the sequence. The evaluator calculates the score.
    # However, we want to minimize the autocorrelation.
    
    # Initial random sequences
    # We will use Simulated Annealing
    
    # Temperature parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.999
    
    # Number of restarts
    max_restarts = 50
    
    for restart in range(max_restarts):
        # Check time
        if time.time() > deadline:
            break
            
        # Initialize random sequence
        s = np.random.choice([-1, 1], size=N)
        
        current_score = _compute_max_autocorr(s)
        
        T = T_start
        step = 0
        max_steps = 5000 # Adjust based on speed
        
        while T > T_end:
            if time.time() > deadline:
                break
                
            # Pick a random index to flip
            idx = np.random.randint(N)
            
            # Flip
            s[idx] *= -1
            
            # Calculate new score
            new_score = _compute_max_autocorr(s)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
                # If this is the global best, update
                if current_score < best_score:
                    best_score = current_score
                    best_seq = s.copy()
            else:
                # Reject, flip back
                s[idx] *= -1
                
            T *= cooling_rate
            step += 1
            
            # Early stopping if we hit a very low value?
            # Theoretical lower bound for max autocorrelation is not 0.
            # For N=400, random is ~20. Good sequences might be ~10-15.
            # If we get below 10, it's very good.
            if current_score < 5.0:
                break

    # Fallback to seed if no improvement (unlikely)
    if best_seq is None:
        return np.ones(N)
        
    return best_seq
