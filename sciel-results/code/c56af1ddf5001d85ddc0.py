"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation coefficients.
    Lower is better.
    """
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    # Exclude the DC term (index 0) which is constant N
    # Actually, including it is fine since it's constant, but excluding it saves a tiny bit of computation?
    # No, np.sum is fast.
    score = np.sum(np.abs(autocorr)**3)
    return score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses Simulated Annealing to minimize the L3 norm of the circular autocorrelation.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initial random solution
    current = np.random.choice([-1, 1], size=N)
    current_score = _compute_score(current)
    
    best = current.copy()
    best_score = current_score
    
    # SA Parameters
    T = 1000.0
    T_min = 0.01
    alpha = 0.9995  # Cooling rate
    
    while T > T_min:
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor: flip a random bit
        neighbor = current.copy()
        idx = np.random.randint(0, N)
        neighbor[idx] *= -1
        
        neighbor_score = _compute_score(neighbor)
        
        delta = neighbor_score - current_score
        
        if delta < 0:
            # Accept improvement
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
        else:
            # Accept worse with probability exp(-delta/T)
            if np.random.random() < np.exp(-delta / T):
                current = neighbor
                current_score = neighbor_score
        
        # Cool down
        T *= alpha
        
    return best
# EVOLVE-BLOCK-END
