import numpy as np
import time

N = 400

def _peak_autocorr(x):
    # Linear autocorrelation
    # x: array of +/- 1
    ac = np.correlate(x, x, mode='full')
    # Exclude lag 0 (index N-1)
    # Lags 1..N-1 correspond to indices N..2N-2 and 0..N-2
    # We take max abs of these
    # Note: ac is symmetric, so we can just take one side
    # ac[N:] corresponds to lags 0..N-1? 
    # Let's check indices.
    # x = [a, b, c]
    # correlate(x, x, 'full')
    # c*c, b*c+a*b, a*a+b*b+c*c, a*b+b*c, a*c
    # Indices: 0, 1, 2, 3, 4
    # N=3. Center is 2.
    # Lag 0: index 2.
    # Lag 1: index 1 and 3.
    # Lag 2: index 0 and 4.
    # So we want max(abs(ac[0:N-1]), abs(ac[N:2N-1]))
    # ac[:N-1] and ac[N:]
    side1 = ac[:N-1]
    side2 = ac[N:]
    peak = np.max(np.abs(np.concatenate((side1, side2))))
    return peak

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initialization
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    x = rng.choice([-1, 1], size=N)
    
    current_peak = _peak_autocorr(x)
    best_peak = current_peak
    best_x = x.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    
    # Run SA until time limit or T is low
    while T > 0.05:
        if time.time() > deadline:
            break
            
        i = rng.integers(0, N)
        old_val = x[i]
        x[i] = -old_val
        
        new_peak = _peak_autocorr(x)
        delta = new_peak - current_peak
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (delta > 0 and rng.random() < np.exp(-delta / T)):
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_x = x.copy()
        else:
            x[i] = old_val
            
        T *= cooling
        
    # Local Search Polish
    # Try flipping each bit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            old_val = x[i]
            x[i] = -old_val
            new_peak = _peak_autocorr(x)
            if new_peak < current_peak:
                current_peak = new_peak
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
                
    return best_x
