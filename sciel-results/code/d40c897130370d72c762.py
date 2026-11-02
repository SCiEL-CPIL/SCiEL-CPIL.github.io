import numpy as np

N = 100

def construct():
    # Generate Thue-Morse sequence
    # Thue-Morse has low autocorrelation and balanced properties
    # Map to [0, 1] with mean 0.5
    tm = np.zeros(N, dtype=float)
    # Thue-Morse: t_n = popcount(n) % 2
    for i in range(N):
        tm[i] = 1.0 if bin(i).count('1') % 2 == 0 else 0.0
    
    # Center to [0, 1] with mean 0.5
    # Thue-Morse has equal 0s and 1s for N power of 2, close for others
    # Adjust to ensure sum is exactly N/2
    # Current sum is approx N/2
    current_sum = np.sum(tm)
    target_sum = N / 2.0
    
    # If sum is not exactly target, adjust the last few elements
    # This ensures validity
    diff = target_sum - current_sum
    if diff > 0:
        # Add to zeros
        indices = np.where(tm == 0)[0]
        for idx in indices:
            if diff <= 0: break
            add = min(diff, 1.0 - tm[idx])
            tm[idx] += add
            diff -= add
    elif diff < 0:
        # Subtract from ones
        indices = np.where(tm == 1)[0]
        for idx in indices:
            if diff >= 0: break
            sub = min(-diff, tm[idx])
            tm[idx] -= sub
            diff += sub
            
    return tm
