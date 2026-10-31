import numpy as np

N = 100

def construct():
    # Initialize a balanced random binary sequence
    h = np.zeros(N, dtype=int)
    indices = np.random.choice(N, N//2, replace=False)
    h[indices] = 1
    
    def calc_min_autocorr(h):
        # Calculate minimum circular autocorrelation for k=1..N-1
        min_val = float('inf')
        for k in range(1, N):
            # Circular shift
            shifted = np.roll(h, -k)
            ac = np.dot(h, shifted)
            if ac < min_val:
                min_val = ac
        return min_val

    current_min = calc_min_autocorr(h)
    best_h = h.copy()
    best_min = current_min
    
    T = 1.0
    T_min = 0.001
    alpha = 0.995
    
    # Simulated Annealing
    # We want to maximize the minimum autocorrelation
    for _ in range(50000):
        # Pick random i with h[i]=1 and j with h[j]=0
        ones = np.where(h == 1)[0]
        zeros = np.where(h == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        if i == j:
            continue
            
        # Apply swap
        h_new = h.copy()
        h_new[i] = 0
        h_new[j] = 1
        
        new_min = calc_min_autocorr(h_new)
        
        delta = new_min - current_min
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            h = h_new
            current_min = new_min
            
            if current_min > best_min:
                best_min = current_min
                best_h = h.copy()
                
        T *= alpha
        if T < T_min:
            break
            
    return best_h.astype(float)
