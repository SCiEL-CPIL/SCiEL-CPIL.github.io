import numpy as np

N = 100

def construct():
    # New approach: Local Search / Simulated Annealing
    # Objective: Maximize minimum circular autocorrelation C_k
    # which minimizes maximum overlap with 1-h
    
    def get_correlations(h):
        F = np.fft.fft(h)
        power = F * np.conj(F)
        C = np.fft.ifft(power).real
        return C

    def objective(h):
        C = get_correlations(h)
        # We want to maximize min(C), so minimize -min(C)
        return -np.min(C)

    # Initialize with a diverse pattern, e.g. sine wave
    # h_i = 0.5 + A * sin(2*pi*i/P)
    # We need sum N/2. Mean must be 0.5.
    # Clip to [0,1] and renormalize.
    
    def normalize(h):
        h = np.clip(h, 0, 1)
        diff = N / 2.0 - np.sum(h)
        if diff > 0:
            # Add to smallest
            idx = np.argsort(h)
            for i in idx:
                add = min(diff, 1.0 - h[i])
                h[i] += add
                diff -= add
                if diff <= 1e-9:
                    break
        else:
            # Subtract from largest
            idx = np.argsort(h)[::-1]
            for i in idx:
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
                if diff >= -1e-9:
                    break
        # Final check
        h[h < 0] = 0
        h[h > 1] = 1
        return h

    # Initial guess: Sine wave with period 5 or 7
    t = np.linspace(0, 2 * np.pi * 5, N, endpoint=False)
    h0 = 0.5 + 0.4 * np.sin(t)
    h = normalize(h0)
    
    best_h = h.copy()
    best_obj = objective(best_h)
    
    # Local Search
    step_size = 0.1
    np.random.seed(123)
    for it in range(20000):
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        delta = np.random.uniform(-step_size, step_size)
        
        h_new = best_h.copy()
        h_new[i] += delta
        h_new[j] -= delta
        
        if h_new[i] < 0 or h_new[i] > 1 or h_new[j] < 0 or h_new[j] > 1:
            continue
            
        obj_new = objective(h_new)
        if obj_new < best_obj:
            best_h = h_new
            best_obj = obj_new
            step_size *= 0.995
        else:
            # Restart step size occasionally if stuck?
            if it % 1000 == 0:
                step_size = 0.05
                
    return best_h
