import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    rng = np.random.default_rng(42)
    
    # Initialize with random signs
    x = rng.choice([-1, 1], size=N)
    
    def calc_cost(x):
        c = np.correlate(x, x, mode='full')
        pos_corr = c[N:] 
        return np.sum(np.abs(pos_corr)**3)

    current_cost = calc_cost(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # Estimate typical cost difference for temperature scaling
    deltas = []
    for _ in range(50):
        idx = rng.integers(0, N)
        old_val = x[idx]
        x[idx] = -old_val
        new_cost = calc_cost(x)
        deltas.append(abs(new_cost - current_cost))
        x[idx] = old_val
        
    mean_delta = np.mean(deltas)
    T_start = mean_delta * 2.0 if mean_delta > 0 else 1000.0
    T_end = T_start * 1e-6
    alpha = 0.9999
    n_iter = 400000
    
    T = T_start
    for i in range(n_iter):
        idx = rng.integers(0, N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_cost = calc_cost(x)
        delta = new_cost - current_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x.copy()
        else:
            x[idx] = old_val
            
        T = max(T * alpha, T_end)
        
    return best_x
# EVOLVE-BLOCK-END
