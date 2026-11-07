import numpy as np
import time

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    # Ensure non-negative and finite
    h = np.maximum(h, 0.0)
    if np.max(h) < 1e-15:
        return 0.0
    
    # Normalize to max 1 to keep numbers stable, though value is scale invariant
    h_norm = h / np.max(h)
    
    g = np.convolve(h_norm, h_norm)
    m = len(g)
    y = np.concatenate([[0.0], g, [0.0]])
    w = 1.0 / (m + 1)
    
    # Calculate L2 term
    # l2 = sum (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    y_sq = y ** 2
    y_cross = y[:-1] * y[1:]
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + y_cross + y_sq[1:])
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf < 1e-15:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    time_limit = 90.0  # Leave buffer for overhead
    
    # Initial guess: A triangular spike often yields better autocorrelation properties than uniform
    # x in [-0.5, 0.5], peak at 0
    x = np.linspace(-0.5, 0.5, N)
    h_current = np.maximum(1.0 - 2.0 * np.abs(x), 0.0)
    
    # Normalize initial h
    h_current = h_current / np.max(h_current)
    
    best_h = h_current.copy()
    best_val = compute_value(best_h)
    current_h = h_current.copy()
    current_val = best_val
    
    # Simulated Annealing Parameters
    T = 0.1
    T_min = 1e-6
    alpha = 0.9995  # Cooling rate
    
    # We want to maximize value, so we minimize -value
    # But SA is easier to think about with energy = -value
    # Let's just use value directly. Accept new if better, or worse with probability exp((new-old)/T)
    
    rng = np.random.default_rng(42)
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Generate neighbor: perturb a random subset of indices
        # Perturb 1-5 random indices
        n_perturb = rng.integers(1, 6)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        h_new = current_h.copy()
        # Add Gaussian noise
        noise = rng.normal(0, 0.1, size=n_perturb)
        h_new[indices] += noise
        h_new = np.maximum(h_new, 0.0)
        
        # Normalize
        max_h = np.max(h_new)
        if max_h < 1e-15:
            h_new = current_h.copy()
            val_new = current_val
        else:
            h_new = h_new / max_h
            val_new = compute_value(h_new)
        
        # Acceptance criterion
        if val_new > current_val:
            current_h = h_new
            current_val = val_new
            if val_new > best_val:
                best_h = h_new.copy()
                best_val = val_new
        else:
            # Metropolis criterion for maximization:
            # P(accept) = exp((val_new - val_current) / T)
            # Since val_new < val_current, exponent is negative
            delta = val_new - current_val
            if delta < -T * 10: # If too bad, reject immediately to save time
                pass
            else:
                if rng.random() < np.exp(delta / T):
                    current_h = h_new
                    current_val = val_new
        
        T *= alpha
        
        # Occasionally reheat if stuck? Or just continue.
        # Simple restart logic: if no improvement for many iterations, restart from best with high T
        # For simplicity and robustness, we stick to standard cooling.
        
    return list(best_h)
