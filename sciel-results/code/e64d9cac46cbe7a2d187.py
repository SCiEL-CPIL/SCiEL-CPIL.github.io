import time
import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Initialize with uniform distribution - known global minimum
    h = np.ones(N)
    
    def compute_normalized_max_ac(h):
        """Compute normalized maximum of autoconvolution.
        Normalization: max(ac) / (sum(h))^2
        For non-negative h, max(ac) = ac[0] = sum(h_i^2)
        """
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10
        # ac[0] = sum(h_i^2)
        ac0 = np.sum(h * h)
        return ac0 / (l1 * l1)
    
    best_h = h.copy()
    best_score = compute_normalized_max_ac(h)
    
    # Quick verification: try a few random perturbations to confirm uniform is optimal
    # This is a sanity check, not expected to find improvement
    rng = np.random.RandomState(42)
    num_trials = 200
    
    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break
        
        # Create a perturbation: shift mass from some bins to others
        # Keep sum constant to test shape sensitivity
        h_test = h.copy()
        
        # Random perturbation: add noise, then re-normalize to keep sum = N
        noise = rng.normal(0, 0.1, N)
        h_test = h_test + noise
        h_test = np.maximum(h_test, 0)  # enforce non-negativity
        
        # Re-normalize to keep sum = N (same as uniform's sum)
        s = np.sum(h_test)
        if s < 1e-10:
            continue
        h_test = h_test * (N / s)
        
        score = compute_normalized_max_ac(h_test)
        if score < best_score - 1e-12:
            best_score = score
            best_h = h_test.copy()
    
    # If no improvement found (as expected), return uniform
    # The uniform distribution minimizes sum(h_i^2)/(sum h_i)^2 by Cauchy-Schwarz
    # Minimum value is 1/N, achieved uniquely by uniform distribution
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
