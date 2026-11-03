"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _autoconv_full(x):
    """Compute the full autoconvolution of x (length 2*N-1)."""
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    conv = np.fft.irfft(f * np.conj(f), n=2*n)
    # Autoconvolution: conv[k] = sum_i x[i]*x[i+k], for k = -(n-1)..(n-1)
    # The FFT gives us conv of length 2n, where the "center" is at index n
    # We want lags from -(n-1) to (n-1), which corresponds to indices 1..2n-1 in the circular conv
    # Actually, let's just return the relevant portion
    # conv[0] = sum x[i]^2 = n (lag 0)
    # conv[k] for k=1..n-1 is lag +k
    # conv[n+1..2n-1] wraps around to negative lags
    return conv

def _max_abs_autoconv(x):
    """Max absolute value of autoconvolution excluding lag 0."""
    conv = _autoconv_full(x)
    n = len(x)
    # Exclude lag 0 (index 0) and the wrap-around that equals lag 0
    # conv[0] is lag 0, conv[n] is also lag 0 (circular)
    # We want lags 1..n-1 and -(n-1)..-1
    # In the FFT output of length 2n: conv[1..n-1] are positive lags, conv[n+1..2n-1] are negative lags
    # But wait, let me verify: for linear autoconv, we pad to 2n
    # The result has the autoconv at indices 0..2n-1 where index 0 is lag 0
    # Actually for autocorrelation, it's symmetric. Let me just compute directly for clarity
    # conv_full[k] = sum_{i} x[i] * x[i+k] for valid i
    # Using FFT: if we compute fft(x, 2n) * conj(fft(x, 2n)), ifft gives circular autocorr of length 2n
    # The linear autocorr is embedded: lags 0..n-1 at indices 0..n-1, lags -(n-1)..-1 at indices n+1..2n-1
    ac = conv
    # Exclude index 0 (lag 0) and index n (which should be 0 for linear, but circular might have artifact)
    # For safety, take max over indices 1..n-1 and n+1..2n-1
    mask = np.ones(2*n, dtype=bool)
    mask[0] = False
    mask[n] = False  # This should be 0 for proper linear autocorr
    return np.max(np.abs(ac[mask]))

def _true_score(x):
    """Compute the true score: max|autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    denom = s * s
    if abs(denom) < 1e-12:
        return np.inf
    return _max_abs_autoconv(x) / denom

def _soft_max_ac(x, beta=10.0):
    """Soft approximation of max|autoconv| using log-sum-exp."""
    conv = _autoconv_full(x)
    n = len(x)
    mask = np.ones(2*n, dtype=bool)
    mask[0] = False
    mask[n] = False
    ac_vals = np.abs(conv[mask])
    # Soft max: (1/beta) * log(sum(exp(beta * ac_vals)))
    # Use numerical stability
    max_val = np.max(ac_vals)
    if max_val < 1e-12:
        return max_val
    shifted = ac_vals - max_val
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    return max_val + np.log(np.sum(np.exp(beta * (ac_vals - max_val)))) / beta

def _soft_objective(x, beta=10.0):
    """Soft objective: soft_max_ac / (sum x)^2"""
    s = np.sum(x)
    denom = s * s
    if abs(denom) < 1e-12:
        return 1e10
    sm = _soft_max_ac(x, beta)
    return sm / denom

def _balanced_random_init(n, seed):
    """Generate a balanced random ±1 sequence (equal number of +1 and -1)."""
    rng = np.random.RandomState(seed)
    if n % 2 == 0:
        half = n // 2
        x = np.concatenate([np.ones(half), -np.ones(half)])
        rng.shuffle(x)
    else:
        # For odd n, have one more +1 than -1
        half = n // 2
        x = np.concatenate([np.ones(half + 1), -np.ones(half)])
        rng.shuffle(x)
    return x

def _fast_flip_eval(x, i):
    """
    Compute the change in autoconvolution when flipping x[i].
    Returns the new max_abs_autoconv efficiently.
    
    When flipping x[i] from v to -v:
    For lag k, the autoconv changes by -2*v*(x[i+k] + x[i-k]) for valid indices.
    """
    n = len(x)
    v = x[i]
    # Compute the delta for all lags
    # delta[k] = -2 * v * (x[i+k] + x[i-k]) for valid i+k and i-k
    # We need to compute the full autoconv after flip, but we can do it incrementally
    
    # Actually, let's just compute the full autoconv after the flip using FFT
    # But that's O(N log N). Let's try a smarter approach.
    
    # The autoconv at lag k: A[k] = sum_j x[j]*x[j+k]
    # After flipping x[i]: A_new[k] = A[k] - 2*v*(x[i+k]*v + x[i-k]*v) ... 
    # Wait, let me redo this.
    # A[k] = sum_j x[j] * x[j+k]
    # Terms involving x[i]: when j=i, term is x[i]*x[i+k]; when j+k=i, i.e., j=i-k, term is x[i-k]*x[i]
    # So A[k] contains x[i]*x[i+k] + x[i-k]*x[i] (if both indices valid)
    # After flip: (-x[i])*x[i+k] + x[i-k]*(-x[i]) = -x[i]*x[i+k] - x[i-k]*x[i]
    # Delta = -2*x[i]*x[i+k] - 2*x[i-k]*x[i] = -2*v*(x[i+k] + x[i-k])
    
    # To get the new max|autoconv|, we need the full new autoconv.
    # Let's compute it via FFT since it's simpler and N=400 is small.
    x_new = x.copy()
    x_new[i] = -x_new[i]
    f = np.fft.rfft(x_new, n=2*n)
    conv = np.fft.irfft(f * np.conj(f), n=2*n)
    mask = np.ones(2*n, dtype=bool)
    mask[0] = False
    mask[n] = False
    return np.max(np.abs(conv[mask]))

def _batched_flip_eval(x, indices, n_eval=100):
    """
    Evaluate flipping each index in 'indices' and return the new max_abs_autoconv for each.
    Uses batched FFT for efficiency.
    """
    n = len(x)
    results = np.zeros(len(indices))
    for idx, i in enumerate(indices):
        x_new = x.copy()
        x_new[i] = -x_new[i]
        f = np.fft.rfft(x_new, n=2*n)
        conv = np.fft.irfft(f * np.conj(f), n=2*n)
        mask = np.ones(2*n, dtype=bool)
        mask[0] = False
        mask[n] = False
        results[idx] = np.max(np.abs(conv[mask]))
    return results

def _greedy_flip_fast(x, max_passes=5, time_budget=30.0, rng=None):
    """Fast greedy search using batched flip evaluation."""
    if rng is None:
        rng = np.random.RandomState()
    start_time = time.time()
    n = len(x)
    current_score = _true_score(x)
    
    for pass_num in range(max_passes):
        if time.time() - start_time > time_budget:
            break
        improved = False
        # Evaluate all flips in batches
        batch_size = 50
        for start in range(0, n, batch_size):
            if time.time() - start_time > time_budget:
                break
            end = min(start + batch_size, n)
            indices = list(range(start, end))
            new_max_acs = _batched_flip_eval(x, indices)
            # For each flip, compute the new score
            for idx, i in enumerate(indices):
                new_sum = np.sum(x) - 2 * x[i]
                new_denom = new_sum * new_sum
                if abs(new_denom) < 1e-12:
                    continue
                new_score = new_max_acs[idx] / new_denom
                if new_score < current_score:
                    x[i] = -x[i]
                    current_score = new_score
                    improved = True
                    # Re-evaluate remaining indices in this batch
                    remaining = list(range(start, end))
                    remaining.remove(i)
                    if remaining:
                        new_max_acs_rem = _batched_flip_eval(x, remaining)
                        for idx2, i2 in enumerate(remaining):
                            new_sum2 = np.sum(x) - 2 * x[i2]
                            new_denom2 = new_sum2 * new_sum2
                            if abs(new_denom2) < 1e-12:
                                continue
                            new_score2 = new_max_acs_rem[idx2] / new_denom2
                            if new_score2 < current_score:
                                x[i2] = -x[i2]
                                current_score = new_score2
                                improved = True
        if not improved:
            break
    return x, current_score

def _simulated_annealing_fast(x, start_temp=1.0, end_temp=0.01, steps=50000, time_budget=40.0, rng=None):
    """Fast simulated annealing with precomputed autoconv."""
    if rng is None:
        rng = np.random.RandomState()
    start_time = time.time()
    n = len(x)
    current_score = _true_score(x)
    best_x = x.copy()
    best_score = current_score
    
    for step in range(steps):
        if time.time() - start_time > time_budget:
            break
        temp = start_temp * (end_temp / start_temp) ** (step / steps)
        
        i = rng.randint(n)
        x[i] = -x[i]
        new_score = _true_score(x)
        
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[i] = -x[i]
    
    return best_x, best_score

def _gradient_polish(x, time_budget=30.0):
    """
    Gradient-based polish using soft-max objective.
    Uses L-BFGS-B to optimize a soft approximation, then rounds.
    """
    start_time = time.time()
    n = len(x)
    
    def objective_and_grad(z):
        """z is a continuous vector; we project to ±1 for evaluation but use soft objective."""
        # Use soft objective for gradient
        beta = 50.0
        sm = _soft_max_ac(z, beta)
        s = np.sum(z)
        denom = s * s
        if abs(denom) < 1e-12:
            return 1e10, np.zeros_like(z)
        val = sm / denom
        
        # Gradient: use finite differences for simplicity (or compute analytically)
        # For now, use numerical gradient
        eps = 1e-5
        grad = np.zeros_like(z)
        for j in range(n):
            z_plus = z.copy()
            z_plus[j] += eps
            z_minus = z.copy()
            z_minus[j] -= eps
            grad[j] = (_soft_objective(z_plus, beta) - _soft_objective(z_minus, beta)) / (2 * eps)
        
        return val, grad
    
    # Start from the discrete x, try to optimize in continuous space
    z0 = x.astype(float)
    result = minimize(objective_and_grad, z0, method='L-BFGS-B', 
                      options={'maxiter': 100, 'time_limit': time_budget})
    
    # Round back to ±1
    x_rounded = np.sign(result.x)
    # Ensure no zeros
    x_rounded[x_rounded == 0] = 1.0
    
    # Check if the rounded version is better
    score_rounded = _true_score(x_rounded)
    score_orig = _true_score(x)
    
    if score_rounded < score_orig:
        return x_rounded, score_rounded
    else:
        return x, score_orig

def _multi_flip_search(x, time_budget=20.0, rng=None):
    """Search over pairs of flips to escape local minima."""
    if rng is None:
        rng = np.random.RandomState()
    start_time = time.time()
    n = len(x)
    best_x = x.copy()
    best_score = _true_score(x)
    
    while time.time() - start_time < time_budget:
        # Pick two random positions to flip
        i, j = rng.choice(n, size=2, replace=False)
        x_new = best_x.copy()
        x_new[i] = -x_new[i]
        x_new[j] = -x_new[j]
        new_score = _true_score(x_new)
        if new_score < best_score:
            best_x = x_new
            best_score = new_score
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with balanced random initialization
    best_x = _balanced_random_init(N, seed=42)
    best_score = _true_score(best_x)
    
    # Try a few different seeds for initialization
    for seed in [1, 7, 13, 100, 2024]:
        if time.time() > deadline:
            break
        x_candidate = _balanced_random_init(N, seed=seed)
        s = _true_score(x_candidate)
        if s < best_score:
            best_x = x_candidate.copy()
            best_score = s
    
    # Phase 1: Fast greedy
    x, score = _greedy_flip_fast(best_x.copy(), max_passes=10, time_budget=20.0)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 2: Simulated annealing
    x, score = _simulated_annealing_fast(best_x.copy(), start_temp=1.0, end_temp=0.01, 
                                          steps=100000, time_budget=40.0)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 3: Multi-flip search to escape local minima
    x, score = _multi_flip_search(best_x.copy(), time_budget=20.0)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 4: Another greedy pass
    x, score = _greedy_flip_fast(best_x.copy(), max_passes=10, time_budget=20.0)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    if time.time() > deadline:
        return best_x
    
    # Phase 5: Final simulated annealing with lower temperature
    x, score = _simulated_annealing_fast(best_x.copy(), start_temp=0.1, end_temp=0.001, 
                                          steps=50000, time_budget=20.0)
    if score < best_score:
        best_x = x.copy()
        best_score = score
    
    return best_x
# EVOLVE-BLOCK-END
