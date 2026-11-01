"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import convolve1d

def _score(h):
    """Compute the autoconvolution upper bound metric: max|conv| / (sum|h|)^2.
    Lower is better."""
    # Compute autoconvolution (linear convolution with itself)
    conv = np.convolve(h, h, mode='full')
    # The relevant part is the centered autocorrelation
    # For a step function on [-1/4, 1/4], the autoconvolution at lag 0 is the main term
    # We want to minimize the maximum absolute value of the sidelobes relative to the main lobe
    # Actually, let's think about what the metric should be.
    # The task says "score is the upper bound on C3" and "raw value (lower is better)"
    # For autocorrelation inequality, C3 relates to the ratio of sidelobe to main lobe
    
    # Let's compute the normalized autocorrelation
    # The main lobe is at lag 0, sidelobes are at other lags
    # We want to minimize the maximum sidelobe / main lobe ratio
    
    # Actually, re-reading: "max|conv|/(sum|h|)^2" from the knowledge section
    # This suggests the metric is max|autoconv| / (sum|h|)^2
    
    main = np.sum(np.abs(h))
    if main == 0:
        return np.inf
    
    # The autoconvolution at lag 0 (center) gives the energy
    # For the inequality, we care about the peak sidelobe level
    # Let's use: max over all lags of |conv[k]| / (sum|h|)^2
    # But the lag-0 value is (sum|h|^2) for real h, which is <= (sum|h|)^2
    
    # Actually from the knowledge: "evaluation metric max|conv|/(sum|h|)^2"
    # This is the max of the entire autoconvolution divided by (sum of absolute heights)^2
    
    max_conv = np.max(np.abs(conv))
    return max_conv / (main * main)

def _construct_initializations():
    """Generate diverse initial sign patterns."""
    inits = []
    
    # All ones (baseline)
    inits.append(np.ones(N))
    
    # All -1
    inits.append(-np.ones(N))
    
    # Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    inits.append(alt)
    
    # Block patterns of various sizes
    for block in [2, 4, 8, 16, 32, 50, 100]:
        pattern = np.ones(N)
        for i in range(0, N, block):
            if (i // block) % 2 == 1:
                pattern[i:i+block] = -1
        inits.append(pattern.copy())
    
    # Random signs
    rng = np.random.RandomState(42)
    for _ in range(10):
        inits.append(rng.choice([-1, 1], size=N).astype(float))
    
    # Smooth random (low frequency)
    for freq in [1, 2, 5, 10]:
        t = np.linspace(0, freq * 2 * np.pi, N)
        inits.append(np.sign(np.sin(t)))
    
    # Random smooth
    for _ in range(5):
        t = np.linspace(0, 2 * np.pi * rng.randint(1, 20), N)
        phase = rng.uniform(0, 2 * np.pi)
        inits.append(np.sign(np.sin(t + phase)))
    
    return inits

def _hill_climb(h, max_iter=5000, deadline=None):
    """Local search: flip individual signs to reduce the metric."""
    current_score = _score(h)
    h = h.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random position
        pos = np.random.randint(N)
        old_val = h[pos]
        h[pos] = -old_val
        new_score = _score(h)
        
        if new_score < current_score:
            current_score = new_score
        else:
            h[pos] = old_val
    
    return h, current_score

def _tanh_relax(h0, deadline=None):
    """Use tanh relaxation: optimize continuous variables, then threshold."""
    # Start with continuous values in (-1, 1)
    x = np.tanh(h0 * 0.5)  # map to (-1, 1)
    
    def objective(x_cont):
        # Soft sign: use tanh to smooth
        h_soft = np.tanh(x_cont * 5)  # sharper tanh
        return _score(h_soft)
    
    def gradient(x_cont):
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(x_cont)
        for i in range(N):
            x_plus = x_cont.copy()
            x_plus[i] += eps
            x_minus = x_cont.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    result = minimize(objective, x, method='L-BFGS-B', 
                      options={'maxiter': 200, 'maxfun': 10000})
    
    if result.success or result.fun < _score(h0):
        h_final = np.sign(result.x)
        # Ensure no zeros
        h_final[h_final == 0] = 1
        return h_final, _score(h_final)
    
    return h0, _score(h0)

def construct():
    """Construct a signed step function with 400 intervals to minimize the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to stay within 120s limit
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Generate diverse initializations
    inits = _construct_initializations()
    
    # Evaluate all initializations
    for h0 in inits:
        if time.time() > deadline:
            break
        s = _score(h0)
        if s < best_score:
            best_score = s
            best_h = h0.copy()
    
    # Phase 1: Hill climbing from best initializations
    for h0 in inits[:20]:  # Try top 20 diverse starts
        if time.time() > deadline:
            break
        
        h_climbed, s_climbed = _hill_climb(h0, max_iter=2000, deadline=deadline)
        if s_climbed < best_score:
            best_score = s_climbed
            best_h = h_climbed.copy()
    
    # Phase 2: More intensive hill climbing from best
    if time.time() < deadline - 30:
        h_climbed, s_climbed = _hill_climb(best_h, max_iter=10000, deadline=deadline - 20)
        if s_climbed < best_score:
            best_score = s_climbed
            best_h = h_climbed.copy()
    
    # Phase 3: Try tanh relaxation if time permits
    if time.time() < deadline - 60:
        h_tanh, s_tanh = _tanh_relax(best_h, deadline=deadline - 10)
        if s_tanh < best_score:
            best_score = s_tanh
            best_h = h_tanh.copy()
        
        # Hill climb after tanh
        if time.time() < deadline - 20:
            h_climbed, s_climbed = _hill_climb(best_h, max_iter=5000, deadline=deadline)
            if s_climbed < best_score:
                best_score = s_climbed
                best_h = h_climbed.copy()
    
    # Phase 4: Final intensive hill climb
    remaining = deadline - time.time()
    if remaining > 5:
        max_iter = int(remaining * 50)  # roughly 50 iterations per second
        h_climbed, s_climbed = _hill_climb(best_h, max_iter=max_iter, deadline=deadline)
        if s_climbed < best_score:
            best_score = s_climbed
            best_h = h_climbed.copy()
    
    # Ensure output is a sequence of signed step heights
    return best_h
# EVOLVE-BLOCK-END
