"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]) for all k.
    We want to minimize the maximum over all shifts k of this overlap.
    """
    h = np.asarray(h)
    N_local = len(h)
    one_minus_h = 1.0 - h
    
    max_overlap = 0.0
    for k in range(N_local):
        shifted = np.roll(h, k)
        overlap = np.sum(shifted * one_minus_h)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _construct_from_params(params):
    """
    Construct a step-height sequence from low-dimensional parameters.
    
    params: [block_start, transition_width, low_val, high_val, n_blocks]
    We model the sequence as having a block structure with a smooth transition.
    
    Actually, let's use a simpler parameterization:
    - The sequence is piecewise constant with two values: low and high
    - There's a transition region of width W where it goes from low to high
    - We parameterize: start index of transition, width of transition, 
      and the two constant values (low and high)
    
    But we also need sum(h) = N/2 exactly.
    
    Let's use: 
    - a: the "level" parameter, h[i] = a for i in some set, and b for others
    - We need sum = N/2, so a*count_a + b*count_b = N/2
    
    Alternative: use a smooth profile parameterized by a few Fourier coefficients
    or by a sigmoid transition.
    
    Let's try: h[i] = 0.5 + A * f(i) where f is a smooth function with zero mean,
    and A controls the amplitude. But we need h in [0,1].
    
    Better: parameterize as a "square wave" with a smooth transition.
    h[i] = 0.5 + 0.5 * sign(sin(2*pi*i/N + phase)) but smoothed.
    
    Actually, the key insight from the task is that the optimal is piecewise constant
    with two distinct values. Let's parameterize:
    - k: number of steps with value 'high'
    - N-k: number of steps with value 'low'
    - high and low values such that k*high + (N-k)*low = N/2
    - The arrangement: which indices get 'high' vs 'low'
    
    For a given k, we have high = (N/2 - (N-k)*low) / k
    And we need 0 <= low <= high <= 1.
    
    Let's parameterize by:
    - k: integer, number of high steps (1 to N-1)
    - low: the low value (0 to 1)
    - high = (N/2 - (N-k)*low) / k, must be in [low, 1]
    - arrangement: a binary pattern of length N with k ones (high) and N-k zeros (low)
    
    For the arrangement, we can use a block structure: first k positions are high,
    rest are low, or some other pattern. The optimal arrangement likely has a specific
    structure related to the overlap minimization.
    
    Let's try: for a given k and low, compute high. Then try several arrangements:
    1. Block: [high]*k + [low]*(N-k)
    2. Alternating (if possible)
    3. Random permutations (sample a few)
    
    But this is combinatorial. Let's instead use a continuous parameterization.
    
    New approach: parameterize h as:
    h[i] = 0.5 + A * g(i/N) where g is a periodic function with zero mean,
    and A is chosen so that max h = 1 or min h = 0 (boundary).
    
    Or even simpler: use the fact that for the "two-value" solution,
    we can directly optimize over k and low, and for the arrangement,
    use a "greedy" or "sorted" placement.
    
    Let me think about the overlap more carefully.
    overlap(k) = sum_i h[i] * (1 - h[(i+k)%N])
    = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    = N/2 - autocorrelation(k)
    
    So minimizing max overlap = N/2 - min_k autocorrelation(k)
    Wait, that's not right. Let me redo.
    
    overlap at shift k = sum_i h[i] * (1 - h[(i+k)%N])
    = sum_i h[i] - sum_i h[i] h[(i+k)%N]
    = N/2 - C(k) where C(k) = sum_i h[i] h[(i+k)%N]
    
    So max overlap = N/2 - min_k C(k)
    
    To minimize max overlap, we want to maximize min_k C(k).
    
    So the problem is: maximize the minimum autocorrelation over all shifts.
    
    For a two-valued sequence, the autocorrelation structure is determined by
    the arrangement of the high values.
    
    Let's just do a direct numerical optimization with a good parameterization.
    
    Parameterization: 
    - Use a "block" structure with a smooth transition
    - h[i] = 0.5 + 0.5 * tanh((i - center) / width) for i in [0, N)
    - But this doesn't wrap around nicely.
    
    Let's use a Fourier-based approach:
    h[i] = 0.5 + sum_{m=1}^{M} a_m * cos(2*pi*m*i/N + phi_m)
    with constraints 0 <= h[i] <= 1 and sum h[i] = N/2.
    
    The sum constraint is automatically satisfied if all a_m are real (no DC component beyond 0.5).
    The bounds are the tricky part.
    
    Let's use M=2 or M=3 harmonics and optimize the amplitudes and phases.
    """
    # params: [a1, phi1, a2, phi2, a3, phi3]
    a1, phi1, a2, phi2, a3, phi3 = params
    i = np.arange(N)
    h = 0.5 + a1*np.cos(2*np.pi*i/N + phi1) + a2*np.cos(4*np.pi*i/N + phi2) + a3*np.cos(6*np.pi*i/N + phi3)
    # Clip to [0, 1]
    h = np.clip(h, 0, 1)
    # Renormalize to ensure sum = N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N/2.0) / s
    # Re-clip after scaling (should be minimal change if s is close to N/2)
    h = np.clip(h, 0, 1)
    # Final check: if sum is off, adjust
    s2 = np.sum(h)
    if abs(s2 - N/2.0) > 1e-10:
        # Small adjustment: add a constant and re-clip
        diff = N/2.0 - s2
        h = h + diff/N
        h = np.clip(h, 0, 1)
    return h

def _objective(params):
    h = _construct_from_params(params)
    return _score(h)

def _objective_fourier(params):
    """Optimize Fourier coefficients directly."""
    # params: [a1, phi1, a2, phi2, a3, phi3, a4, phi4]
    a1, phi1, a2, phi2, a3, phi3, a4, phi4 = params
    i = np.arange(N)
    h = 0.5
    h += a1*np.cos(2*np.pi*i/N + phi1)
    h += a2*np.cos(4*np.pi*i/N + phi2)
    h += a3*np.cos(6*np.pi*i/N + phi3)
    h += a4*np.cos(8*np.pi*i/N + phi4)
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    if s > 1e-10:
        h = h * (N/2.0) / s
    h = np.clip(h, 0, 1)
    return _score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try several initializations and local optimizations
    # Phase 1: Global search with differential evolution on Fourier params
    bounds = [
        (0, 0.5), (-np.pi, np.pi),  # a1, phi1
        (0, 0.5), (-np.pi, np.pi),  # a2, phi2
        (0, 0.5), (-np.pi, np.pi),  # a3, phi3
        (0, 0.5), (-np.pi, np.pi),  # a4, phi4
    ]
    
    # First, try a few random starts with local optimization
    rng = np.random.default_rng(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Random Fourier coefficients
        params = np.array([
            rng.uniform(0, 0.4), rng.uniform(-np.pi, np.pi),
            rng.uniform(0, 0.3), rng.uniform(-np.pi, np.pi),
            rng.uniform(0, 0.2), rng.uniform(-np.pi, np.pi),
            rng.uniform(0, 0.15), rng.uniform(-np.pi, np.pi),
        ])
        
        # Local optimization
        try:
            result = minimize(_objective_fourier, params, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            h_opt = _construct_from_params(result.x[:6])  # Use first 6 for 3 harmonics
            # Actually let's use all 4 harmonics
            a1, phi1, a2, phi2, a3, phi3, a4, phi4 = result.x
            i = np.arange(N)
            h_opt = 0.5
            h_opt += a1*np.cos(2*np.pi*i/N + phi1)
            h_opt += a2*np.cos(4*np.pi*i/N + phi2)
            h_opt += a3*np.cos(6*np.pi*i/N + phi3)
            h_opt += a4*np.cos(8*np.pi*i/N + phi4)
            h_opt = np.clip(h_opt, 0, 1)
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt = h_opt * (N/2.0) / s
            h_opt = np.clip(h_opt, 0, 1)
            
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 2: Try the "two-block" analytic initialization
    # For a two-valued sequence with k high values and N-k low values:
    # low = (N/2 - k*high) / (N-k)
    # We want to maximize min_k autocorrelation
    
    for k in range(1, N):
        if time.time() > deadline:
            break
        
        # Try different arrangements of k high values
        # Arrangement 1: block
        for low_val in np.linspace(0.05, 0.45, 10):
            high_val = (N/2.0 - (N-k)*low_val) / k
            if high_val < low_val or high_val > 1.0:
                continue
            
            # Block arrangement
            h = np.full(N, low_val)
            h[:k] = high_val
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
            
            # Arrangement 2: evenly spaced
            h = np.full(N, low_val)
            indices = np.linspace(0, N, k+1, dtype=int)[:k]
            if k > 0:
                h[indices] = high_val
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
    
    # Phase 3: Local refinement of best solution using direct h optimization
    if time.time() < deadline:
        # Use the best_h as initial guess and do a local optimization
        # on the h values directly with constraints
        
        def _direct_objective(h_flat):
            h = np.clip(h_flat, 0, 1)
            s = np.sum(h)
            if s > 1e-10:
                h = h * (N/2.0) / s
            h = np.clip(h, 0, 1)
            return _score(h)
        
        try:
            result = minimize(_direct_objective, best_h, method='L-BFGS-B',
                             bounds=[(0, 1)]*N, options={'maxiter': 500, 'ftol': 1e-12})
            h_opt = np.clip(result.x, 0, 1)
            s = np.sum(h_opt)
            if s > 1e-10:
                h_opt = h_opt * (N/2.0) / s
            h_opt = np.clip(h_opt, 0, 1)
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure sum is exactly N/2
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-12:
        best_h = best_h * (N/2.0) / s
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
