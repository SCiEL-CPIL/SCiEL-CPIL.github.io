import numpy as np
import time

N = 400
DEADLINE = 135.0  # safety margin before 150s kill

def _compute_c3_bound(signs):
    """
    Compute an upper bound on C3 for a signed step function.
    Uses L1 norm of autoconvolution as the metric.
    """
    dx = 0.5 / N
    n_fft = 2 * N
    F = np.fft.rfft(signs.astype(float), n=n_fft)
    conv = np.fft.irfft(F * F, n=n_fft) * dx
    return np.sum(np.abs(conv)) * dx

def _spectral_phase_optimize(signs, max_rounds=50):
    """
    Optimize by adjusting the phase of a low-frequency sine wave to generate
    a smooth sign pattern, then refine with local flips.
    This contrasts with pure random/greedy by using a structured parametric family.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    # Generate sign patterns from sine waves with varying frequency and phase
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    
    best_pattern = current
    best_score = current_score
    
    # Try different fundamental frequencies and phases
    for freq in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]:
        for phase in np.linspace(0, 2*np.pi, 20, endpoint=False):
            elapsed = time.time()
            if elapsed > DEADLINE - 30:
                break
            
            # Create sign pattern from sine wave
            sine_vals = np.sin(2 * np.pi * freq * x + phase)
            sign_pattern = np.sign(sine_vals)
            sign_pattern[sign_pattern == 0] = 1
            
            s = _compute_c3_bound(sign_pattern)
            if s < best_score:
                best_score = s
                best_pattern = sign_pattern.copy()
    
    return best_pattern, best_score

def _greedy_flip_optimize(signs, max_rounds=50):
    """
    Greedy coordinate descent: flip individual signs if it reduces the C3 bound.
    Multiple passes until no improvement or max rounds reached.
    """
    current = signs.astype(np.float64).copy()
    current_score = _compute_c3_bound(current)
    
    for _ in range(max_rounds):
        improved = False
        # Try flipping each position
        for i in range(N):
            old_val = current[i]
            current[i] = -old_val
            new_score = _compute_c3_bound(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
        if not improved:
            break
    
    return current, current_score

def construct():
    start_time = time.time()
    
    # Start with spectral phase optimization to find a good structured pattern
    rng = np.random.RandomState(42)
    
    # Initial diverse patterns
    initializations = []
    # All ones
    initializations.append(np.ones(N))
    # Alternating
    initializations.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)]))
    # Random
    initializations.append(rng.choice([1, -1], size=N))
    # Block structure: +1 -1 +1 -1
    p4 = np.ones(N)
    p4[N//4:N//2] = -1
    p4[3*N//4:] = -1
    initializations.append(p4)
    # Sine pattern
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    p5 = np.sign(np.cos(4 * np.pi * x))
    p5[p5 == 0] = 1
    initializations.append(p5)
    
    # Find best initial pattern
    best_pattern = initializations[0]
    best_score = _compute_c3_bound(best_pattern)
    
    for init in initializations[1:]:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 40:
            break
        s = _compute_c3_bound(init)
        if s < best_score:
            best_score = s
            best_pattern = init
    
    # Apply spectral phase optimization to find a better structured pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 40:
        spectral_pattern, spectral_score = _spectral_phase_optimize(best_pattern, max_rounds=10)
        if spectral_score < best_score:
            best_score = spectral_score
            best_pattern = spectral_pattern
    
    # Run greedy optimization on the best pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        optimized, opt_score = _greedy_flip_optimize(best_pattern, max_rounds=30)
        if opt_score < best_score:
            best_score = opt_score
            best_pattern = optimized
    
    # Try greedy from a second random start if time permits
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 25:
        rand_start = rng.choice([1, -1], size=N)
        rand_score = _compute_c3_bound(rand_start)
        if rand_score < best_score:
            best_score = rand_score
            best_pattern = rand_start
        optimized2, opt_score2 = _greedy_flip_optimize(best_pattern, max_rounds=20)
        if opt_score2 < best_score:
            best_score = opt_score2
            best_pattern = optimized2
    
    # Ensure we return integer signs
    return np.sign(best_pattern).astype(int)
