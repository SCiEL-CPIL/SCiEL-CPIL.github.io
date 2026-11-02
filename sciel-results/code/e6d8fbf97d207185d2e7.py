"""Model-and-optimize approach for signed step functions minimizing max|autoconv|/(sum x)^2."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 145
SAFETY_MARGIN = 5

def _check_deadline():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _sign_flip_search(x_init, max_flips=5000):
    """
    Local search over ±1 sign patterns.
    Flip individual signs to reduce the objective.
    """
    x = x_init.copy()
    best_val = _objective(x)
    
    for flip_iter in range(max_flips):
        if _check_deadline():
            break
        
        improved = False
        # Try flipping each position
        for j in range(N):
            trial = x.copy()
            trial[j] = -trial[j]
            val = _objective(trial)
            if val < best_val:
                best_val = val
                x = trial
                improved = True
                break
        
        if not improved:
            # Try random flips if no single flip helps
            for _ in range(10):
                j = np.random.randint(N)
                trial = x.copy()
                trial[j] = -trial[j]
                val = _objective(trial)
                if val < best_val:
                    best_val = val
                    x = trial
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, best_val

def _generate_sign_patterns(rng):
    """Generate diverse initial ±1 sign patterns."""
    patterns = []
    
    # All ones
    patterns.append(np.ones(N))
    
    # Alternating
    patterns.append(np.ones(N) * np.where(np.arange(N) % 2 == 0, 1, -1))
    
    # Blocks
    block_sizes = [N//4, N//8, N//16, N//32]
    for bs in block_sizes:
        pat = np.ones(N)
        for i in range(0, N, bs*2):
            pat[i:i+bs] = -1
        patterns.append(pat)
    
    # Random
    for _ in range(20):
        patterns.append(rng.choice([-1, 1], size=N))
    
    # Structured: first half positive, second half negative
    pat = np.ones(N)
    pat[N//2:] = -1
    patterns.append(pat)
    
    # Quarter pattern
    pat = np.ones(N)
    pat[N//4:3*N//4] = -1
    patterns.append(pat)
    
    return patterns

def _amplitude_search(x_signs, max_iters=2000):
    """
    Given a sign pattern, optimize the amplitudes (positive values).
    x[j] = signs[j] * amp[j] where amp[j] > 0
    """
    signs = np.sign(x_signs)
    # Ensure no zeros in signs
    signs[signs == 0] = 1
    
    # Initialize amplitudes
    amps = np.ones(N)
    
    best_val = _objective(signs * amps)
    best_amps = amps.copy()
    
    for it in range(max_iters):
        if _check_deadline():
            break
        
        improved = False
        for j in range(N):
            # Try different amplitude values
            for mult in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
                trial_amps = best_amps.copy()
                trial_amps[j] = best_amps[j] * mult
                trial_x = signs * trial_amps
                val = _objective(trial_x)
                if val < best_val:
                    best_val = val
                    best_amps = trial_amps
                    improved = True
                    break
        
        if not improved:
            break
    
    return signs * best_amps, best_val

def construct():
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    if _check_deadline():
        return best_x
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Sign pattern search
    patterns = _generate_sign_patterns(rng)
    
    for pat in patterns:
        if _check_deadline():
            break
        
        # First evaluate the raw ±1 pattern
        val = _objective(pat)
        if val < best_val:
            best_val = val
            best_x = pat.copy()
        
        # Then do sign flip local search from this pattern
        x_refined, val_refined = _sign_flip_search(pat, max_flips=1000)
        if val_refined < best_val:
            best_val = val_refined
            best_x = x_refined.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 2: Amplitude optimization on best sign pattern
    x_amp, val_amp = _amplitude_search(best_x, max_iters=1000)
    if val_amp < best_val:
        best_val = val_amp
        best_x = x_amp.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 3: More sign patterns with longer search
    for _ in range(30):
        if _check_deadline():
            break
        
        # Generate new random patterns
        pat = rng.choice([-1, 1], size=N)
        
        # Quick evaluation
        val = _objective(pat)
        if val < best_val * 1.5:  # Only pursue if competitive
            x_refined, val_refined = _sign_flip_search(pat, max_flips=2000)
            if val_refined < best_val:
                best_val = val_refined
                best_x = x_refined.copy()
    
    if _check_deadline():
        return best_x
    
    # Phase 4: Final amplitude polish
    x_amp, val_amp = _amplitude_search(best_x, max_iters=2000)
    if val_amp < best_val:
        best_val = val_amp
        best_x = x_amp.copy()
    
    # Phase 5: Continuous gradient polish on the discrete solution
    def _obj_smooth(x, beta=100.0):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(x)
        
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        max_val = np.max(abs_ac)
        
        if max_val < 1e-15:
            return 1e10, np.zeros_like(x)
        
        shifted = beta * (abs_ac - max_val)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        soft_max = max_val + (1.0 / beta) * np.log(sum_exp)
        val = soft_max / (s * s)
        
        g = np.zeros(2*N-1)
        g[:len(ac)] = weights * np.sign(ac)
        
        L_fft = 1
        while L_fft < 3*N:
            L_fft *= 2
        
        G = np.fft.rfft(g, n=L_fft)
        Xf = np.fft.rfft(x, n=L_fft)
        
        term1 = np.fft.irfft(np.conj(G) * Xf, n=L_fft)[:N]
        term2 = np.fft.irfft(G * Xf, n=L_fft)[:N]
        
        d_softmax_dx = term1 + term2
        
        grad = (1.0 / (s * s)) * d_softmax_dx - (2.0 * soft_max / (s ** 3)) * np.ones(N)
        
        return val, grad
    
    try:
        result = minimize(_obj_smooth, best_x, method='L-BFGS-B', jac=True,
                        options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
