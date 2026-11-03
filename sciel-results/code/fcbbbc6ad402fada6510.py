"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, so dx = 1/(2N)
    
    def compute_score(heights):
        """Compute the normalized maximum of autoconvolution."""
        heights = np.asarray(heights, dtype=np.float64)
        heights = np.maximum(heights, 0.0)  # enforce nonnegativity
        
        l2_sq = dx * np.sum(heights**2)
        if l2_sq < 1e-15:
            return 1e10
        
        # Autoconvolution via FFT
        n_fft = 1 << (2 * N).bit_length()
        f_pad = np.zeros(n_fft)
        f_pad[:N] = heights * dx
        
        F = np.fft.fft(f_pad)
        autoconv = np.fft.ifft(F * F).real
        # Take the linear part (first N values correspond to lags 0 to (N-1)*dx,
        # and the rest wrap around; for autoconvolution, max is at center)
        # The linear convolution of length N with itself has length 2N-1
        # In the FFT result, indices 0..N-1 are lags 0..(N-1)*dx
        # Indices N..n_fft-1 are negative lags wrapped around
        # For autoconvolution (symmetric), max should be near index 0 or N
        # Actually, let's just take the max of the first 2N-1 values properly
        # The linear convolution is in indices 0..N-1 and n_fft-N+1..n_fft-1 (wrapped)
        # Simpler: just use np.convolve for accuracy since N=600
        autoconv_lin = np.convolve(heights, heights, mode='full') * dx**2
        return np.max(autoconv_lin) / l2_sq

    def build_function(params):
        """Build the step function from parameters.
        params: [envelope_width, mod_amplitude, mod_freq, mod_phase, base_level]
        h(x) = base_level + envelope(x) * (1 + mod_amplitude * cos(mod_freq * x + mod_phase))
        where envelope is a raised cosine on [-1/4, 1/4]
        """
        if time.time() > deadline:
            return None
            
        env_width = params[0]  # controls width of envelope
        mod_amp = params[1]    # modulation amplitude
        mod_freq = params[2]   # modulation frequency
        mod_phase = params[3]  # modulation phase
        base_level = params[4] # base level to ensure nonnegativity
        
        x = np.linspace(-0.25, 0.25, N + 1)[:-1] + dx/2  # centers of intervals
        
        # Raised cosine envelope: 0.5 * (1 + cos(pi * x / (0.25 * env_width)))
        # This gives a smooth bump that is zero at the edges
        # env_width in [0.1, 1.0], where 1.0 means full support
        env = 0.5 * (1 + np.cos(np.pi * x / (0.25 * max(env_width, 0.01))))
        env = np.maximum(env, 0)  # clip to ensure nonnegative
        
        # Periodic modulation
        mod = 1.0 + mod_amp * np.cos(mod_freq * x + mod_phase)
        mod = np.maximum(mod, 0)  # ensure nonnegative modulation
        
        h = base_level + env * mod
        h = np.maximum(h, 0)  # final nonnegativity guarantee
        
        return h

    def objective(params):
        if time.time() > deadline:
            return 1e10
        h = build_function(params)
        if h is None:
            return 1e10
        return compute_score(h)

    # Define parameter bounds
    bounds = [
        (0.1, 1.0),      # env_width: 0.1 to 1.0 (fraction of half-width)
        (0.0, 1.0),      # mod_amp: 0 to 1
        (0.0, 20.0),     # mod_freq: 0 to 20
        (0.0, 2*np.pi),  # mod_phase: 0 to 2*pi
        (0.0, 2.0),      # base_level: 0 to 2
    ]
    
    best_score = float('inf')
    best_h = None
    
    # First, try some simple initial guesses
    initial_guesses = [
        [0.5, 0.0, 0.0, 0.0, 0.0],   # pure raised cosine
        [0.3, 0.0, 0.0, 0.0, 0.0],   # narrower raised cosine
        [1.0, 0.0, 0.0, 0.0, 0.0],   # full raised cosine
        [0.5, 0.5, 5.0, 0.0, 0.0],   # with modulation
        [0.3, 0.5, 10.0, 0.0, 0.0],  # narrow with modulation
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        h = build_function(guess)
        if h is not None:
            s = compute_score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    # Use differential evolution for global optimization
    if time.time() < deadline - 30:  # leave 30s for final evaluation
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=True,
                update_interval=1,
                callback=lambda x, convergence: (time.time() > deadline)
            )
            if result.success or result.fun < best_score:
                h = build_function(result.x)
                if h is not None:
                    s = compute_score(h)
                    if s < best_score:
                        best_score = s
                        best_h = h.copy()
        except Exception:
            pass
    
    # Fallback: if nothing worked, return a simple peaked function
    if best_h is None:
        x = np.linspace(-0.25, 0.25, N + 1)[:-1] + dx/2
        best_h = 0.5 * (1 + np.cos(np.pi * x / 0.25))
        best_h = np.maximum(best_h, 0)
    
    return list(best_h)

# EVOLVE-BLOCK-END
