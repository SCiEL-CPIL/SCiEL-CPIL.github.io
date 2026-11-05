# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using Zadoff-Chu sequences modulated by chirps and windowed.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    t = np.arange(N, dtype=np.float64)
    
    def get_h(params):
        # params: mu, alpha, win_param, dc
        mu, alpha, win_param, dc = params
        # Ensure mu is an integer for ZC properties, or allow continuous for general chirp?
        # ZC requires integer. Let's round mu to nearest integer to preserve ZC structure if possible,
        # or just use the phase formula which works for any mu but ZC property holds for integer.
        mu_int = int(round(mu))
        if mu_int == 0:
            mu_int = 1
            
        # Zadoff-Chu like sequence: exp(-1j * pi * mu * t * (t + 1) / N)
        zc = np.exp(-1j * np.pi * mu_int * t * (t + 1) / N)
        
        # Chirp modulation: exp(1j * pi * alpha * t * t / N)
        chirp = np.exp(1j * np.pi * alpha * t * t / N)
        
        signal = zc * chirp
        
        # Windowing
        # win_param < 1.0: Cosine/Hann blend
        # win_param >= 1.0: Gaussian
        if win_param < 1.0:
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            hann = 0.5 * (1 + np.cos(np.pi * n_norm))
            window = (1 - win_param) * 1.0 + win_param * hann
        else:
            sigma = win_param - 1.0 + 0.1
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            window = np.exp(-0.5 * (n_norm / (sigma/2.0))**2)
        
        h_real = np.real(signal) * window + dc
        return h_real

    best_candidate = None
    best_candidate_val = float('inf')
    
    # Multi-start optimization
    np.random.seed(42)
    
    # Define bounds for parameters
    # mu: 1 to 20 (integer-ish)
    # alpha: -1.0 to 1.0
    # win_param: 0.0 to 3.0
    # dc: -2.0 to 2.0
    
    bounds = [(1, 20), (-1.0, 1.0), (0.0, 3.0), (-2.0, 2.0)]
    
    # Run differential evolution for a global search
    # This might be slow, so we limit iterations.
    # Alternatively, use multiple random starts with Powell.
    
    # Let's try a few random starts with Powell, as it's often faster and good for smooth landscapes.
    # We will sample 20 random points in the parameter space and refine them.
    
    for _ in range(30):
        start = np.random.uniform(bounds[0][0], bounds[0][1], 1)
        start = np.append(start, np.random.uniform(bounds[1][0], bounds[1][1], 1))
        start = np.append(start, np.random.uniform(bounds[2][0], bounds[2][1], 1))
        start = np.append(start, np.random.uniform(bounds[3][0], bounds[3][1], 1))
        
        try:
            res = minimize(
                lambda p: evaluate(get_h(p)),
                start,
                method='Powell',
                options={'maxiter': 100, 'maxfev': 500, 'disp': False}
            )
            if res.fun < best_candidate_val:
                best_candidate_val = res.fun
                best_candidate = get_h(res.x).copy()
        except Exception:
            pass

    # Also try some structured points
    structured_starts = [
        [1, 0.0, 0.5, 0.0],
        [2, 0.0, 0.5, 0.0],
        [3, 0.0, 0.5, 0.0],
        [5, 0.0, 0.5, 0.0],
        [1, 0.5, 0.5, 0.0],
        [1, -0.5, 0.5, 0.0],
        [1, 0.0, 1.5, 0.0],
        [1, 0.0, 2.5, 0.0],
    ]
    
    for s in structured_starts:
        try:
            res = minimize(
                lambda p: evaluate(get_h(p)),
                np.array(s),
                method='Powell',
                options={'maxiter': 100, 'maxfev': 500, 'disp': False}
            )
            if res.fun < best_candidate_val:
                best_candidate_val = res.fun
                best_candidate = get_h(res.x).copy()
        except Exception:
            pass

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
