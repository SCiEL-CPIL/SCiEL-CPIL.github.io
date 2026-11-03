"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv(x):
    """Compute autoconvolution of x using FFT. x is length N."""
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * X, size)[:n]
    return ac

def _score(x):
    """Score: max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Start with random sign pattern
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x)
        if abs(s) < 1e-6:
            continue
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Local search: flip individual elements to minimize score
    # Use gradient-like approach: for each element, try flipping and see if score improves
    # Since N=400, we can afford several passes
    
    current_x = best_x.copy()
    current_score = best_score
    
    # Precompute for efficiency
    def try_flip(x, i):
        """Return new score if we flip element i."""
        x_new = x.copy()
        x_new[i] = -x_new[i]
        return _score(x_new)
    
    # Multiple passes of local search
    max_passes = 50
    for p in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        # Random order to avoid bias
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            
            new_x = current_x.copy()
            new_x[i] = -new_x[i]
            new_score = _score(new_x)
            
            if new_score < current_score - 1e-12:
                current_x = new_x
                current_score = new_score
                improved = True
        
        if not improved:
            break
        
        if current_score < best_score:
            best_score = current_score
            best_x = current_x.copy()
    
    # Try a few more random restarts with local search if time permits
    for _ in range(10):
        if time.time() > deadline:
            break
        
        x = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x)
        if abs(s) < 1e-6:
            continue
        sc = _score(x)
        
        # Quick local search on this initialization
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = False
            order = rng.permutation(N)
            for i in order:
                new_x = x.copy()
                new_x[i] = -new_x[i]
                new_score = _score(new_x)
                if new_score < sc - 1e-12:
                    x = new_x
                    sc = new_score
                    improved = True
            if not improved:
                break
        
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    return best_x
# EVOLVE-BLOCK-END
