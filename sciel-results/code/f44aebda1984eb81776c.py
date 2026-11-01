import numpy as np
import time

N = 400
M = 2 * N  # FFT size for full autocorrelation

def _calc_score_fft(h):
    """
    Calculate the score using FFT-based autocorrelation.
    Score = max_{k != 0} |C[k]| / ||h||^2
    """
    h_padded = np.zeros(M)
    h_padded[:N] = h
    
    H = np.fft.fft(h_padded)
    C = np.fft.ifft(H * np.conj(H)).real
    
    # C[0] is lag 0, C[1..N-1] are positive lags
    # We want max |C[k]| for k != 0
    # Since C is real and symmetric, we can just look at C[1:N]
    peak = np.max(np.abs(C[1:N]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _calc_score(h):
    """
    Wrapper for backward compatibility if needed, but we use FFT version internally.
    """
    return _calc_score_fft(h)

def _local_search(h, max_iter=50, time_limit=1.0):
    """
    Discrete local search: flip bits to reduce the score.
    Uses a time limit to ensure it doesn't run too long.
    """
    h = h.copy()
    current_score = _calc_score_fft(h)
    start_time = time.time()
    
    # Precompute indices for faster access? No, just simple loop.
    # To speed up, we can try a random subset of indices or all indices.
    # With N=400, a full pass is 400 FFTs.
    # FFT of size 800 is ~800*log(800) ~ 8000 ops. 400 * 8000 = 3.2M ops.
    # In Python/NumPy, this is quite fast, maybe < 0.1s per pass.
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            h[i] *= -1
            new_score = _calc_score_fft(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                h[i] *= -1  # Revert
        
        if not improved:
            break
            
    return h, current_score

def _random_restart_local_search(num_restarts=20, time_per_restart=1.0):
    """
    Perform multiple random restarts with local search.
    """
    best_h = None
    best_score = float('inf')
    
    for r in range(num_restarts):
        h_init = np.random.choice([-1, 1], N)
        h_refined, score = _local_search(h_init, max_iter=20, time_limit=time_per_restart)
        
        if score < best_score:
            best_score = score
            best_h = h_refined.copy()
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    # Strategy:
    # 1. Start with a few random restarts to get a baseline.
    # 2. Use simulated annealing or more aggressive local search if time permits.
    
    # Phase 1: Random restarts
    # We have ~140 seconds.
    # Let's try to fit as many restarts as possible.
    # Each restart takes ~1-2 seconds.
    # So we can do ~70-100 restarts.
    
    while time.time() < deadline - 10:
        h_init = np.random.choice([-1, 1], N)
        h_refined, score = _local_search(h_init, max_iter=10, time_limit=1.5)
        
        if score < best_score:
            best_score = score
            best_h = h_refined.copy()
            
    # Fallback
    if best_h is None:
        best_h = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
        best_score = _calc_score_fft(best_h)

    return best_h
