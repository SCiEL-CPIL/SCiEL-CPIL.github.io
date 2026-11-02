import numpy as np
import time

N = 400
DEADLINE = 120.0
SAFETY_MARGIN = 10.0

def _autocorr_max_abs(s):
    """
    Compute the maximum absolute value of the non-zero lag autocorrelation.
    This is the metric used in the current best-scoring program.
    """
    n_fft = 2 * N
    S = np.fft.fft(s, n=n_fft)
    r = np.fft.ifft(S * np.conj(S)).real
    # r[0] is lag 0. r[1:N] are positive lags.
    # Due to symmetry, max(abs(r[1:N])) covers the peak sidelobe.
    return np.max(np.abs(r[1:N]))

def _try_pairwise_swap(s, rng, current_val):
    """
    Attempt a random pairwise swap of signs.
    Returns (new_s, new_val, accepted)
    """
    i, j = rng.choice(N, size=2, replace=False)
    if s[i] == s[j]:
        return s, current_val, False
    
    s[i], s[j] = s[j], s[i]
    new_val = _autocorr_max_abs(s)
    
    if new_val < current_val:
        return s, new_val, True
    else:
        s[i], s[j] = s[j], s[i]
        return s, current_val, False

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    rng = np.random.default_rng(12345)
    
    best_s = None
    best_val = np.inf
    
    # 1. Generate Diverse Initializations (from Inspiration 1)
    x = np.linspace(-1/4, 1/4, N, endpoint=False)
    inits = []
    
    # Random
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    # Sine patterns
    for freq in [2, 4, 8, 16]:
        p = np.sign(np.cos(2 * np.pi * freq * x))
        p[p == 0] = 1.0
        inits.append(p)
        
    # Block structures
    p_block = np.ones(N)
    p_block[N//4:N//2] = -1
    p_block[3*N//4:] = -1
    inits.append(p_block)
    
    # Alternating
    p_alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    inits.append(p_alt)

    # 2. Evaluate and Optimize each initialization
    for init_s in inits:
        if time.time() > deadline:
            break
            
        cur_s = init_s.copy()
        cur_val = _autocorr_max_abs(cur_s)
        
        if cur_val < best_val:
            best_val = cur_val
            best_s = cur_s.copy()
            
        # Simulated Annealing Phase
        # Use a fraction of the total time for SA per restart
        # We don't know how many inits will fit, so use a time check inside the loop
        T = 1.0
        T_min = 0.005
        cooling = 0.9995
        steps_this_restart = 0
        
        # Estimate steps per second roughly to set a cap? 
        # Better: just run until time limit for this specific restart portion?
        # No, we have multiple inits. 
        # Strategy: Allocate a fixed time budget per init, or just loop until global deadline.
        # Given N=400, FFT is fast. ~1000 steps/sec is reasonable.
        
        while time.time() < deadline - 20: # Leave 20s for polish
            block_size = int(rng.integers(1, 21)) # Smaller blocks for finer tuning
            if block_size >= N:
                block_size = N - 1
            start = int(rng.integers(0, N - block_size + 1))
            
            old_block = cur_s[start:start+block_size].copy()
            cur_s[start:start+block_size] = -old_block
            
            new_val = _autocorr_max_abs(cur_s)
            delta = new_val - cur_val
            
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                cur_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_s = cur_s.copy()
            else:
                cur_s[start:start+block_size] = old_block
            
            steps_this_restart += 1
            T = max(T_min, T * cooling)
            
            # Occasional pairwise swap during SA to escape local minima
            if steps_this_restart % 50 == 0:
                cur_s, cur_val, _ = _try_pairwise_swap(cur_s, rng, cur_val)
                if cur_val < best_val:
                    best_val = cur_val
                    best_s = cur_s.copy()
                    
        # Polish Phase: Greedy Single Bit + Pairwise Swaps
        # Run for the remaining time allocated to this init or until no improvement
        polish_deadline = deadline - 5
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            # 1. Greedy Single Bit Flips
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                old = cur_s[i]
                cur_s[i] = -old
                nv = _autocorr_max_abs(cur_s)
                if nv < cur_val:
                    cur_val = nv
                    if nv < best_val:
                        best_val = nv
                        best_s = cur_s.copy()
                    improved = True
                else:
                    cur_s[i] = old
                    
            # 2. Random Pairwise Swaps (limited attempts)
            if time.time() < polish_deadline:
                for _ in range(200):
                    if time.time() > polish_deadline:
                        break
                    cur_s, nv, accepted = _try_pairwise_swap(cur_s, rng, cur_val)
                    if accepted:
                        cur_val = nv
                        if nv < best_val:
                            best_val = nv
                            best_s = cur_s.copy()
                        improved = True

    if best_s is None:
        best_s = np.ones(N)
    
    return best_s
