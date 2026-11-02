import numpy as np
import time

N = 400
DEADLINE = 115.0  # Safety margin below 150s evaluator limit

def _compute_r3(h):
    """
    Compute R3[k] = sum_i h[i] h[i+k] h[i+2k] for k = 0 .. N//2
    Using FFT for initialization speed.
    """
    h_float = h.astype(np.float64)
    size = 2 * N + 1
    h_fft = np.fft.fft(h_float, n=size)
    r3_fft = h_fft * h_fft * h_fft
    r3_full = np.fft.ifft(r3_fft).real
    return r3_full[:N+1]

def _score(h):
    r3 = _compute_r3(h)
    return np.max(np.abs(r3))

def _compute_r3_incremental(h, r3, i):
    """
    Flip sign at position i and update r3 incrementally.
    Returns updated r3 array.
    """
    h_float = h.astype(np.float64)
    # When we flip h[i], the change in R3[k] is:
    # R3_new[k] = R3_old[k] - 2 * h[i] * sum_j h[j+k] h[j+2k] where j, j+k, j+2k are valid and one of them is i
    # More precisely: for each k, the terms involving h[i] change.
    # Term h[a]*h[b]*h[c] where {a,b,c} = {i, i+k, i+2k} or {i, i-k, i-2k} etc.
    
    # For lag k, R3[k] = sum_j h[j] h[j+k] h[j+2k]
    # Flipping h[i] affects terms where i = j, i = j+k, or i = j+2k
    # Case 1: j = i, so terms h[i] h[i+k] h[i+2k] for valid i+k, i+2k
    # Case 2: j+k = i, so j = i-k, terms h[i-k] h[i] h[i+k] for valid i-k, i+k
    # Case 3: j+2k = i, so j = i-2k, terms h[i-2k] h[i-k] h[i] for valid i-2k, i-k
    
    # The change is: -2 * h[i] * (sum of the other two factors for each affected term)
    # Since h[i] flips sign, the product changes by factor -1, so delta = -2 * old_term_value
    
    # Actually: new_term = -old_term, so delta = new - old = -2*old_term
    # old_term = h[i] * h[i+k] * h[i+2k] etc.
    # So delta for each term = -2 * h[i] * h[i+k] * h[i+2k] (before flip)
    
    # But we need to be careful: only count each (j, j+k, j+2k) triple once per k.
    
    new_r3 = r3.copy()
    hi = h_float[i]
    
    for k in range(1, N//2 + 1):
        delta_k = 0.0
        
        # Case 1: j = i, need i+k < N and i+2k < N
        if i + 2*k < N:
            delta_k -= 2.0 * hi * h_float[i+k] * h_float[i+2*k]
        
        # Case 2: j = i-k, need i-k >= 0 and i+k < N
        if i - k >= 0 and i + k < N:
            delta_k -= 2.0 * hi * h_float[i-k] * h_float[i+k]
        
        # Case 3: j = i-2k, need i-2k >= 0 and i-k < N
        if i - 2*k >= 0 and i - k < N:
            delta_k -= 2.0 * hi * h_float[i-2*k] * h_float[i-k]
        
        new_r3[k] += delta_k
    
    # k=0: R3[0] = sum h[i]^3 = sum h[i] (since h[i]=±1)
    # Flipping h[i] changes R3[0] by -2*h[i]
    new_r3[0] -= 2.0 * hi
    
    return new_r3

def _greedy_descent(h, max_rounds=50):
    """
    Greedy coordinate descent: repeatedly flip the sign that most reduces the max |R3|.
    """
    h_float = h.astype(np.float64).copy()
    r3 = _compute_r3(h_float)
    current_score = np.max(np.abs(r3))
    best_score = current_score
    best_h = h_float.copy()
    
    start_time = time.time()
    
    for round_num in range(max_rounds):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try flipping position i
            new_r3 = _compute_r3_incremental(h_float, r3, i)
            new_score = np.max(np.abs(new_r3))
            
            if new_score < current_score:
                h_float[i] *= -1.0
                r3 = new_r3
                current_score = new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_float.copy()
        
        if not improved:
            break
    
    return best_h, best_score

def _random_restart_greedy(h_init, num_restarts=10):
    """
    Run greedy descent from multiple random perturbations of h_init.
    """
    best_h = h_init.astype(np.float64).copy()
    best_score = _score(best_h)
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Create a perturbation: flip a random subset of signs
        n_flips = np.random.randint(10, 50)
        flip_positions = np.random.choice(N, size=n_flips, replace=False)
        h_perturbed = best_h.copy()
        h_perturbed[flip_positions] *= -1.0
        
        h_greedy, score_greedy = _greedy_descent(h_perturbed, max_rounds=20)
        
        if score_greedy < best_score:
            best_score = score_greedy
            best_h = h_greedy.copy()
    
    return best_h, best_score

def construct():
    start_time = time.time()
    
    def elapsed_ok():
        return (time.time() - start_time) < DEADLINE
    
    # Start with a random initialization
    np.random.seed(42)
    best_h = np.random.choice([-1.0, 1.0], size=N)
    best_score = _score(best_h)
    
    # Run greedy descent from initial
    h_greedy, score_greedy = _greedy_descent(best_h, max_rounds=30)
    if score_greedy < best_score:
        best_score = score_greedy
        best_h = h_greedy.copy()
    
    # Try a few different initializations
    for init in range(3):
        if not elapsed_ok():
            break
        
        if init == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif init == 1:
            # Sine-based initialization
            t = np.linspace(0, 2 * np.pi, N)
            h = np.sign(np.sin(t))
            h[h == 0] = 1.0
        else:
            # Block-based initialization
            block_size = 8
            n_blocks = N // block_size
            blocks = np.random.choice([-1.0, 1.0], size=n_blocks)
            h = np.repeat(blocks, block_size)
        
        h_greedy, score_greedy = _greedy_descent(h, max_rounds=20)
        if score_greedy < best_score:
            best_score = score_greedy
            best_h = h_greedy.copy()
        
        if not elapsed_ok():
            break
    
    # Random restarts with greedy polish
    if elapsed_ok():
        h_final, score_final = _random_restart_greedy(best_h, num_restarts=8)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score(h))
