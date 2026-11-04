"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_R(f):
    """Compute R[L] for L=0..N-1 where R[L] = sum_{i=0}^{N-L-1} f[i]*f[i+L]."""
    n = len(f)
    R = np.zeros(n)
    for L in range(n):
        R[L] = np.dot(f[:n-L], f[L:])
    return R

def _score_l3(R):
    """L3 norm of autocorrelation (excluding L=0): sum of R[L]^3 for L=1..N-1."""
    return np.sum(R[1:] ** 3)

def _score_linf(R):
    """L-infinity norm: max |R[L]| for L=1..N-1."""
    return np.max(np.abs(R[1:]))

def _thue_morse_init(n):
    """Generate Thue-Morse sequence of length n."""
    h = np.zeros(n)
    for i in range(n):
        count = bin(i).count('1')
        h[i] = 1.0 if count % 2 == 0 else -1.0
    return h

def _local_search_l3(f, time_limit, rng):
    """Local search minimizing L3 norm of autocorrelation with incremental updates."""
    n = len(f)
    f = f.copy()
    R = _compute_R(f)
    current_score = _score_l3(R)
    best_score = current_score
    best_f = f.copy()
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        if no_improve >= max_no_improve:
            break
        
        # Find the lag with largest |R[L]| to target
        L_max = np.argmax(np.abs(R[1:])) + 1
        
        # Try flipping each position, compute delta in L3 score
        improved = False
        indices = np.arange(n)
        rng.shuffle(indices)
        
        for k in indices:
            if time.time() - start_time > time_limit:
                break
            
            old_val = f[k]
            new_val = -old_val
            
            # Compute delta in R for each lag
            # delta_R[L] = -2 * old_val * (f[k-L] if k>=L else 0 + f[k+L] if k+L<n else 0)
            delta_R = np.zeros(n)
            if k > 0:
                # For L = 1..k: f[k-L]
                delta_R[1:k+1] = -2 * old_val * f[k-1::-1]
            if k + 1 < n:
                # For L = 1..n-k-1: f[k+L]
                delta_R[1:n-k] += -2 * old_val * f[k+1:]
            
            # New R
            R_new = R + delta_R
            new_score = _score_l3(R_new)
            
            if new_score < current_score - 1e-9:
                # Accept
                f[k] = new_val
                R = R_new
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                no_improve = 0
                break
            # else: don't change, move to next k
        
        if not improved:
            no_improve += 1
    
    return best_f, best_score

def _local_search_linf(f, time_limit, rng):
    """Local search minimizing L-infinity norm of autocorrelation."""
    n = len(f)
    f = f.copy()
    R = _compute_R(f)
    current_peak = _score_linf(R)
    best_peak = current_peak
    best_f = f.copy()
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        if no_improve >= max_no_improve:
            break
        
        L_max = np.argmax(np.abs(R[1:])) + 1
        
        improved = False
        indices = np.arange(n)
        rng.shuffle(indices)
        
        for k in indices:
            if time.time() - start_time > time_limit:
                break
            
            old_val = f[k]
            new_val = -old_val
            
            delta_R = np.zeros(n)
            if k > 0:
                delta_R[1:k+1] = -2 * old_val * f[k-1::-1]
            if k + 1 < n:
                delta_R[1:n-k] += -2 * old_val * f[k+1:]
            
            R_new = R + delta_R
            new_peak = _score_linf(R_new)
            
            if new_peak < current_peak - 1e-12:
                f[k] = new_val
                R = R_new
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_f = f.copy()
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
    
    return best_f, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.RandomState(42)
    
    best_f = None
    best_l3 = np.inf
    best_linf = np.inf
    
    # Generate diverse initializations
    inits = []
    
    # 1. All +1 (uniform) - likely optimal for L3
    inits.append(np.ones(N))
    
    # 2. Thue-Morse
    inits.append(_thue_morse_init(N))
    
    # 3. Random sequences
    for seed in range(10):
        r = np.random.RandomState(seed)
        inits.append(r.choice([-1.0, 1.0], size=N))
    
    # 4. Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1.0
    inits.append(h_alt)
    
    # 5. Block patterns
    for block_size in [4, 8, 16, 32]:
        h_block = np.ones(N)
        for i in range(N):
            if (i // block_size) % 2 == 1:
                h_block[i] = -1.0
        inits.append(h_block)
    
    # Evaluate all initializations
    init_scores = []
    for h in inits:
        R = _compute_R(h)
        l3 = _score_l3(R)
        linf = _score_linf(R)
        init_scores.append((l3, linf, h))
    
    # Sort by L3 score
    init_scores.sort(key=lambda x: x[0])
    
    # Run local search on top candidates for L3 optimization
    for i, (l3_init, linf_init, h_init) in enumerate(init_scores[:5]):
        if time.time() >= deadline - 5.0:
            break
        
        remaining = deadline - time.time() - 2.0
        if remaining <= 0:
            break
        
        h_opt, l3_opt = _local_search_l3(h_init, remaining, rng)
        
        if l3_opt < best_l3:
            best_l3 = l3_opt
            best_f = h_opt
    
    # Also try L-infinity optimization on the best L3 result
    if best_f is not None and time.time() < deadline - 5.0:
        h_linf, linf_opt = _local_search_linf(best_f, deadline - time.time() - 2.0, rng)
        # Check if L-inf optimization also improved L3
        R_check = _compute_R(h_linf)
        l3_check = _score_l3(R_check)
        if l3_check < best_l3:
            best_l3 = l3_check
            best_f = h_linf
    
    # Final polish
    if best_f is not None and time.time() < deadline - 1.0:
        h_final, l3_final = _local_search_l3(best_f, deadline - time.time() - 0.5, rng)
        if l3_final < best_l3:
            best_l3 = l3_final
            best_f = h_final
    
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f

# EVOLVE-BLOCK-END
