import numpy as np
import time
import math

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _ising_mean_field_init(beta, power_law_exp=1.5, rng=None):
    """
    Generate initial sign sequence using Ising model with power-law interactions.
    
    The coupling J[i,j] is derived from the inverse Fourier transform of a 
    flat-spectrum target. We use a mean-field approximation where each spin
    feels the average field from all others weighted by power-law coupling.
    
    Parameters:
    - beta: inverse temperature (higher = more ordered)
    - power_law_exp: exponent for long-range interactions
    - rng: numpy random generator
    
    Returns:
    - s: array of +/- 1 signs
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Power-law coupling: J[i,j] ~ |i-j|^(-alpha) for i != j
    # This mimics long-range frustrated magnetism
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    # Avoid division by zero on diagonal
    dist_safe = np.where(dist > 0, dist, 1.0)
    J = np.where(dist > 0, dist_safe ** (-power_law_exp), 0.0)
    
    # Normalize coupling
    J_max = np.max(J)
    if J_max > 0:
        J = J / J_max
    
    # Mean-field: h_i = sum_j J[i,j] * <s_j>
    # Start with random spins
    s = rng.choice([-1.0, 1.0], size=N)
    
    # Iterate mean-field updates
    for iteration in range(200):
        # Compute mean field at each site
        h = J @ s  # h_i = sum_j J[i,j] * s_j
        
        # Update spins with Metropolis-like acceptance
        # P(flip) = 1 / (1 + exp(-2 * beta * s_i * h_i))
        energy_change = -2.0 * s * h  # delta E if we flip s_i
        accept_prob = 1.0 / (1.0 + np.exp(energy_change * beta))
        
        # Stochastic update
        flip_mask = rng.random(N) < accept_prob
        s[flip_mask] *= -1.0
    
    return s

def _block_flip_local_search(s, deadline, max_block_sizes=None):
    """
    Local search using block flips of varying sizes.
    This explores larger regions of sign space than single-bit flips.
    """
    if max_block_sizes is None:
        max_block_sizes = [1, 2, 4, 8, 16, 32]
    
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    
    local_start = time.time()
    improved = True
    passes = 0
    max_passes = 40
    
    while improved and passes < max_passes and time.time() - local_start < 12.0 and time.time() < deadline:
        improved = False
        passes += 1
        
        # Try different block sizes
        for block_size in max_block_sizes:
            if time.time() > deadline:
                break
            
            n_blocks = N // block_size
            if n_blocks < 1:
                continue
            
            # Randomly select blocks to flip
            n_attempts = min(50, n_blocks)
            for _ in range(n_attempts):
                if time.time() > deadline:
                    break
                
                # Select a random block
                block_start = np.random.randint(0, N - block_size + 1)
                block_end = block_start + block_size
                
                # Flip the block
                s[block_start:block_end] *= -1.0
                
                A_new = _calc_acf(s)
                val_new = _max_abs_acf_from_array(A_new)
                
                if val_new < current_val:
                    current_val = val_new
                    A = A_new
                    improved = True
                else:
                    # Revert
                    s[block_start:block_end] *= -1.0
    
    return s, current_val

def _simulated_annealing(s, deadline, t_start=1.0, t_end=0.01, cooling=0.97):
    """
    Simulated annealing with block-flip moves.
    """
    A = _calc_acf(s)
    current_val = _max_abs_acf_from_array(A)
    best_val = current_val
    best_s = s.copy()
    
    local_start = time.time()
    t = t_start
    max_block_sizes = [4, 8, 16, 32]
    
    while time.time() < deadline and t > t_end:
        if time.time() - local_start > 15.0:
            break
        
        # Select random block
        block_size = np.random.choice(max_block_sizes)
        block_start = np.random.randint(0, N - block_size + 1)
        block_end = block_start + block_size
        
        # Flip the block
        s[block_start:block_end] *= -1.0
        
        A_new = _calc_acf(s)
        val_new = _max_abs_acf_from_array(A_new)
        
        # Metropolis acceptance
        delta = val_new - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            current_val = val_new
            if val_new < best_val:
                best_val = val_new
                best_s = s.copy()
        else:
            # Revert
            s[block_start:block_end] *= -1.0
        
        t *= cooling
    
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Ising mean-field initialization with various parameters
    beta_values = [0.5, 1.0, 2.0, 4.0]
    power_law_exps = [1.0, 1.5, 2.0]
    
    for beta in beta_values:
        if time.time() > deadline - 60.0:
            break
        
        for alpha in power_law_exps:
            if time.time() > deadline - 60.0:
                break
            
            s = _ising_mean_field_init(beta, alpha, rng)
            
            # Refine with block-flip local search
            s, current_val = _block_flip_local_search(s, deadline - 30.0)
            
            if current_val < best_val:
                best_val = current_val
                best_seq = s.copy()
    
    # Phase 2: Simulated annealing from best solution and random restarts
    if time.time() < deadline - 20.0:
        # Continue from best
        s_anneal, val_anneal = _simulated_annealing(best_seq.copy(), deadline - 10.0)
        if val_anneal < best_val:
            best_val = val_anneal
            best_seq = s_anneal.copy()
    
    # Phase 3: Random restarts with quick local search
    while time.time() < deadline - 5.0:
        s = rng.choice([-1.0, 1.0], size=N)
        s, current_val = _block_flip_local_search(s, deadline - 5.0)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
    
    return best_seq
