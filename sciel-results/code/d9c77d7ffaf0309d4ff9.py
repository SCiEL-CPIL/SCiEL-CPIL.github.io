"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # Safety margin before 150s kill

def _score(f):
    """Compute the C3 upper bound: max |autoconvolution| at non-zero lags."""
    n = len(f)
    # Direct convolution: O(n^2)
    # conv[k] = sum_i f[i] * f[i+k] for k=0..n-1
    # We need max |conv[k]| for k=1..n-1
    # Using matrix approach for speed
    F = np.asarray(f, dtype=np.float64)
    # conv[k] = F @ shift(F, k)
    # Build the convolution matrix
    # For k in 1..n-1: conv[k] = sum_{i=0}^{n-1-k} F[i]*F[i+k]
    max_val = 0.0
    for k in range(1, n):
        val = np.dot(F[:n-k], F[k:])
        if val < 0:
            val = -val
        if val > max_val:
            max_val = val
    return max_val

def _score_fast(f):
    """Faster score using FFT when beneficial, direct otherwise."""
    n = len(f)
    F = np.asarray(f, dtype=np.float64)
    # Use FFT for speed
    size = 1
    while size < 2 * n - 1:
        size *= 2
    Ff = np.fft.rfft(F, n=size)
    conv = np.fft.irfft(Ff * Ff, n=size)
    # conv[k] for k=0..n-1
    max_val = 0.0
    for k in range(1, n):
        val = conv[k]
        if val < 0:
            val = -val
        if val > max_val:
            max_val = val
    return max_val

def _rule90_sequence(seed, length):
    """Generate a Rule 90 cellular automaton sequence."""
    # Rule 90: cell(i, t+1) = cell(i-1, t) XOR cell(i+1, t)
    # Start with a single 1 at position seed
    state = np.zeros(length, dtype=np.int8)
    state[seed % length] = 1
    
    sequences = []
    for step in range(10):
        sequences.append(state.copy())
        # Apply Rule 90
        new_state = np.zeros(length, dtype=np.int8)
        new_state[1:] = state[:-1] ^ state[2:]
        new_state[:-1] = state[1:] ^ state[:-2]
        new_state[0] = state[1]
        new_state[-1] = state[-2]
        state = new_state
    
    return sequences

def _thue_morse(length):
    """Generate Thue-Morse sequence."""
    seq = np.zeros(length, dtype=np.int8)
    for i in range(length):
        # Count bits
        seq[i] = bin(i).count('1') % 2
    return seq

def _local_search(f, active_indices, deadline, max_rounds=200):
    """
    Greedy local search with batch sign flips.
    Returns updated f and score.
    """
    current_score = _score(f)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Try flipping each active index
        for idx in active_indices:
            if time.time() > deadline:
                break
            old_val = f[idx]
            f[idx] = -old_val
            new_score = _score(f)
            if new_score < current_score - 1e-10:
                current_score = new_score
                improved = True
            else:
                f[idx] = old_val
        
        if not improved:
            break
    
    return f, current_score

def _gradient_optimize(f, active_indices, deadline, lr=0.5, max_iter=500):
    """
    Continuous gradient-based optimization with binary projection.
    """
    n = len(f)
    current_score = _score(f)
    best_f = f.copy()
    best_score = current_score
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute gradient numerically
        grad = np.zeros(n)
        eps = 0.01
        for idx in active_indices:
            f[idx] += eps
            s_plus = _score(f)
            f[idx] -= 2 * eps
            s_minus = _score(f)
            f[idx] += eps
            grad[idx] = (s_plus - s_minus) / (2 * eps)
        
        # Gradient step
        f[active_indices] -= lr * grad[active_indices]
        
        # Project to binary: round to nearest sign
        for idx in active_indices:
            f[idx] = 1.0 if f[idx] >= 0 else -1.0
        
        new_score = _score(f)
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
        
        # Decay learning rate
        lr *= 0.99
    
    return best_f, best_score

def _multi_start_search(deadline, num_starts=5):
    """
    Multi-start search with different initialization strategies.
    """
    best_f = np.zeros(N)
    best_score = np.inf
    
    start_time = time.time()
    
    # Strategy 1: Rule 90 sequences
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        
        for step in range(5):
            if time.time() > deadline - 20:
                break
            
            seq = _rule90_sequence(seed, N)[step]
            f = seq.astype(np.float64)
            # Map 0/1 to -1/1
            f = 2.0 * f - 1.0
            
            active_indices = np.nonzero(f)[0]
            if len(active_indices) == 0:
                continue
            
            # Local search
            f, score = _local_search(f, active_indices, deadline - 10, max_rounds=50)
            
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    # Strategy 2: Thue-Morse
    if time.time() < deadline - 20:
        tm = _thue_morse(N)
        f = 2.0 * tm.astype(np.float64) - 1.0
        active_indices = np.nonzero(f)[0]
        
        f, score = _local_search(f, active_indices, deadline - 10, max_rounds=100)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Strategy 3: Random sparse
    rng = np.random.RandomState(42)
    for K in [20, 40, 60, 80]:
        if time.time() > deadline - 20:
            break
        
        active_indices = rng.choice(N, size=K, replace=False)
        signs = rng.choice([-1.0, 1.0], size=K)
        
        f = np.zeros(N)
        f[active_indices] = signs
        
        f, score = _local_search(f, active_indices, deadline - 10, max_rounds=100)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Gradient optimization on best
    if time.time() < deadline - 5:
        active = np.nonzero(best_f)[0]
        if len(active) > 0:
            best_f, best_score = _gradient_optimize(best_f, active, deadline - 2, lr=0.3, max_iter=200)
    
    # Final polish
    if time.time() < deadline - 1:
        active = np.nonzero(best_f)[0]
        if len(active) > 0:
            best_f, _ = _local_search(best_f, active, deadline, max_rounds=50)
    
    return best_f

def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_f = _multi_start_search(deadline)
    
    return best_f

# EVOLVE-BLOCK-END
