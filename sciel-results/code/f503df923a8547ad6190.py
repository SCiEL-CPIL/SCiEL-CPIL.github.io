import numpy as np
import time

N = 400

def _compute_score(h):
    """Compute the maximum absolute autocorrelation at non-zero lags, normalized."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def _linear_autocorr(h):
    """Compute linear (non-circular) autocorrelation for all lags 1..N-1."""
    # Use FFT for efficiency
    n = len(h)
    # Zero-pad to avoid circular effects
    size = 2 * n
    H = np.fft.fft(h, n=size)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[k] for k=1..n-1 gives the linear autocorrelation at lag k
    return R[1:n]

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    # Thue-Morse: t(n) = popcount(n) mod 2
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bits[i] = bin(i).count('1') % 2
    return 1 - 2 * bits  # map 0->1, 1->-1

def _m_sequence(n, seed=0):
    """Generate an m-sequence-like pattern using a simple LFSR-like construction."""
    # Use a known good initial pattern and apply a simple transformation
    # For n=400, we'll use a structured pattern
    h = np.ones(n, dtype=np.float64)
    # Create a pattern based on quadratic residues mod a prime near n
    p = 401  # prime
    for i in range(n):
        # Quadratic residue test mod p
        if i == 0:
            h[i] = -1.0
        else:
            # i^2 mod p, check if it's a quadratic residue
            # Use Euler's criterion: a^((p-1)/2) mod p
            if pow(i, (p-1)//2, p) == 1:
                h[i] = 1.0
            else:
                h[i] = -1.0
    return h

def _structured_init():
    """Create a diverse set of structured initializations."""
    inits = []
    
    # Thue-Morse
    inits.append(_thue_morse(N))
    
    # M-sequence-like (quadratic residues)
    inits.append(_m_sequence(N))
    
    # Alternating blocks
    h = np.ones(N)
    for i in range(N):
        block = i // 10
        if block % 2 == 1:
            h[i] = -1
    inits.append(h)
    
    # Sine-based
    x = np.linspace(0, 4*np.pi, N)
    inits.append(np.sign(np.sin(x)))
    
    # Random
    rng = np.random.RandomState(42)
    inits.append(rng.choice([-1.0, 1.0], size=N))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    # Try multiple structured initializations
    inits = _structured_init()
    
    best_h = None
    best_score = float('inf')
    
    # Evaluate all initializations
    for h in inits:
        score = _compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        return np.ones(N)
    
    # Simulated annealing with hybrid moves
    rng = np.random.RandomState(12345)
    
    current_h = best_h.copy()
    current_score = best_score
    best_h = current_h.copy()
    best_score = current_score
    
    T = 1.0  # initial temperature
    T_min = 0.001
    cooling = 0.999
    
    # Precompute for efficiency
    n = N
    
    while time.time() < deadline:
        T = max(T_min, T * cooling)
        
        # Choose move type: flip (60%), swap (25%), adjacent swap (15%)
        move_type = rng.random()
        
        if move_type < 0.6:
            # Flip a single bit
            pos = rng.randint(n)
            new_h = current_h.copy()
            new_h[pos] *= -1
        elif move_type < 0.85:
            # Swap two random positions
            pos1, pos2 = rng.choice(n, size=2, replace=False)
            new_h = current_h.copy()
            new_h[pos1], new_h[pos2] = new_h[pos2], new_h[pos1]
        else:
            # Adjacent swap
            pos1 = rng.randint(n)
            pos2 = (pos1 + 1) % n
            new_h = current_h.copy()
            new_h[pos1], new_h[pos2] = new_h[pos2], new_h[pos1]
        
        new_score = _compute_score(new_h)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            current_h = new_h
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        # Every 1000 iterations, try a larger perturbation if stuck
        # (implicit via temperature)
    
    return best_h
