"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _objective(x):
    """Compute circular autocorrelation peak (max absolute sidelobe)."""
    f = np.fft.fft(x, n=2*N)
    r = np.fft.ifft(f * np.conj(f))
    return np.max(np.abs(r[1:N]))

def _legendre_sequence(prime):
    """Generate Legendre symbol sequence of length prime-1."""
    seq = np.zeros(prime - 1, dtype=np.float64)
    for i in range(1, prime):
        # Compute Legendre symbol (i | prime)
        if i % prime == 0:
            seq[i-1] = 0
        else:
            # Euler's criterion
            val = pow(i, (prime - 1) // 2, prime)
            if val == 1:
                seq[i-1] = 1
            elif val == prime - 1:
                seq[i-1] = -1
            else:
                seq[i-1] = 0
    return seq

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(n):
        # Count number of 1s in binary representation
        seq[i] = np.binary_repr(i).count('1') % 2
    # Convert to signed: 0 -> 1, 1 -> -1
    return 1.0 - 2.0 * seq.astype(np.float64)

def _local_search(x, deadline, max_passes=50):
    """Bit-flip local search with deadline guard."""
    x = x.copy()
    current_obj = _objective(x)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = x[i]
            x[i] = -old_val
            new_obj = _objective(x)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                x[i] = old_val
    return x, current_obj

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_x = None
    best_obj = float('inf')
    
    # Strategy 1: Legendre symbol sequences
    # Try primes close to 400
    primes_to_try = [401, 397, 389, 383, 379, 373, 367, 359, 353, 349]
    for p in primes_to_try:
        if time.time() > deadline - 5.0:
            break
        seq = _legendre_sequence(p)
        # Take first N elements, pad or trim as needed
        if len(seq) >= N:
            x = seq[:N].copy()
        else:
            # Pad with zeros (or repeat)
            x = np.zeros(N)
            x[:len(seq)] = seq
            # Fill remaining with 1s
            x[len(seq):] = 1.0
        
        # Replace zeros with 1
        x[x == 0] = 1.0
        
        x_opt, obj = _local_search(x, deadline)
        if obj < best_obj:
            best_obj = obj
            best_x = x_opt
    
    # Strategy 2: Thue-Morse sequence
    if time.time() < deadline - 5.0:
        x_tm = _thue_morse(N)
        x_opt, obj = _local_search(x_tm, deadline)
        if obj < best_obj:
            best_obj = obj
            best_x = x_opt
    
    # Strategy 3: Quadratic residue with phase shifts
    if time.time() < deadline - 10.0:
        p = 401
        base_seq = _legendre_sequence(p)
        base_seq[base_seq == 0] = 1.0
        for shift in range(5):
            if time.time() > deadline - 5.0:
                break
            # Circular shift
            x = np.roll(base_seq[:N], shift)
            x_opt, obj = _local_search(x, deadline)
            if obj < best_obj:
                best_obj = obj
                best_x = x_opt
    
    # Strategy 4: Random initializations if time permits
    if time.time() < deadline - 10.0:
        np.random.seed(42)
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            x = np.random.choice([-1, 1], N).astype(np.float64)
            x_opt, obj = _local_search(x, deadline)
            if obj < best_obj:
                best_obj = obj
                best_x = x_opt
    
    if best_x is None:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
