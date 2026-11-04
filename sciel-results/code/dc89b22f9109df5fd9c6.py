"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _objective(x):
    """
    Compute the L3 norm of the autocorrelation sidelobes.
    r_k = sum_{t=0}^{N-1} x_t x_{t+k mod N}
    We compute circular autocorrelation via FFT.
    """
    f = np.fft.fft(x, n=2*N)
    r = np.fft.ifft(f * np.conj(f))
    # r is real (within numerical precision)
    r = np.real(r)
    # Sidelobes are indices 1..N-1. 
    # Due to symmetry r[k] = r[N-k], we can just take 1..N-1.
    # The L3 norm is sum |r_k|^3 for k=1..N-1
    return np.sum(np.abs(r[1:N])**3)

def _local_search(x, deadline, max_passes=50):
    """
    Greedy local search: flip bits if it reduces the objective.
    """
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

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/- 1."""
    t = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        t[i] = t[i >> 1] ^ (i & 1)
    return 1 - 2*t # Convert 0/1 to 1/-1

def _legendre_truncated(prime, n):
    """Generate Legendre symbol sequence for 'prime', truncated to length n."""
    seq = np.ones(n, dtype=np.float64)
    for i in range(1, n):
        # Euler's criterion: (i|p) = i^((p-1)/2) mod p
        val = pow(i, (prime - 1) // 2, prime)
        if val == 1:
            seq[i] = 1
        elif val == prime - 1:
            seq[i] = -1
        else:
            seq[i] = 1 # Should not happen for 1 < i < p
    return seq

def construct():
    start_time = time.time()
    # 150s limit, keep 10s margin
    deadline = start_time + 140.0
    
    best_x = None
    best_obj = float('inf')
    
    # 1. Thue-Morse
    x_tm = _thue_morse(N)
    x_opt, obj = _local_search(x_tm, deadline)
    if obj < best_obj:
        best_obj = obj
        best_x = x_opt
        
    # 2. Legendre (p=401 is prime)
    x_leg = _legendre_truncated(401, N)
    x_opt, obj = _local_search(x_leg, deadline)
    if obj < best_obj:
        best_obj = obj
        best_x = x_opt
        
    # 3. Random initializations
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        x_rand = np.random.choice([-1, 1], N)
        x_opt, obj = _local_search(x_rand, deadline)
        if obj < best_obj:
            best_obj = obj
            best_x = x_opt
            
    # 4. If time permits, try a few more random restarts or a simple simulated annealing
    if time.time() < deadline - 20 and best_x is not None:
        # Simple SA-like refinement on the best found so far
        x = best_x.copy()
        current_obj = best_obj
        T = 1.0
        for step in range(10000):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            old_val = x[i]
            x[i] = -old_val
            new_obj = _objective(x)
            delta = new_obj - current_obj
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_obj = new_obj
            else:
                x[i] = old_val
            T *= 0.999
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x.copy()

    if best_x is None:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
