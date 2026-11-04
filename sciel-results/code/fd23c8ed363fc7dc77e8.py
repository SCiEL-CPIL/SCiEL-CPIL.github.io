import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Fast PSL calculation using FFT
    # c = ifft(|fft(h)|^2)
    # Note: np.fft.correlate is not available in all versions, but we can do:
    # H = np.fft.fft(h, n=2*N)
    # c = np.fft.ifft(H * np.conj(H)).real
    # The correlation lags are c[0:N] and c[N:2*N-1] (with appropriate indexing)
    # Actually, for aperiodic autocorrelation of length N:
    # Zero-pad h to length 2N.
    # h_padded = np.zeros(2*N)
    # h_padded[:N] = h
    # H = np.fft.fft(h_padded)
    # c_full = np.fft.ifft(H * np.conj(H)).real
    # The main lobe is at c_full[0] (lag 0).
    # Lags 1..N-1 are c_full[1:N] and c_full[N+1:2*N] (lag -1 is at 2*N-1, etc.)
    # Wait, the standard ifft of power spectrum gives circular correlation.
    # For aperiodic, we need to extract the linear part.
    # If h is zero-padded to 2N, the circular correlation of length 2N
    # contains the linear correlation in the first N and the last N-1 elements.
    # Specifically, c[k] for k=0..N-1 is in c_full[0:N].
    # c[-k] for k=1..N-1 is in c_full[2*N-k].
    # So sidelobes are c_full[1:N] and c_full[N+1:2*N].
    
    def get_psl(h):
        # h is float array of +/- 1
        h_pad = np.zeros(2 * N)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        c = np.fft.ifft(H * np.conj(H)).real
        # Main lobe is c[0]
        # Sidelobes: lags 1 to N-1 are c[1:N]
        # lags -1 to -(N-1) are c[N+1:2*N]
        # Note: c[N] is lag N, which is 0 for zero-padded? 
        # Actually, for length N signal, max lag is N-1.
        # So we take c[1:N] and c[N+1:2*N]
        # Let's verify indices.
        # c_full[0] is lag 0.
        # c_full[1] is lag 1.
        # ...
        # c_full[N-1] is lag N-1.
        # c_full[N] is lag N (which is 0 because h[N] is 0).
        # c_full[N+1] is lag N+1? No, in circular correlation of size 2N,
        # index i corresponds to lag i if i < N, and lag i-2N if i >= N.
        # So c_full[N+1] is lag (N+1) - 2N = -N+1 = -(N-1). Correct.
        # So sidelobes are c[1:N] and c[N+1:2*N].
        
        left = c[1:N]
        right = c[N+1:2*N]
        if len(left) == 0 and len(right) == 0:
            return 0.0
        return np.max(np.abs(np.concatenate([left, right])))

    def generate_chirp(f1, f2):
        # f1: linear phase frequency
        # f2: quadratic phase coefficient
        t = np.arange(N)
        # Signal: cos(2*pi*f1*t + pi*f2*t^2)
        # We want the sign of this.
        # To avoid zeros, add a tiny epsilon or just use sign directly (sign(0)=0, so map to 1)
        phase = 2 * np.pi * f1 * t + np.pi * f2 * t**2
        sig = np.cos(phase)
        # Map to +/- 1
        h = np.sign(sig)
        h[h == 0] = 1.0
        return h

    # Initialize best solution
    best_h = None
    best_cost = float('inf')
    
    rng = np.random.RandomState(42)

    # 1. Try several Chirp initializations
    # f1 in [0, 1), f2 in [0, 1)
    for _ in range(50):
        f1 = rng.rand()
        f2 = rng.rand()
        h = generate_chirp(f1, f2)
        cost = get_psl(h)
        if cost < best_cost:
            best_cost = cost
            best_h = h.copy()
            
    # Also try random
    for _ in range(50):
        h = rng.choice([-1, 1], size=N).astype(np.float64)
        cost = get_psl(h)
        if cost < best_cost:
            best_cost = cost
            best_h = h.copy()

    # 2. Simulated Annealing / Local Search
    # We will run multiple SA instances, each with a time budget.
    # Current best is the starting point.
    
    current_h = best_h.copy()
    current_cost = best_cost
    
    # SA Parameters
    # T should be related to the typical delta. PSL deltas are small integers or 0.
    # Let's start T=1.0 or 2.0.
    
    num_restarts = 0
    max_restarts = 1000
    
    while time.time() < deadline and num_restarts < max_restarts:
        # Restart logic: If we are stuck, we might re-seed or just continue.
        # Let's do a fixed number of SA steps per "epoch" and then check time.
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999  # Slow cooling for more exploration
        
        # Run SA for a short burst
        steps_in_burst = 10000
        burst_start = time.time()
        
        for _ in range(steps_in_burst):
            if time.time() > deadline:
                break
            
            # Pick random index
            i = rng.randint(N)
            old_val = current_h[i]
            current_h[i] = -old_val
            
            new_cost = get_psl(current_h)
            
            delta = new_cost - current_cost
            
            # Accept if better or with prob exp(-delta/T)
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_h = current_h.copy()
            else:
                current_h[i] = old_val
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # End of burst
        # If we improved, keep going with the new best.
        # If not, we might want to restart from a perturbed best or a new chirp.
        
        # Check if we should restart from a fresh chirp or random
        # If best_cost hasn't improved in a long time, try a new seed
        # For now, just continue SA from current state, but occasionally re-seed.
        
        # Simple heuristic: Every 10 bursts, try a new random/chirp start if we haven't improved
        num_restarts += 1
        if num_restarts % 10 == 0:
             # Try a new chirp
             f1 = rng.rand()
             f2 = rng.rand()
             h_new = generate_chirp(f1, f2)
             cost_new = get_psl(h_new)
             if cost_new < best_cost:
                 best_cost = cost_new
                 best_h = h_new.copy()
                 current_h = best_h.copy()
                 current_cost = best_cost
             else:
                 # Stay with current best, but maybe perturb it
                 # Just continue SA from current_h which might have drifted
                 # It's better to reset to best_h to avoid drifting too far
                 current_h = best_h.copy()
                 current_cost = best_cost
                 # Reheat
                 T = 5.0

    return best_h

# Test to ensure it runs
if __name__ == "__main__":
    h = construct()
    print("Constructed h length:", len(h))
    print("First 10:", h[:10])
    # Calculate PSL for verification
    h_pad = np.zeros(2 * 400)
    h_pad[:400] = h
    H = np.fft.fft(h_pad)
    c = np.fft.ifft(H * np.conj(H)).real
    left = c[1:400]
    right = c[401:800]
    psl = np.max(np.abs(np.concatenate([left, right])))
    print("PSL:", psl)
