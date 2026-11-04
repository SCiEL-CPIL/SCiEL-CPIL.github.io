import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def get_psl(h):
        """
        Compute Peak Sidelobe Level (PSL) of the autocorrelation.
        h is an array of +/- 1.
        """
        # Zero-pad to 2N to compute linear autocorrelation via circular autocorrelation
        h_pad = np.zeros(2 * N)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        c = np.fft.ifft(H * np.conj(H)).real
        
        # Extract sidelobes
        # Lag 0 is c[0] (main lobe)
        # Lags 1 to N-1 are c[1:N]
        # Lags -1 to -(N-1) are c[2N-1] down to c[N+1], i.e., c[N+1:2N] reversed?
        # Let's verify indices for ifft of |fft|^2.
        # ifft(fft(x)*conj(fft(x))) gives circular correlation r[k] = sum x[n] conj(x[n-k])
        # For real x, r[k] = sum x[n] x[n-k].
        # r[0] is at index 0.
        # r[1] is at index 1.
        # ...
        # r[N-1] is at index N-1.
        # r[-1] is at index 2N-1.
        # r[-(N-1)] is at index N+1.
        # So positive lags: 1..N-1 -> indices 1..N-1
        # Negative lags: -1..-(N-1) -> indices 2N-1..N+1
        # We take the max absolute value of all these.
        
        left = c[1:N]
        right = c[N+1:2*N]
        
        if len(left) == 0 and len(right) == 0:
            return 0.0
            
        return np.max(np.abs(np.concatenate([left, right])))

    def generate_chirp(f1, f2):
        t = np.arange(N)
        phase = 2 * np.pi * f1 * t + np.pi * f2 * t**2
        sig = np.cos(phase)
        h = np.sign(sig)
        h[h == 0] = 1.0
        return h

    rng = np.random.RandomState(42)

    # 1. Initialization
    # Try diverse chirps and random starts
    best_h = None
    best_cost = float('inf')
    
    for _ in range(100):
        f1 = rng.rand()
        f2 = rng.rand()
        h = generate_chirp(f1, f2)
        cost = get_psl(h)
        if cost < best_cost:
            best_cost = cost
            best_h = h.copy()

    for _ in range(100):
        h = rng.choice([-1, 1], size=N).astype(np.float64)
        cost = get_psl(h)
        if cost < best_cost:
            best_cost = cost
            best_h = h.copy()

    # 2. Block-Flip Local Search
    # Strategy: Instead of flipping single bits, flip contiguous blocks of bits.
    # This allows the search to move between distant local minima more effectively
    # than single-bit flips, which get stuck in "rugged" landscapes where no single
    # bit flip improves the PSL.
    
    current_h = best_h.copy()
    current_cost = best_cost
    
    # Block sizes to try
    block_sizes = [1, 2, 3, 5, 8, 10, 15, 20]
    
    # We will cycle through block sizes or pick randomly.
    # To be efficient, we prioritize smaller blocks for fine-tuning but occasionally use larger ones.
    
    while time.time() < deadline:
        # Randomly select a block size, biased towards smaller ones
        # but ensure we explore larger structures
        # Using a simple distribution:
        # 1: 40%, 2: 20%, 3: 10%, 5: 10%, 8: 5%, 10: 5%, 15: 5%, 20: 5%
        r = rng.rand()
        if r < 0.4:
            b_size = 1
        elif r < 0.6:
            b_size = 2
        elif r < 0.7:
            b_size = 3
        elif r < 0.8:
            b_size = 5
        elif r < 0.85:
            b_size = 8
        elif r < 0.9:
            b_size = 10
        elif r < 0.95:
            b_size = 15
        else:
            b_size = 20

        # Pick a random start index for the block
        # Ensure the block fits within the array
        if b_size >= N:
            start_idx = 0
            b_size = N
        else:
            start_idx = rng.randint(0, N - b_size + 1)
            
        end_idx = start_idx + b_size
        
        # Flip the block
        current_h[start_idx:end_idx] *= -1.0
        
        # Evaluate new cost
        new_cost = get_psl(current_h)
        
        # Accept if better, or with a small probability if worse (Simulated Annealing style)
        # However, pure greedy often works well for PSL if we have diverse starts.
        # Let's use a mild annealing schedule to escape local minima.
        # Temperature T starts high, decays.
        
        # For simplicity and robustness, let's use a "First Improvement" or "Best Improvement" strategy
        # within a batch of flips, or just simple acceptance with a cooling schedule.
        
        # Let's implement a simple cooling schedule per epoch.
        # Actually, let's just use a fixed small probability of accepting worse moves to escape traps.
        # P_accept = exp(-delta / T). Let T be dynamic.
        
        # To keep it simple and fast, let's try:
        # If new_cost < current_cost: Accept
        # Else: Accept with probability 0.05 (random walk to escape)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = current_h.copy()
        else:
            # Reject and revert, unless we are stuck and want to explore
            # Revert
            current_h[start_idx:end_idx] *= -1.0
            # Occasionally, force a random perturbation to escape local minima
            if rng.rand() < 0.01:
                # Perturb: flip a random larger block
                pert_size = rng.randint(10, 50)
                if pert_size < N:
                    p_start = rng.randint(0, N - pert_size + 1)
                    p_end = p_start + pert_size
                    current_h[p_start:p_end] *= -1.0
                    pert_cost = get_psl(current_h)
                    # Don't track pert_cost as current_cost strictly, just update if it's good
                    if pert_cost < current_cost:
                        current_cost = pert_cost
                        if pert_cost < best_cost:
                            best_cost = pert_cost
                            best_h = current_h.copy()
                    else:
                        # Revert perturbation
                        current_h[p_start:p_end] *= -1.0
            # If we rejected the main flip, we don't change current_h (it was reverted)

    return best_h

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
