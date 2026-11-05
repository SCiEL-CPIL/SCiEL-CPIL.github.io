import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

def construct():
    """
    Constructs a signed step function with 400 intervals to minimize the
    third autocorrelation coefficient upper bound using simulated annealing
    with block flips and FFT-based autocorrelation for speed.
    """
    
    def compute_autocorr(s):
        """Compute full autocorrelation using FFT for speed."""
        ac = fftconvolve(s, s, mode='full')
        return ac
    
    def compute_c3_bound(s):
        """Compute the C3 upper bound: max |autocorrelation sidelobe|."""
        ac = compute_autocorr(s)
        sidelobes = ac[N:]  # lags 1 to N-1
        return np.max(np.abs(sidelobes))
    
    def compute_sidelobe_energy(s):
        """Compute sum of squared sidelobes as secondary objective."""
        ac = compute_autocorr(s)
        sidelobes = ac[N:]
        return np.sum(sidelobes**2)
    
    def combined_cost(s, weight=0.05):
        """Combined cost: primarily C3 bound, secondarily sidelobe energy."""
        c3 = compute_c3_bound(s)
        energy = compute_sidelobe_energy(s)
        return c3 + weight * np.sqrt(energy / N)
    
    def simulated_annealing(s, max_iter=60000, T_start=50.0, T_end=0.1, cooling=0.99995):
        """Simulated annealing with block flips to minimize the C3 bound."""
        current = s.copy()
        current_cost = combined_cost(current)
        best = current.copy()
        best_cost = current_cost
        T = T_start
        
        for i in range(max_iter):
            T = T_start * (cooling ** i)
            if T < T_end:
                T = T_end
            
            # Choose flip type: single bit or block
            if np.random.random() < 0.3:
                # Single bit flip
                idx = np.random.randint(N)
                new = current.copy()
                new[idx] = -new[idx]
            else:
                # Block flip
                block_len = np.random.randint(2, 10)
                idx = np.random.randint(N - block_len + 1)
                new = current.copy()
                new[idx:idx+block_len] = -new[idx:idx+block_len]
            
            new_cost = combined_cost(new)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = new
                current_cost = new_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
        
        return best, best_cost
    
    # Multiple random restarts with different initializations
    best_signs = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if restart == 0:
            # Chirp initialization (Barker-like)
            n = np.arange(N)
            phase = (n * (n - 1)) // 2
            signs = np.ones(N)
            signs[phase % 2 == 1] = -1.0
        elif restart == 1:
            # Pure random initialization
            signs = rng.choice([-1.0, 1.0], size=N)
        elif restart == 2:
            # Zadoff-Chu like: sign of chirp with specific parameter
            n = np.arange(N)
            freq = 1.0 / N
            phase = 2 * np.pi * freq * n * n
            signs = np.sign(np.cos(phase))
            signs[signs == 0] = 1.0
        elif restart == 3:
            # Low-frequency sinusoidal sign pattern
            t = np.linspace(0, 4 * np.pi, N)
            signs = np.sign(np.sin(t) + 0.3 * np.sin(2 * t))
            signs[signs == 0] = 1.0
        elif restart == 4:
            # M-sequence style using linear feedback
            signs = np.zeros(N)
            state = 1
            for i in range(N):
                signs[i] = 1.0 if (state & 1) == 1 else -1.0
                state = ((state << 1) ^ (state >> 7)) & 0xFF
        elif restart == 5:
            # Block pattern with varying block sizes
            signs = np.zeros(N)
            pos = 0
            block_sign = 1.0
            while pos < N:
                block_len = rng.integers(3, 15)
                signs[pos:pos+block_len] = block_sign
                pos += block_len
                block_sign = -block_sign
        elif restart == 6:
            # Random walk with occasional jumps
            signs = np.zeros(N)
            val = 1.0
            for i in range(N):
                signs[i] = val
                if rng.random() < 0.15:
                    val = -val
        elif restart == 7:
            # Fourier-based: sign of sum of a few random frequencies
            freqs = rng.uniform(0.01, 0.15, size=5)
            phases = rng.uniform(0, 2*np.pi, size=5)
            t = np.arange(N) / N
            signal = np.zeros(N)
            for f, p in zip(freqs, phases):
                signal += np.sin(2 * np.pi * f * t * N + p)
            signs = np.sign(signal)
            signs[signs == 0] = 1.0
        elif restart == 8:
            # Hybrid: chirp + random perturbation
            n = np.arange(N)
            phase = (n * (n - 1)) // 2
            signs = np.ones(N)
            signs[phase % 2 == 1] = -1.0
            flip_mask = rng.random(N) < 0.1
            signs[flip_mask] = -signs[flip_mask]
        else:
            # Another chirp variant with different parameter
            n = np.arange(N)
            rate = rng.uniform(0.5, 2.0)
            phase = rate * n * n
            signs = np.sign(np.cos(phase))
            signs[signs == 0] = 1.0
        
        # Run simulated annealing
        result, score = simulated_annealing(signs, max_iter=40000)
        
        if score < best_score:
            best_score = score
            best_signs = result
    
    # Final local search refinement on the best solution
    current = best_signs.copy()
    current_cost = combined_cost(current)
    
    # Stochastic local search: flip one bit or small block if it improves
    for _ in range(15000):
        if np.random.random() < 0.5:
            idx = np.random.randint(N)
            new = current.copy()
            new[idx] = -new[idx]
        else:
            block_len = np.random.randint(2, 5)
            idx = np.random.randint(N - block_len + 1)
            new = current.copy()
            new[idx:idx+block_len] = -new[idx:idx+block_len]
        
        new_cost = combined_cost(new)
        if new_cost < current_cost:
            current = new
            current_cost = new_cost
    
    return current

# EVOLVE-BLOCK-END
