"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr(x):
    """Compute autocorrelation of x using FFT."""
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return ac

def _peak_sidelobe(x):
    """Compute peak sidelobe of autocorrelation."""
    ac = _autocorr(x)
    return np.max(np.abs(ac[1:]))

def _l2_sidelobe(x):
    """Compute L2 norm of autocorrelation sidelobes."""
    ac = _autocorr(x)
    return np.sqrt(np.sum(ac[1:]**2))

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    # Phase 1: Continuous optimization to find a good initial point
    np.random.seed(42)
    x_cont = np.random.randn(N)
    
    # Gradient descent on L2 norm of sidelobes
    lr = 0.01
    prev_loss = _l2_sidelobe(x_cont)
    
    for step in range(5000):
        if time.time() > deadline - 30:
            break
        
        # Compute gradient of L2 sidelobe norm
        ac = _autocorr(x_cont)
        # Gradient of sum(ac[k]^2) for k=1..N-1 w.r.t. x
        # d/dx sum_{k=1}^{N-1} ac[k]^2 = 2 * sum_{k=1}^{N-1} ac[k] * d(ac[k])/dx
        # d(ac[k])/dx_i = x_{i+k} + x_{i-k} (with appropriate boundaries)
        # This is essentially the convolution of ac[1:] with a specific kernel
        
        # Simpler: use numerical gradient via finite differences or analytic
        # For speed, use the fact that grad = 2 * (ac_sidelobes convolved with x-like structure)
        # Actually, let's just do a simple gradient step using FFT-based computation
        
        # The gradient of sum_{k=1}^{N-1} |ac[k]|^2 w.r.t. x is:
        # 2 * real(ifft(fft(x) * conj(fft(x)) * mask)) where mask zeroes out lag 0
        # More precisely: grad_i = 2 * sum_{k=1}^{N-1} ac[k] * (x[i+k] + x[i-k])
        
        # Efficient computation:
        f = np.fft.fft(x_cont, n=2*N)
        power = f * np.conj(f)
        # Zero out the DC and keep only sidelobe contributions
        # The gradient in frequency domain
        grad = np.fft.ifft(power * np.conj(f)).real[:N]  # This isn't quite right
        
        # Let me use a different approach: analytic gradient
        # ac[k] = sum_i x[i] * x[i+k]
        # d(ac[k])/dx[j] = x[j+k] + x[j-k] (with wrap or zero-padding)
        # d(sum_{k=1}^{N-1} ac[k]^2)/dx[j] = 2 * sum_{k=1}^{N-1} ac[k] * (x[j+k] + x[j-k])
        
        # For zero-padded autocorrelation:
        # d(ac[k])/dx[j] = x[j+k] if j+k < N else 0, plus x[j-k] if j-k >= 0 else 0
        # This is getting complex. Let's use a simpler approach.
        
        # Use the identity: grad = 2 * (ac[1:] * x) convolved appropriately
        # Actually, let's just use a small step and evaluate
        
        # Simpler gradient: use the fact that for the L2 norm of sidelobes,
        # the gradient can be computed as:
        # grad = 2 * (np.convolve(ac[1:], x, 'full')[:N] + np.convolve(ac[1:], x[::-1], 'full')[N-1:2*N-1])
        # This is approximate. Let me just use a random perturbation approach for speed.
        
        # Actually, let's use a proper gradient via FFT
        # The autocorrelation is ac = ifft(|fft(x)|^2)
        # d(ac[k])/dx = ifft(fft(x) * conj(fft(x)) * something)
        
        # Let me just do a simple coordinate descent or random search for the continuous phase
        # For speed, let's do a few iterations of a simple gradient step
        
        # Gradient of L2 sidelobe: 
        # We want to minimize sum_{k=1}^{N-1} ac[k]^2
        # grad_x = 2 * ifft(fft(ac_sidelobes_padded) * conj(fft(x_padded)))... 
        # This is getting too complex. Let me use a different strategy.
        
        # Strategy: Use simulated annealing / random flips for binary directly,
        # but initialize with a good pattern.
        
        break  # Skip continuous optimization for now, go to binary search
    
    # Phase 2: Binary local search
    # Start with a random binary sequence
    x_bin = np.random.choice([-1, 1], size=N)
    
    best_x = x_bin.copy()
    best_score = _peak_sidelobe(best_x)
    
    # Local search: try flipping each bit
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            # Try flipping bit i
            x_test = best_x.copy()
            x_test[i] = -x_test[i]
            score = _peak_sidelobe(x_test)
            
            if score < best_score:
                best_score = score
                best_x = x_test
                improved = True
    
    # Phase 3: Multi-start local search if time remains
    while time.time() < deadline - 10:
        # Start from a perturbed version of best
        x_start = best_x.copy()
        # Flip a few random bits
        n_flips = np.random.randint(5, 20)
        flip_idx = np.random.choice(N, size=n_flips, replace=False)
        x_start[flip_idx] = -x_start[flip_idx]
        
        # Local search from this start
        current_x = x_start
        current_score = _peak_sidelobe(current_x)
        
        local_improved = True
        while local_improved and time.time() < deadline - 10:
            local_improved = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                x_test = current_x.copy()
                x_test[i] = -x_test[i]
                score = _peak_sidelobe(x_test)
                if score < current_score:
                    current_score = score
                    current_x = x_test
                    local_improved = True
        
        if current_score < best_score:
            best_score = current_score
            best_x = current_x
    
    return best_x
# EVOLVE-BLOCK-END
