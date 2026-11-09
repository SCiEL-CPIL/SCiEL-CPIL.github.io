import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3 (interpreted as peak autocorrelation sidelobe).
    
    Uses a structural change: parameterizes the discrete signs as the sign
    of a continuous sinusoidal waveform, then uses gradient-based optimization
    (L-BFGS) to find optimal frequency and phase, followed by local refinement.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin (150s limit)

    def compute_peak_sidelobe(h):
        """
        Computes the maximum absolute value of the autocorrelation
        at non-zero lags.
        """
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H) ** 2).real
        peak = np.max(np.abs(R[1:]))
        return peak

    def smooth_objective(params):
        """
        Smooth objective function for gradient-based optimization.
        Parameters: [frequency, phase]
        Returns: smoothed approximation of peak sidelobe.
        """
        freq, phase = params
        # Generate continuous sinusoidal signal
        t = np.linspace(0, 1, N, endpoint=False)
        signal = np.sin(2 * np.pi * freq * t + phase)
        
        # Smooth approximation of sign function using tanh
        h_continuous = np.tanh(10 * signal)
        
        # Compute autocorrelation using FFT
        H = np.fft.fft(h_continuous)
        R = np.fft.ifft(np.abs(H) ** 2).real
        
        # Use smooth max approximation instead of hard max
        # This makes the objective differentiable
        sidelobes = np.abs(R[1:N])
        # Smooth max using log-sum-exp
        beta = 10.0  # Sharpness parameter
        smooth_max = np.log(np.sum(np.exp(beta * sidelobes))) / beta
        
        return smooth_max

    best_h = None
    best_score = float('inf')

    # Phase 1: Gradient-based optimization over continuous parameters
    # Try multiple initial frequencies and phases
    np.random.seed(42)
    
    # Sample initial conditions
    num_initial = 20
    for i in range(num_initial):
        if time.time() > deadline - 20:
            break
        
        # Random initial frequency and phase
        freq0 = np.random.uniform(0.1, 5.0)
        phase0 = np.random.uniform(0, 2 * np.pi)
        
        try:
            result = minimize(
                smooth_objective,
                x0=[freq0, phase0],
                method='L-BFGS-B',
                bounds=[(0.01, 10.0), (0, 2 * np.pi)],
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if result.success:
                freq_opt, phase_opt = result.x
                t = np.linspace(0, 1, N, endpoint=False)
                signal = np.sin(2 * np.pi * freq_opt * t + phase_opt)
                
                # Convert to discrete signs
                h_discrete = np.sign(signal)
                # Handle zeros by assigning +1
                h_discrete[h_discrete == 0] = 1
                
                score = compute_peak_sidelobe(h_discrete)
                if score < best_score:
                    best_score = score
                    best_h = h_discrete.copy()
                    
        except Exception:
            continue

    # Phase 2: Local refinement using greedy single-flip moves
    if best_h is not None:
        h = best_h.copy()
        current_score = compute_peak_sidelobe(h)
        
        # Multiple passes of greedy improvement
        max_passes = 10
        for _ in range(max_passes):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                h[i] *= -1
                new_score = compute_peak_sidelobe(h)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    h[i] *= -1
            
            if not improved:
                break
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    # Fallback: if no good solution found, use random sequence
    if best_h is None:
        best_h = np.random.choice([-1, 1], N)

    return best_h
