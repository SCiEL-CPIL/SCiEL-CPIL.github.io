import time
import numpy as np
from scipy.optimize import minimize

N = 600
W = 1.0 / (N * 2)  # width of each interval
DEADLINE = 125.0   # safety margin before 150s kill

# Precompute basis for fast height construction
I = np.arange(N)

def _raw_score(heights):
    """Compute the raw value to be minimized."""
    dx = 1.0 / (N * 2)
    s = np.sum(heights)
    if s <= 0:
        return float('inf')
    conv = np.convolve(heights, heights, mode='full') * dx
    return np.max(conv) / (s * dx)**2

def construct():
    start_time = time.time()
    
    # Uniform baseline
    uniform = np.ones(N)
    uniform_score = _raw_score(uniform)
    
    best_heights = uniform.copy()
    best_score = uniform_score
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    dx = 1.0 / (N * 2)
    
    def soft_max(conv, tau=0.01):
        """Smooth approximation to max using log-sum-exp."""
        m = np.max(conv)
        return m + tau * np.log(np.sum(np.exp((conv - m) / tau)))
    
    def objective_smooth(h, tau=0.01):
        """Smooth objective: soft_max(conv) / (sum(h)*dx)^2"""
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s <= 0:
            return 1e10
        conv = np.convolve(h, h, mode='full') * dx
        sm = soft_max(conv, tau)
        return sm / (s * dx)**2
    
    # Strategy: Represent the step function as the squared magnitude of a phase-coded signal
    # h[i] = |A[i] * exp(i * phi[i])|^2 = A[i]^2
    # We optimize the phase profile phi[i] and envelope A[i] to minimize auto-correlation
    # This exploits the low auto-correlation properties of pseudo-random sequences
    
    # Parameterization: 
    # - Envelope: smooth low-frequency component (Fourier modes)
    # - Phase: piecewise constant with K jumps, optimized via simulated annealing
    
    K_ENV = 8  # number of envelope Fourier modes
    K_PHASE = 20  # number of phase jump points
    
    def env_fourier_to_heights(params_env):
        """Convert envelope Fourier parameters to nonnegative heights."""
        a0 = params_env[0]
        heights = np.full(N, a0)
        for k in range(1, K_ENV+1):
            a_k = params_env[1 + 2*(k-1)]
            b_k = params_env[2 + 2*(k-1)]
            angles = 2 * np.pi * k * I / N
            heights += a_k * np.cos(angles) + b_k * np.sin(angles)
        return np.maximum(heights, 0)
    
    def phase_to_signal(phase_jumps, jump_magnitudes):
        """Create a phase-coded signal from jump points and magnitudes."""
        phi = np.zeros(N)
        # Sort jump points
        order = np.argsort(phase_jumps)
        jumps_sorted = phase_jumps[order]
        mags_sorted = jump_magnitudes[order]
        
        # Build cumulative phase
        current_phase = 0
        for jump_pos, mag in zip(jumps_sorted, mags_sorted):
            idx = int(jump_pos * N)
            if 0 <= idx < N:
                current_phase += mag
                phi[idx:] = current_phase
        
        return phi
    
    def signal_to_heights(envelope, phi):
        """Convert envelope and phase to nonnegative heights."""
        # h[i] = envelope[i] * |exp(i*phi[i])|^2 = envelope[i] (since |exp(i*phi)|=1)
        # But we want to use the phase to create interference patterns
        # Actually, let's use: h[i] = |sum_k c_k * exp(i*(phi[i] + k*delta))|^2
        # For simplicity, use: h[i] = envelope[i] * (1 + 0.5*cos(phi[i]))
        # This ensures non-negativity and allows phase to modulate the height
        return np.maximum(envelope * (1.0 + 0.5 * np.cos(phi)), 0)
    
    # Simulated annealing for phase optimization
    def simulated_annealing_phase(envelope, T0=1.0, T_end=0.01, n_iter=500):
        """Optimize phase jumps via simulated annealing."""
        # Initialize with random phase jumps
        n_jumps = 10
        phase_jumps = np.random.uniform(0, 1, n_jumps)
        jump_magnitudes = np.random.uniform(-np.pi, np.pi, n_jumps)
        
        phi = phase_to_signal(phase_jumps, jump_magnitudes)
        heights = signal_to_heights(envelope, phi)
        current_score = objective_smooth(heights, tau=0.01)
        
        best_heights = heights.copy()
        best_score = current_score
        
        for it in range(n_iter):
            if not check_time():
                break
            
            # Temperature schedule
            T = T0 * (T_end / T0) ** (it / n_iter)
            
            # Random perturbation: move a jump or change its magnitude
            if np.random.rand() < 0.5:
                # Move a jump
                idx = np.random.randint(n_jumps)
                phase_jumps[idx] = np.clip(phase_jumps[idx] + np.random.normal(0, 0.1), 0, 1)
            else:
                # Change magnitude
                idx = np.random.randint(n_jumps)
                jump_magnitudes[idx] += np.random.normal(0, 0.5)
            
            phi_new = phase_to_signal(phase_jumps, jump_magnitudes)
            heights_new = signal_to_heights(envelope, phi_new)
            new_score = objective_smooth(heights_new, tau=0.01)
            
            # Accept if better or with probability exp(-(new-old)/T)
            if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / max(T, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_heights = heights_new.copy()
            else:
                # Revert
                pass
        
        return best_heights, best_score
    
    # Optimize envelope using L-BFGS-B
    def objective_envelope(params_env, phi):
        """Objective for envelope optimization given fixed phase."""
        envelope = env_fourier_to_heights(params_env)
        heights = signal_to_heights(envelope, phi)
        return objective_smooth(heights, tau=0.01)
    
    # Main optimization loop
    np.random.seed(42)
    
    num_rounds = 0
    max_rounds = 15
    
    while check_time() and num_rounds < max_rounds:
        num_rounds += 1
        
        if num_rounds == 1:
            # Start with uniform envelope
            params_env = np.zeros(1 + 2*K_ENV)
            params_env[0] = 1.0
        elif num_rounds <= 5:
            # Small random perturbation
            params_env = np.zeros(1 + 2*K_ENV)
            params_env[0] = 1.0
            params_env[1:] = 0.1 * np.random.randn(2*K_ENV)
        elif num_rounds <= 10:
            # Moderate random
            params_env = np.zeros(1 + 2*K_ENV)
            params_env[0] = 0.5 + 0.5 * np.random.rand()
            params_env[1:] = 0.3 * np.random.randn(2*K_ENV)
        else:
            # Larger random
            params_env = np.zeros(1 + 2*K_ENV)
            params_env[0] = np.random.uniform(0.2, 2.0)
            params_env[1:] = 0.5 * np.random.randn(2*K_ENV)
        
        # Optimize envelope with fixed zero phase first
        try:
            result = minimize(
                lambda p: objective_envelope(p, np.zeros(N)),
                params_env,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if check_time():
                envelope = env_fourier_to_heights(result.x)
                
                # Now optimize phase via simulated annealing
                heights_sa, score_sa = simulated_annealing_phase(envelope)
                
                if score_sa < best_score:
                    best_score = score_sa
                    best_heights = heights_sa.copy()
                    
                    # Update envelope params from best heights
                    # This is approximate; we'll just use the current params_env
        except Exception:
            pass
    
    # Final polish: coordinate descent on the actual heights
    if check_time():
        h = best_heights.copy()
        s = np.sum(h)
        if s > 0:
            h = h / s * N  # normalize to sum = N
            
            # Coordinate descent with smooth objective
            tau = 0.01
            improved = True
            while improved and check_time():
                improved = False
                for i in range(0, N, 5):  # step by 5 for speed
                    if not check_time():
                        break
                    best_val = objective_smooth(h, tau)
                    best_h = h[i]
                    
                    # Try a few perturbations
                    for delta in [-0.3, -0.1, 0.1, 0.3]:
                        h[i] = max(0, best_h + delta)
                        val = objective_smooth(h, tau)
                        if val < best_val:
                            best_val = val
                            best_h = h[i]
                    
                    h[i] = best_h
                    if best_val < objective_smooth(h, tau) - 1e-12:
                        pass
                    # Check if we improved
                    if best_val < _raw_score(h) * 0.999:
                        improved = True
            
            s_final = _raw_score(h)
            if s_final < best_score:
                best_score = s_final
                best_heights = h.copy()
    
    # Final check
    final_score = _raw_score(best_heights)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_heights)
