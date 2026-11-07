"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Spectral phase optimization approach
    # We represent h = |z|^2 where z = IFFT(Z)
    # We optimize the phase of Z to maximize the score
    
    M = 128  # FFT size
    # Target grid for h
    t_h = np.linspace(-0.25, 0.25, N)
    # Grid for z from FFT
    t_z = np.linspace(-0.5, 0.5, M, endpoint=False)
    # Note: np.fft.ifft returns values at 0, 1/M, ...
    # We can shift them to be centered
    # Or just use the raw output and interpolate
    
    # Initial guess: flat spectrum (random phase)
    # Random phase helps escape local minima
    np.random.seed(42)
    initial_phases = np.random.uniform(0, 2 * np.pi, M)
    
    def objective(phases):
        Z = np.exp(1j * phases)
        # Add a small amplitude window to ensure h is well-behaved?
        # Or just unit amplitude.
        z = np.fft.ifft(Z)
        # z is complex, length M
        # We need h on t_h
        # The time points of z from ifft are 0, 1/M, ..., (M-1)/M
        # Let's map these to a grid centered at 0 for interpolation
        # t_z_raw = np.arange(M) / M
        # To center: shift by M/2
        # Actually, np.fft.ifft output corresponds to t = 0..M-1 steps.
        # If we use fftshift, we get t = -M/2 .. M/2 - 1 steps.
        z_shifted = np.fft.fftshift(z)
        t_z_shifted = np.linspace(-0.5, 0.5, M, endpoint=False)
        
        # Interpolate |z|^2 onto t_h
        h = np.interp(t_h, t_z_shifted, np.abs(z_shifted)**2)
        
        # Normalize h to unit energy to make scale invariant
        norm_h = np.linalg.norm(h)
        if norm_h == 0:
            return 0.0
        h = h / norm_h
        
        # Compute g
        g = np.convolve(h, h)
        m = len(g)
        
        # Construct y
        y = np.concatenate(([0.0], g, [0.0]))
        w = 1.0 / (m + 1)
        
        # Compute L2 term
        # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
        # y has length m+2. Indices 0..m+1.
        # The sum is over i=0..m.
        # y[i] and y[i+1] are defined for i=0..m.
        terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
        l2 = np.sum(terms) * (w / 3.0)
        
        # Compute norms
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        norm_inf = np.max(np.abs(g))
        
        if norm_1 * norm_inf == 0:
            return 0.0
            
        value = l2 / (norm_1 * norm_inf)
        return -value # Minimize negative value

    # Run optimization
    # Powell is robust for non-smooth functions
    result = minimize(objective, initial_phases, method='Powell', options={'maxiter': 500, 'ftol': 1e-6})
    
    # Get the best phases
    best_phases = result.x
    Z = np.exp(1j * best_phases)
    z = np.fft.ifft(Z)
    z_shifted = np.fft.fftshift(z)
    t_z_shifted = np.linspace(-0.5, 0.5, M, endpoint=False)
    
    h = np.interp(t_h, t_z_shifted, np.abs(z_shifted)**2)
    
    # Ensure non-negative
    h = np.maximum(h, 0.0)
    
    # Scale h to a reasonable magnitude? 
    # The score is scale invariant, so any scale works.
    # But we should ensure it's not zero.
    if np.sum(h) == 0:
        return [1.0] * N
        
    return list(h)
# EVOLVE-BLOCK-END
