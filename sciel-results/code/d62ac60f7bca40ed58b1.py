"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 600
DEADLINE = 120.0  # seconds
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # We want to find a nonnegative step function h with N=600 equal intervals on [-1/4, 1/4]
    # that minimizes the normalized maximum of its autoconvolution.
    # 
    # The autoconvolution of h is (h * h)(t) = integral h(s) h(t-s) ds.
    # The "normalized maximum" is max_t (h*h)(t) / (integral h)^2 perhaps, or similar.
    # Let's think about what "normalized maximum of its autoconvolution" means.
    # 
    # For a step function with heights h_i on intervals of width dx, the autoconvolution at a point
    # can be computed. The normalization is likely by the square of the integral (L1 norm squared).
    # 
    # Strategy: Represent h as |f|^2 where f is a complex signal. Then the autoconvolution of h
    # is related to the autoconvolution of |f|^2. But actually, the autoconvolution of |f|^2 is
    # not simply related to f. 
    # 
    # Alternative: The problem is equivalent to minimizing the peak of the autoconvolution of a
    # nonnegative function with fixed L1 norm (or fixed L2 norm?). Let's assume fixed L1 norm = 1.
    # Then we want to minimize max_t (h*h)(t).
    # 
    # For a uniform function h=1 on [-1/4,1/4], the autoconvolution is a triangle peaking at 0.
    # The peak value is 1/4 (since the integral of h is 1/2, and the triangle height is 1/2 * 1/2 = 1/4? 
    # Let me compute: if h=1 on [-a,a] with a=1/4, then (h*h)(t) for |t|<=2a is 2a-|t|. Peak at t=0 is 2a=1/2.
    # The L1 norm is 2a=1/2. Normalized by (L1)^2 = 1/4. So normalized peak = (1/2)/(1/4) = 2.
    # 
    # Wait, let me reconsider. If the function is on [-1/4, 1/4] with 600 equal intervals, each interval has width dx = (1/2)/600 = 1/1200.
    # If all heights are 1, the L1 norm is 600 * 1 * (1/1200) = 1/2.
    # The autoconvolution of a box function of width 1/2 has a triangular autoconvolution with peak 1/2.
    # Normalized by (L1)^2 = (1/2)^2 = 1/4, the normalized peak is (1/2)/(1/4) = 2.
    # 
    # So the seed score of 1.0 corresponds to a normalized maximum of 2.0? Or maybe the score is the value itself.
    # The problem says "score is the upper bound on C1" and "the evaluator measures the raw value (lower is better)".
    # So we want to minimize the normalized maximum of the autoconvolution.
    # 
    # Let's use the approach: represent h_i = |c_i|^2 where c_i are complex numbers with fixed sum of |c_i|^2 (L2 norm) or fixed sum of h_i (L1 norm).
    # Actually, let's fix the L1 norm to 1 for simplicity, so sum(h_i) * dx = 1, i.e., sum(h_i) = 1200.
    # 
    # The autoconvolution of h at discrete level: if h is a step function with heights h_i on intervals [x_i, x_{i+1}],
    # then (h*h)(t) can be computed. For the peak, it's likely near t=0.
    # 
    # Let's compute the autoconvolution peak numerically for a given h.
    # 
    # For a step function, the autoconvolution at t=0 is sum_i h_i^2 * dx (approximately, if we consider the self-convolution at the center).
    # Actually, (h*h)(0) = integral h(s)h(-s) ds. If h is even, this is 2*integral_0^{1/4} h(s)^2 ds.
    # 
    # Let's just compute the autoconvolution numerically using FFT or direct computation.
    
    def autoconvolution_peak(h, dx):
        """Compute the normalized peak of the autoconvolution of step function h."""
        # h is array of N heights
        # L1 norm
        l1 = np.sum(h) * dx
        if l1 == 0:
            return np.inf
        # Compute autoconvolution using FFT
        # The autoconvolution of a discrete signal can be computed via FFT
        # But we need to be careful about the continuous nature.
        # For a step function, the autoconvolution is a piecewise linear function.
        # Let's compute it directly for the peak (which should be at t=0 if h is nonnegative).
        
        # For nonnegative h, the peak of the autoconvolution is at t=0.
        # (h*h)(0) = integral_{-1/4}^{1/4} h(s) h(-s) ds
        # Since h is nonnegative, and we're on [-1/4, 1/4], h(-s) is defined.
        # For a step function with N intervals, let's compute this integral.
        
        # The intervals are [x_i, x_{i+1}] where x_i = -1/4 + i*dx
        # h(s) = h_i for s in [x_i, x_{i+1}]
        # h(-s) = h_j where -s is in [x_j, x_{j+1}]
        
        # This is complex to compute directly. Let's use a numerical approach.
        # Sample the autoconvolution at many points and take the max.
        
        # Actually, for the peak at t=0:
        # (h*h)(0) = sum over pairs of intervals where s and -s overlap.
        
        # Let's use a simpler numerical method: compute h on a fine grid and use np.convolve or FFT.
        
        # Create a fine grid
        n_fine = N * 10  # 6000 points
        x_fine = np.linspace(-0.25, 0.25, n_fine)
        dx_fine = x_fine[1] - x_fine[0]
        
        # Interpolate h to the fine grid
        x_coarse = np.linspace(-0.25, 0.25, N + 1)
        h_coarse = np.concatenate([h, [h[-1]]])
        h_fine = np.interp(x_fine, x_coarse[:-1] + dx/2, h)  # center of intervals
        
        # Compute autoconvolution using FFT
        # Pad to avoid circular convolution
        n_fft = 2 * n_fine
        h_fft = np.fft.rfft(h_fine, n=n_fft)
        ac_fft = h_fft * np.conj(h_fft)
        ac_fine = np.fft.irfft(ac_fft, n=n_fft)[:n_fine]
        
        # The autoconvolution values are scaled by dx_fine
        ac_fine *= dx_fine
        
        peak = np.max(ac_fine)
        normalized_peak = peak / (l1 ** 2)
        return normalized_peak
    
    def objective(params):
        """Objective function: minimize the normalized peak of the autoconvolution."""
        # params are phases for N/2 complex coefficients (real and imaginary parts)
        # Actually, let's use N real and imaginary parts for N complex coefficients.
        # h_i = |c_i|^2 where c_i = a_i + i*b_i
        # We need to normalize so that sum(h_i) = 1200 (L1 norm = 1)
        
        a = params[:N]
        b = params[N:2*N]
        c = a + 1j * b
        h = np.abs(c) ** 2
        
        # Normalize to L1 norm = 1
        s = np.sum(h)
        if s == 0:
            return np.inf
        h = h * (1200.0 / s)  # so that sum(h) * dx = 1
        
        return autoconvolution_peak(h, 1.0/1200.0)
    
    # Initial guess: uniform function
    # For uniform h, all c_i should have the same magnitude.
    # Let's start with c_i = 1 for all i, so h_i = 1.
    x0 = np.zeros(2 * N)
    x0[:N] = 1.0  # a_i = 1
    x0[N:] = 0.0  # b_i = 0
    
    # Use differential evolution
    # This might be slow for 1200 dimensions. Let's try a smaller parameterization.
    
    # Alternative: Use a smaller number of parameters. For example, use a low-rank representation.
    # h_i = |sum_{k=1}^K d_k * exp(i * omega_k * i)|^2
    # This is a sum of K complex exponentials, squared magnitude.
    
    K = 10  # number of complex exponentials
    # Parameters: for each k, we have amplitude, frequency, phase
    # Total parameters: 3*K
    
    def objective_small(params):
        """Objective with K complex exponentials."""
        amplitudes = params[:K]
        frequencies = params[K:2*K]
        phases = params[2*K:3*K]
        
        # Compute h_i = |sum_k amp_k * exp(i*(freq_k * i + phase_k))|^2
        i = np.arange(N)
        c = np.zeros(N, dtype=complex)
        for k in range(K):
            c += amplitudes[k] * np.exp(1j * (frequencies[k] * i + phases[k]))
        h = np.abs(c) ** 2
        
        # Normalize to L1 norm = 1
        s = np.sum(h)
        if s == 0:
            return np.inf
        h = h * (1200.0 / s)
        
        return autoconvolution_peak(h, 1.0/1200.0)
    
    # Bounds for differential evolution
    bounds = [(-5, 5)] * K + [(-10, 10)] * K + [(-np.pi, np.pi)] * K
    
    # Initial population
    popsize = 20
    
    # Run differential evolution with time limit
    result = differential_evolution(
        objective_small,
        bounds,
        maxiter=50,
        popsize=popsize,
        tol=1e-6,
        seed=42,
        init='sobol',
        polish=True
    )
    
    # Check time
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # Return the best found so far
        best_params = result.x
        amplitudes = best_params[:K]
        frequencies = best_params[K:2*K]
        phases = best_params[2*K:3*K]
        
        i = np.arange(N)
        c = np.zeros(N, dtype=complex)
        for k in range(K):
            c += amplitudes[k] * np.exp(1j * (frequencies[k] * i + phases[k]))
        h = np.abs(c) ** 2
        s = np.sum(h)
        if s > 0:
            h = h * (1200.0 / s)
        return h.tolist()
    
    # If we have time, try to improve with more iterations or a different method
    # For now, return the result
    best_params = result.x
    amplitudes = best_params[:K]
    frequencies = best_params[K:2*K]
    phases = best_params[2*K:3*K]
    
    i = np.arange(N)
    c = np.zeros(N, dtype=complex)
    for k in range(K):
        c += amplitudes[k] * np.exp(1j * (frequencies[k] * i + phases[k]))
    h = np.abs(c) ** 2
    s = np.sum(h)
    if s > 0:
        h = h * (1200.0 / s)
    return h.tolist()
# EVOLVE-BLOCK-END
