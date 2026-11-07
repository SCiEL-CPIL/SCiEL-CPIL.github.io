"""Improved program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(50):
        s = h.sum()
        if abs(s - target) < 1e-13:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    # Final adjustment for precision
    s = h.sum()
    if abs(s - target) > 1e-12:
        diff = target - s
        free = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy: Explore multiple frequency components and optimize each.
    # The cosine wave h = 0.5 + A*cos(2*pi*k*i/N) naturally satisfies sum=0.5*N
    # and stays in [0,1] if A<=0.5.
    # We will try frequencies 1 through 8, and also try to optimize the amplitude A
    # and phase, though phase is less critical for the max correlation value
    # due to rotational symmetry of the objective (circular correlation max is invariant to shift).
    # Actually, the objective is max of full correlation, which includes all shifts.
    # So phase doesn't matter. Amplitude does.
    
    # For a pure cosine h_i = 0.5 + A*cos(2*pi*k*i/N):
    # The correlation with 1-h = 0.5 - A*cos(2*pi*k*i/N)
    # c[m] = sum h[i] (1-h)[i+m]
    # This is a convolution.
    # Let's just use the optimizer to find the best A and potentially mix frequencies.
    
    # First, try single frequency optimization with L-BFGS-B
    # We parameterize h by its Fourier coefficients? No, just optimize h directly
    # but start from good initializations.
    
    # Initialization 1: Pure cosines with varying amplitude
    # We know A=0.5 is the max amplitude. Maybe smaller A is better?
    # Let's try optimizing A for each frequency k.
    
    for k in range(1, 9):
        if remaining() < 5:
            break
        
        # Try different initial amplitudes
        for A_init in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if remaining() < 2:
                break
            
            h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N)
            h_init = project(h_init)
            
            # Optimize using L-BFGS-B
            # We use a numerical gradient to avoid implementing the analytical gradient of the max-correlation,
            # which is tricky due to the max operation.
            
            def obj_func(x):
                return objective(project(x))
            
            # Use finite difference gradient for robustness
            # Or use 'BFGS' with numerical gradient if L-BFGS-B struggles?
            # L-BFGS-B with bounds is good.
            
            try:
                result = minimize(
                    obj_func,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 2: Mix frequencies
    # Start from the best single frequency found, and try to perturb it with other frequencies
    if remaining() > 10:
        # Try adding small amounts of other frequencies to the best solution
        h_curr = best_h.copy()
        
        for k in range(1, 9):
            if remaining() < 5:
                break
            
            # Skip if k is already dominant? Hard to tell. Just try.
            # Add a small cosine component
            delta = 0.05 * np.cos(2 * np.pi * k * i / N)
            h_mix = h_curr + delta
            h_mix = project(h_mix)
            
            # Optimize this mixture
            def obj_func(x):
                return objective(project(x))
            
            try:
                result = minimize(
                    obj_func,
                    h_mix,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 3: Local search / random restarts from best
    if remaining() > 5:
        np.random.seed(123)
        for _ in range(10):
            if remaining() < 2:
                break
            
            # Small random perturbation
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_func(x):
                return objective(project(x))
            
            try:
                result = minimize(
                    obj_func,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass

    # Final check and projection
    best_h = project(best_h)
    return best_h
