import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Use the 10 parameters as the x-coordinates of the spline knots
    # We map the parameters to the domain [-2, 2] to ensure they cover the input range
    # We use a tanh-like mapping or linear scaling. 
    # To ensure robustness and distinct knots, we sort them and add a small jitter if they collapse.
    # However, BFGS starts at 1.0, so all knots would be at 1.0 initially.
    # We need a mapping that spreads them out or handles duplicates.
    # Let's map params to [-1.9, 1.9] using a linear scaling from the param space.
    # But if params are all 1.0, they map to the same point.
    # A better strategy for "adaptive knots":
    # Use the params to define a set of centers, but ensure they are distinct by adding a fixed offset pattern?
    # No, that breaks the differentiability if we hard-code offsets.
    # Instead, let's use the parameters to define the *shape* of a spline via a different mechanism.
    
    # Alternative Structural Change:
    # Use a "Sum of Sine" or "Harmonic" basis? 
    # The current program uses RBF + Chebyshev. 
    # Let's try a pure "Neural Network" style activation stack? No, too complex for 10 params.
    
    # Let's stick to the "Adaptive Knot Spline" idea but make it robust to initial collapse.
    # If all knots are at the same location, the spline is ill-defined.
    # We can enforce distinctness by mapping the sorted parameters to a fixed grid?
    # If we sort p, the function is non-differentiable at ties, but BFGS uses numerical gradients.
    # However, if p_i = p_j, the gradient might be zero or undefined.
    
    # Let's try a different structural change: 
    # **Global Harmonic Basis with Learned Frequencies and Phases**
    # y(x) = sum_{i=1}^{10} A_i * sin(f_i * x + phi_i)
    # This is a sum of 10 sine waves.
    # Params: p[0..4] are Amplitudes, p[5..9] are Frequencies? 
    # We need phases too. We only have 10 params.
    # Let's use 5 sine terms: y(x) = sum_{i=0}^{4} p[2i] * sin(p[2i+1] * x)
    # This uses 10 params: 5 amplitudes, 5 frequencies.
    # This is a non-linear model in params (because of the frequency inside the sin).
    # BFGS can handle this.
    # This family can represent a wide variety of smooth periodic and quasi-periodic functions.
    # It is structurally different from the linear RBF/Chebyshev mixture.
    
    A = p[0:5]
    F = p[5:10]
    
    # To avoid issues with very large frequencies causing numerical instability or aliasing,
    # we might clip frequencies, but let's let the optimizer decide.
    # However, sin(large_freq * x) oscillates rapidly. With 200 points, we can resolve up to ~100 cycles.
    # So frequencies up to ~50 are safe.
    
    y_hat = np.zeros_like(x)
    for i in range(5):
        y_hat += A[i] * np.sin(F[i] * x)
        
    return y_hat
