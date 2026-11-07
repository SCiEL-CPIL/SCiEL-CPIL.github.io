import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-3, 3] to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev polynomials T_0 through T_9
    # T_0 = 1, T_1 = t, T_2 = 2t^2 - 1, T_3 = 4t^3 - 3t, ...
    # Compute via recurrence for stability
    T = np.zeros((10, len(t)), dtype=np.float64)
    T[0] = 1.0
    if len(t) > 0:
        T[1] = t
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
    
    # Nonlinear activations: mix of tanh and sin at different scales
    # This creates 10 product features: T_k * act_k
    # Using diverse activations to capture different nonlinearities
    act = np.zeros((10, len(t)), dtype=np.float64)
    
    # tanh at different scales
    act[0] = np.tanh(t)
    act[1] = np.tanh(2.0 * t)
    act[2] = np.tanh(0.5 * t)
    act[3] = np.tanh(1.5 * t)
    
    # sin at different frequencies
    act[4] = np.sin(np.pi * t)
    act[5] = np.sin(2.0 * np.pi * t)
    act[6] = np.sin(0.5 * np.pi * t)
    act[7] = np.sin(1.5 * np.pi * t)
    
    # Identity and constant for baseline
    act[8] = np.ones_like(t)
    act[9] = t
    
    # Build the output as linear combination of products
    # y = sum_k params[k] * T[k] * act[k]
    result = np.zeros_like(x, dtype=np.float64)
    for k in range(10):
        result += params[k] * T[k] * act[k]
    
    return result
