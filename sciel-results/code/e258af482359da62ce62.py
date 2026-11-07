import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a 10-parameter Fourier series with learnable
    angular frequencies and phases.
    
    The model is a sum of 5 sinusoidal terms:
    f(x) = sum_{k=1}^{5} A_k * cos(omega_k * x + phi_k)
    
    where A_k = 1/k (fixed deterministic decay), and
    omega_k, phi_k are the 10 learnable parameters.
    
    params[0:5] = angular frequencies (omega_1..omega_5)
    params[5:10] = phases (phi_1..phi_5)
    
    Fixed amplitudes with 1/k decay ensure the optimization landscape
    remains well-behaved for BFGS, while the learnable frequencies and
    phases allow the model to capture complex smooth structures through
    constructive interference of harmonics.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if params.shape[0] != 10:
        return np.zeros_like(x)
        
    # Extract frequencies and phases
    omegas = params[0:5]
    phases = params[5:10]
    
    # Fixed amplitudes with 1/k decay
    amplitudes = 1.0 / np.arange(1, 6)
    
    # Compute prediction as sum of 5 cosine terms
    prediction = np.zeros_like(x, dtype=np.float64)
    for k in range(5):
        prediction += amplitudes[k] * np.cos(omegas[k] * x + phases[k])
    
    # Ensure finite output
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
        
    return prediction
