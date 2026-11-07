import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use all 10 params to define 5 Gaussian RBFs with learnable centers and widths
    # Each RBF i (i=0..4) uses params[2*i] as amplitude, params[2*i+1] as center,
    # and we derive width from a fixed base to keep it simple
    # Actually, let's use a different strategy:
    # Use 5 RBFs with fixed centers but learnable amplitudes and a global width
    # Or better: use 10 params as amplitudes for 10 RBFs with fixed centers
    # But make the centers and widths adaptive via a smooth transformation
    
    # Strategy: Use 10 RBFs with centers derived from params in a stable way
    # First 5 params: amplitudes for RBFs at fixed centers
    # Last 5 params: control the spread (width) of each RBF
    
    # Fixed base centers
    base_centers = np.linspace(-2.0, 2.0, 10)
    
    # Extract amplitudes and width parameters
    amps = params[:5]
    widths = params[5:]
    
    # Ensure widths are positive by using softplus-like transformation
    # width_i = 0.5 + exp(widths[i]) to keep them in a reasonable range
    sigma_vals = 0.5 + np.exp(np.clip(widths, -5, 5))
    
    # For the first 5 RBFs, use the first 5 centers
    centers = base_centers[:5]
    
    # Compute Gaussian basis for 5 RBFs
    # x shape: (N,)
    # centers shape: (5,)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # sigma_vals shape: (5,)
    # We need to compute exp(-((x - c_i)^2) / (2 * sigma_i^2)) for each i
    # diff shape: (N, 5)
    # sigma_vals shape: (5,)
    phi = np.exp(-0.5 * (diff / sigma_vals[np.newaxis, :]) ** 2)
    
    # Linear combination: sum(amps[i] * phi[i])
    # amps shape: (5,)
    # result shape: (N,)
    return phi @ amps
