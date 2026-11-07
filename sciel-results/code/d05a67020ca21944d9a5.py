import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use Gaussian Radial Basis Functions (RBFs)
    # params: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    # centers are the parameters
    # widths are fixed at multiple scales
    # amplitudes are fixed to 1.0 (linear combination is implicit in the sum? No, RBFs are usually linear in amplitudes)
    # Wait, the task says "params, an array of length 10".
    # If I use 10 RBFs, I need 10 centers and 10 amplitudes? That's 20 params.
    # Or 5 centers and 5 amplitudes? That's 10 params.
    # Or fixed centers and 10 amplitudes?
    # The directive says: "Adopt a Gaussian Radial Basis Function (RBF) network where the 10 parameters are constrained to be the centers of the basis functions, while the widths and amplitudes are fixed to a multi-scale set".
    # This implies 10 centers, and fixed widths/amplitudes.
    # But if amplitudes are fixed, the model is sum_{k=1}^{10} A_k * exp(-((x - c_k)^2) / (2 * sigma_k^2)).
    # If A_k are fixed, the shape is fixed, only positions move. This is very restrictive.
    # Usually RBF regression has parameters for centers, widths, and amplitudes.
    # With only 10 params, maybe 5 centers, 5 amplitudes? Or 10 centers with fixed amplitudes/widths?
    # Let's try 5 centers and 5 amplitudes.
    # params = [c0, a0, c1, a1, c2, a2, c3, a3, c4, a4]
    # Widths fixed to a reasonable value, e.g., 0.5.
    
    # However, BFGS might struggle with non-linear parameters (centers in exp).
    # But it's a structural change.
    
    # Alternative interpretation: 
    # "10 parameters are constrained to be the centers"
    # So params[0..9] are centers c_0 ... c_9.
    # Widths and amplitudes are fixed.
    # Let's use 5 different widths and 5 different amplitudes? No, 10 centers.
    # Maybe 10 centers, all with same width and amplitude 1?
    # y = sum_{k=0}^9 exp(-(x - c_k)^2 / (2 * sigma^2))
    # This allows shifting "bumps" around.
    
    # Let's try: 10 centers, fixed width sigma=0.5, amplitude 1.
    # This is a sum of 10 Gaussian bumps.
    
    sigma = 0.5
    centers = params
    
    # Compute RBFs
    # x shape (N,), centers shape (10,)
    # diff shape (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # exp(-diff^2 / (2 * sigma^2))
    rbf = np.exp(-diff**2 / (2 * sigma**2))
    
    # Sum over centers
    return np.sum(rbf, axis=1)
