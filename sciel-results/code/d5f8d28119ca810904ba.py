"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    We use a universal approximator: a 3-layer tanh network with 2 hidden units per layer.
    This allows modeling complex non-linear interactions and saturation effects
    that polynomial bases cannot capture well.
    
    Architecture:
    - Input: [x, y, z]
    - Hidden layer 1: 2 tanh units
    - Hidden layer 2: 2 tanh units
    - Output: 1 linear unit
    
    Parameters (10 total):
    p[0..2]: weights from input to hidden layer 1 (2 units x 3 inputs + 2 biases = 8, but we use a compact form)
    
    Actually, let's use a simpler but still expressive form:
    - 2 hidden units in layer 1, 2 in layer 2, 1 output
    - Total parameters: 3*2+2 (layer 1) + 2*2+2 (layer 2) + 2*1+1 (output) = 6+2+4+2+2+1 = 17... too many
    
    Let's use a different approach: a single hidden layer with 5 units and 1 output.
    Parameters: 3*5+5 (hidden) + 5*1+1 (output) = 15+5+5+1 = 26... still too many
    
    We have exactly 10 parameters. Let's use:
    - 2 hidden tanh units
    - Parameters: w1(3), b1(2), w2(2), b2(1) = 3+2+2+1 = 8... plus 2 more for scaling
    
    Actually, let's use a more flexible basis:
    p[0]: bias
    p[1..3]: linear coefficients for x, y, z
    p[4..6]: quadratic coefficients for x^2, y^2, z^2
    p[7]: coefficient for x*y*z (trilinear)
    p[8]: coefficient for tanh(x) * tanh(y) * tanh(z)
    p[9]: coefficient for exp(-((x-1)^2 + (y-1)^2 + (z-1)^2))
    
    This gives a mix of polynomial, trilinear, and localized terms.
    """
    p = params
    # Center the inputs around 1 for better conditioning
    xc = x - 1.0
    yc = y - 1.0
    zc = z - 1.0
    
    # Polynomial terms
    linear = p[1] * x + p[2] * y + p[3] * z
    quadratic = p[4] * x**2 + p[5] * y**2 + p[6] * z**2
    
    # Trilinear interaction
    trilinear = p[7] * x * y * z
    
    # Nonlinear interaction using tanh
    tanh_interaction = p[8] * np.tanh(x) * np.tanh(y) * np.tanh(z)
    
    # Localized Gaussian term
    gaussian = p[9] * np.exp(-(xc**2 + yc**2 + zc**2))
    
    return p[0] + linear + quadratic + trilinear + tanh_interaction + gaussian
# EVOLVE-BLOCK-END
