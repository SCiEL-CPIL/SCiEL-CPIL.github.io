import numpy as np

N = 600

# EVOLVE-BLOCK-START
def construct():
    # Use a Gaussian profile to potentially lower the ratio max(f*f)/||f||_1^2
    # compared to the constant function.
    # The constant function yields a ratio of 2 for L=0.5.
    # A Gaussian with appropriate width might yield a lower ratio.
    
    # Define the grid
    x = np.linspace(-0.25, 0.25, N)
    
    # Choose a standard deviation. 
    # For a Gaussian, the ratio is ~ 1/(sigma * sqrt(2*pi)).
    # To minimize ratio, we want large sigma.
    # But we are constrained to [-0.25, 0.25].
    # If we set sigma too large, the function is nearly constant in the middle and drops at edges.
    # If sigma is small, it's a spike.
    # Let's try sigma = 0.1 (approx 1/10) or 0.12.
    # L = 0.5. Half-width = 0.25.
    # A Gaussian with sigma = 0.125 (L/4) is a good candidate.
    sigma = 0.125
    f = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Ensure non-negative (Gaussian is always positive)
    # Return as list
    return list(f)
# EVOLVE-BLOCK-END
