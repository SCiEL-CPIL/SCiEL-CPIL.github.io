import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse signed step function to minimize the autoconvolution peak.
    By selecting a single non-zero interval, the autoconvolution is reduced to a single spike,
    minimizing both the peak value and the side-lobe energy.
    """
    # Initialize with zeros
    x = np.zeros(N)
    
    # Select a random index to place the non-zero value
    # Using a fixed seed for reproducibility in the evaluation context if needed,
    # or just random. The task says "Initialize this subset using a random permutation".
    # For M=1, the position doesn't affect the autoconvolution magnitude.
    idx = np.random.randint(N)
    
    # Set the value to 1.0
    x[idx] = 1.0
    
    return x
# EVOLVE-BLOCK-END
