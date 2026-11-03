<<<<<<< SEARCH
def _make_basis_functions(n, n_basis=30):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 10):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 10):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions with varied widths
    for center_frac in [0.15, 0.3, 0.5, 0.7, 0.85]:
        mu = center_frac * n
        sigma = n / 10
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosines
    for k in [12, 16, 20, 25]:
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Cubic polynomial for more shape flexibility
    basis.append(((x - n/2) / (n/2))**3)
    
    # Additional basis: alternating sign pattern (helps with cancellation)
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    
    # Block patterns for structured cancellation
    block_size = n // 8
    for b in range(4):
        block = np.zeros(n)
        block[b*block_size:(b+1)*block_size] = 1.0
        block[(b+4)*block_size:(b+5)*block_size] = -1.0
        basis.append(block)
    
    return np.array(basis[:n_basis])
=======
def _make_basis_functions(n, n_basis=50):
    """
    Create a set of basis functions for parameterizing h.
    Returns a 2D array of shape (n_basis, N).
    """
    x = np.arange(n, dtype=float)
    basis = []
    
    # Constant function (needed for sum normalization)
    basis.append(np.ones(n))
    
    # Low-frequency cosines
    for k in range(1, 12):
        basis.append(np.cos(2 * np.pi * k * x / n))
    
    # Low-frequency sines
    for k in range(1, 12):
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Gaussian bumps at different positions with varied widths
    for center_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mu = center_frac * n
        sigma = n / 12
        basis.append(np.exp(-(x - mu)**2 / (2 * sigma**2)))
    
    # Cosine window
    basis.append(np.cos(np.pi * x / n))
    
    # Quadratic polynomial
    basis.append((x - n/2)**2 / (n**2))
    
    # Linear ramp
    basis.append(x / n - 0.5)
    
    # Higher frequency cosines
    for k in [14, 18, 22, 28, 35, 40]:
        basis.append(np.cos(2 * np.pi * k * x / n))
        basis.append(np.sin(2 * np.pi * k * x / n))
    
    # Cubic polynomial for more shape flexibility
    basis.append(((x - n/2) / (n/2))**3)
    
    # Additional basis: alternating sign pattern (helps with cancellation)
    basis.append(np.where(x % 2 == 0, 1.0, -1.0))
    basis.append(np.where(x % 3 == 0, 1.0, -1.0))
    basis.append(np.where(x % 4 == 0, 1.0, -1.0))
    
    # Block patterns for structured cancellation
    # Various block sizes
    for block_size in [n // 4, n // 8, n // 16]:
        for b in range(n // block_size // 2):
            block = np.zeros(n)
            block[b*block_size:(b+1)*block_size] = 1.0
            # Opposite phase block
            opp_idx = (b + n // block_size // 2) * block_size
            if opp_idx + block_size <= n:
                block[opp_idx:opp_idx+block_size] = -1.0
            basis.append(block)
            
    # Random smooth basis functions for diversity
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Generate a smooth random function via Fourier series
        coeffs = rng.standard_normal(10)
        func = np.zeros(n)
        for k, c in enumerate(coeffs):
            func += c * np.cos(2 * np.pi * (k+1) * x / n + rng.uniform(0, 2*np.pi))
        basis.append(func)
    
    # Ensure we have at least n_basis, pad with zeros if needed (unlikely)
    if len(basis) < n_basis:
        for i in range(n_basis - len(basis)):
            basis.append(np.zeros(n))
            
    return np.array(basis[:n_basis])
>>>>>>> REPLACE

