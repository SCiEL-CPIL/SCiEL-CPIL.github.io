def strassen_terms(n):
    if n == 1:
        # U, V, W are scalars 1
        return [ (np.array([1.]), np.array([1.]), np.array([1.])) ]
    
    half = n // 2
    # Define the 7 linear combinations
    # We need to map the indices of the n x n matrix to the half x half blocks.
    
    # This is getting complicated to do with arrays directly in a recursive way
    # without knowing the final index mapping.
    
    # Alternative: Use the known explicit decomposition for 4x4.
    # Or just implement the recursion carefully.
