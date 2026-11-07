<<<<<<< SEARCH
    # Ensure finite values
    result = np.where(np.isfinite(result), result, edge_distance)
    
    # Ensure non-negative
    result = np.maximum(result, 0)
    
    # Ensure diagonal is 0
    np.fill_diagonal(result, 0)
    
    return result
=======
    # Ensure finite values
    result = np.where(np.isfinite(result), result, edge_distance)
    
    # Ensure non-negative
    result = np.maximum(result, 0)
    
    # Ensure diagonal is 0
    np.fill_diagonal(result, 0)
    
    # Final safety: ensure the matrix is symmetric
    result = (result + result.T) / 2.0
    
    return result
>>>>>>> REPLACE
