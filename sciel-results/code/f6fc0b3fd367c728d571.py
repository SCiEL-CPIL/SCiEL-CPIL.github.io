<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Start with schoolbook
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition using the split approach:
    # A(4x4) * B(4x7) = A(4x4) * [B1(4x4) | B2(4x3)]
    # = [A*B1(4x4) | A*B2(4x3)]
    
    # For 4x4x4, use Strassen-based decomposition (rank 49)
    # For 4x4x3, use schoolbook (rank 48)
    # Total: 49 + 48 = 97
    
    # Let me implement the Strassen-based 4x4x4 decomposition.
    # The 4x4x4 tensor has rank 49. Here's a construction:
    
    # Actually, let me just try random search for lower ranks.
    # Start from schoolbook and try to reduce the rank.
    
    random.seed(42)
    
    for target_rank in [100, 95, 90, 85, 80, 75, 70, 65, 60]:
        if target_rank < max(n * m, m * p, p * n):
            continue
        if time.time() > deadline - 30:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random factors
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Try to refine using gradient descent or ALS
            # For now, just check if the random initialization works (very unlikely)
            # This is a placeholder; a real implementation would use optimization.
            
            if time.time() > deadline - 10:
                break
    
    # If we didn't find anything better, return schoolbook
    return best_result
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Use the Strassen-based decomposition: rank 49 + 48 = 97
    result = _construct_4x4x7_decomposition()
    
    # Verify the decomposition
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, 
                             len(result["U"][0])):
        return result
    
    # If verification fails, fall back to schoolbook
    return _schoolbook_decomposition(n, m, p)
>>>>>>> REPLACE
