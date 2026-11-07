import numpy as np

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Create a mask for diagonal to ignore self-loops in ranking
    dm = distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    
    # Get ranks for each row
    # argsort gives indices that would sort the row
    # We need the rank of each element
    ranks = np.zeros_like(dm, dtype=float)
    for i in range(n):
        # Sort indices by distance
        sorted_indices = np.argsort(dm[i])
        # Assign ranks 1 to n
        r = np.arange(1, n+1)
        # Map ranks back to original positions
        ranks[i, sorted_indices] = r
        
    # Heuristic: 1/rank
    # Make it symmetric?
    # If we use H[i,j] = 1/rank[i,j], it's not symmetric.
    # Probability uses H[i,j].
    # Let's use the geometric mean of the two ranks?
    # Or just use the minimum rank (most attractive view)?
    # H[i,j] = 1 / min(rank[i,j], rank[j,i])
    
    # Let's try H[i,j] = 1 / (rank[i,j] + rank[j,i]) * 2 ?
    # Or simply H[i,j] = 1 / rank[i,j] and rely on the fact that ACO is directed?
    # The problem says "closed tour", "travel direction only" for pheromone.
    # But the heuristic is usually symmetric in standard ACO.
    # If H is not symmetric, the ant is biased towards edges where the *source* node finds the *target* close.
    # This is actually a valid "Local" heuristic.
    
    # Let's try a symmetric version:
    # H[i,j] = 1 / sqrt(rank[i,j] * rank[j,i])
    
    # Let's go with the symmetric geometric mean of inverse ranks.
    inv_ranks = 1.0 / ranks
    H = np.sqrt(inv_ranks * inv_ranks.T)
    
    # Ensure diagonal is 0 or very small (though ants don't pick visited nodes)
    np.fill_diagonal(H, 0.0)
    
    # Handle any non-positive or non-finite (though ranks are 1..N, so 1/rank is finite and positive)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
