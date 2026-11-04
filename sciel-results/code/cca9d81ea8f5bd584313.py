def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 3:
        return _construct_luw(k, g)
    
    # Try to find a smaller graph via random search?
    # Given the difficulty, we might skip or limit it.
    # Let's try a few random graphs of size near Moore bound.
    # But the LUW graph is a guaranteed valid construction (if the girth formula holds).
    
    # Get the LUW graph
    n_luw, edges_luw = _construct_luw(k, g)
    
    # If we have time, try to find something smaller?
    # The known smallest is 936.
    # LUW with d=8 is 13122.
    # Can we find 936?
    # Random search is unlikely.
    # So we return LUW.
    
    return n_luw, edges_luw
