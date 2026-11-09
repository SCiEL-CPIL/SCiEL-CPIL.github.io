"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Calculate the weight (number of non-zero coordinates) of the vector.
    # In F_3^n, vectors are tuples of 0, 1, 2.
    # Higher weight vectors are generally "farther" from the origin and each other,
    # making them less likely to form 3-term arithmetic progressions with 
    # previously selected vectors. 
    # We prioritize high weight vectors first.
    
    weight = sum(1 for x in vector if x != 0)
    
    # To break ties deterministically but with some variation to avoid 
    # pathological symmetries, we add a small fraction based on a hash of the vector.
    # This ensures that vectors of the same weight are not always in the same 
    # lexicographic order, which might be suboptimal for specific structures.
    # We use a simple hash function.
    h = hash(vector)
    # Normalize hash to a small range [0, 1)
    # Python's hash can be negative, so we take modulo.
    tie_breaker = (h % 1000000) / 1000000.0
    
    return weight * 10.0 + tie_breaker

# EVOLVE-BLOCK-END
