def priority(vector, n):
    # Calculate Hamming weight (number of non-zero elements)
    # Vectors with higher weight are prioritized.
    # This is a structural heuristic, distinct from random hashing.
    weight = sum(1 for x in vector if x != 0)
    return float(weight)
