<<<<<<< SEARCH
def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns Newick string.
    """
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if n == 1 else ""

    if n == 2:
        return f"({labels[0]},{labels[1]});"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)

    # Time check
    elapsed = time.time() - start_time
    if elapsed > 12.0:
        # Still return the result; NJ is fast enough typically
        pass

    return result
=======
def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns Newick string.
    """
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if n == 1 else ""

    if n == 2:
        return f"({labels[0]},{labels[1]});"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)

    # NNI refinement if time permits
    elapsed = time.time() - start_time
    if elapsed < 120.0 and n >= 4:
        result = _nni_refine(result, D, labels, time_budget=min(60.0, 180.0 - elapsed))

    return result
>>>>>>> REPLACE
