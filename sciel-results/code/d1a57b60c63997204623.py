<<<<<<< SEARCH
    # Use fast NJ for topology recovery
    result = _nj_newick_fast(dist, labels)

    return result
=======
    # Use quartet-based supertree for better topology recovery
    # The quartet method is more robust to distance matrix errors
    time_budget = max(5.0, _DEADLINE - time.time() - 15)
    result = _quartet_supertree(dist, labels, time_budget)

    return result
>>>>>>> REPLACE
