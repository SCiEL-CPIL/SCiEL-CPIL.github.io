<<<<<<< SEARCH
    # Initial score
    best_int = internal_edges[:]
    best_score = score(best_int)
=======
    # Initial score
    best_int = internal_edges[:]
    best_score = score(best_int)
    # Initialize best_global with the FULL edge list of the initial best candidate
    # to avoid returning just the internal edges list later
    best_global = get_all_edges(best_int)
>>>>>>> REPLACE

