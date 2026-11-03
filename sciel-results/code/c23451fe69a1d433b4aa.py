<<<<<<< SEARCH
    # Final validation
    if best_rows is not None:
        if not validate_rows(best_rows):
            fb = greedy_baseline()
            if fb:
                best_rows = fb
                best_scope = get_scope(fb)

    return best_rows
=======
    # Final validation
    if best_rows is not None:
        if not validate_rows(best_rows):
            fb = greedy_baseline()
            if fb:
                best_rows = fb
                best_scope = get_scope(fb)

    return best_rows
>>>>>>> REPLACE
