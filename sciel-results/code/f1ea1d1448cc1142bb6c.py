<<<<<<< SEARCH
    # Phase 4: Gradient-based polish (smooth surrogate)
    if time.time() < deadline - 15:
        polished = _gradient_polish(best_pos, deadline - 12)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 5: High-precision final polish (progressive beta) - more time
    if time.time() < deadline - 8:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 3)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj
=======
    # Phase 4: Gradient-based polish (smooth surrogate) - vectorized is faster
    if time.time() < deadline - 18:
        polished = _gradient_polish(best_pos, deadline - 15)
        if len(polished) == 3:
            p_pol, r_pol, obj_pol = polished
            if obj_pol > best_obj:
                best_pos = p_pol
                best_obj = obj_pol

    # Phase 5: High-precision final polish (progressive beta) - more time with vectorized
    if time.time() < deadline - 6:
        hp_pos, hp_r, hp_obj = _high_precision_polish(best_pos, deadline - 2)
        if hp_obj > best_obj:
            best_pos = hp_pos
            best_obj = hp_obj
>>>>>>> REPLACE
