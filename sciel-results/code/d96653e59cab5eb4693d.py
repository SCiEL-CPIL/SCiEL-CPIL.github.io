<<<<<<< SEARCH
    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
=======
    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (final_center[0], final_center[1]), final_side, final_angle
>>>>>>> REPLACE
