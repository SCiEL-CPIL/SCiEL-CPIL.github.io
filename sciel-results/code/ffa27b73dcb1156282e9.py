    side = try_configuration(pattern)
    if side < best_side:
        best_side = side
        best_inner = pattern
        # ... recalculate center/angle
    