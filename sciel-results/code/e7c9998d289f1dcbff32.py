<<<<<<< SEARCH
    # Ensure strict disjointness by scaling up slightly.
    # Min distance in honeycomb is sqrt(3) ~ 1.732.
    # We want min distance >= 1.75.
    # Scale factor s = 1.75 / sqrt(3)
    scale = 1.75 / sqrt3
    centers = centers * scale
=======
    # Ensure strict disjointness by scaling up.
    # Min distance in honeycomb is sqrt(3) ~ 1.732.
    # We want min distance >= 1.8 to be safe against floating point/evaluator tolerance.
    # Scale factor s = 1.8 / sqrt(3)
    scale = 1.8 / sqrt3
    centers = centers * scale
>>>>>>> REPLACE
