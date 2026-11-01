def axial_to_cartesian(q, r, spacing=1.0):
    x = 1.5 * spacing * q
    y = SQRT3 * spacing * (r + q / 2.0)
    return x, y
