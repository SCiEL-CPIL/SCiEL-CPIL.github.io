    SPACING = 2.0 + 1e-6
    def axial_to_cart(i, j):
        x = SPACING * i + 0.5 * SPACING * j  # Wait, the basis vectors need to scale
        # Original basis: v1=(2,0), v2=(1, sqrt3). Length of v1 is 2. Length of v2 is 2.
        # If we scale by k, v1'=(2k, 0), v2'=(k, k*sqrt3).
        # So x = 2k*i + k*j, y = k*sqrt3*j.
        k = 1.0 + 5e-7 # 2k = 2.000001
        x = 2.0 * k * i + k * j
        y = k * sqrt3 * j
        return x, y
