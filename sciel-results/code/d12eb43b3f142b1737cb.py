def value_uv3(u3, v3):
    # c2=1, c0=u3, c1=v3
    coeffs = build_P(u3, v3, 1.0)
    return value_from_coeffs(coeffs)
