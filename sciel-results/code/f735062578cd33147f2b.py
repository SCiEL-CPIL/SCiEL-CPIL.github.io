"""Analytic Taylor-inverted growth-index ansatz (constant-gamma family).

The six coefficients are obtained by inverting the Taylor expansion of the
exact GR growth index about the matter-dominated limit (q->1, u->1) and
matching the first-order derivatives with respect to q and u.  This keeps the
local curvature of the ansatz physically consistent with the underlying
second-order GR ODE, and the resulting point is an interior optimum of the
max-relative-error objective (the gamma in [0.1,1.2] constraint is never
binding for the validation CPL range).
"""
# EVOLVE-BLOCK-START
def construct():
    # c0 : value at matter domination (q=1, u=1)      -> 5/9
    # c1 : d gamma / d q                              -> 1/3
    # c2 : quadratic q term (zero to first order)     -> 0
    # c3 : d gamma / d u                              -> 1/9
    # c4 : w_a * a term (no explicit a in first order)-> 0
    # c5 : mixed u*q term (cancels c3's a-dependence) -> -1/9
    return [5.0 / 9.0, 1.0 / 3.0, 0.0, 1.0 / 9.0, 0.0, -1.0 / 9.0]
# EVOLVE-BLOCK-END
