import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # Field operations for F_49 = F_7[alpha] where alpha^2 = 3
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def field_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    def field_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*3) + (a0*b1 + a1*b0)*alpha
        a0, a1 = a
        b0, b1 = b
        return ((a0 * b0 + a1 * b1 * k) % p, (a0 * b1 + a1 * b0) % p)

    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    def field_inv(a):
        # a = (a0, a1). Inverse is (a0 - a1*alpha) / (a0^2 - 3*a1^2)
        # since (a0 + a1*alpha)(a0 - a1*alpha) = a0^2 - 3*a1^2
        a0, a1 = a
        denom = (a0 * a0 - 3 * a1 * a1) % p
        if denom == 0:
            raise ZeroDivisionError("Cannot invert zero")
        # Invert denom in F_7
        denom_inv = pow(denom, -1, p)
        return ((a0 * denom_inv) % p, ((-a1) * denom_inv) % p)

    # Precompute all elements of F_49
    all_elements = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(all_elements)}
    idx_to_elem = {i: e for i, e in enumerate(all_elements)}

    # Precompute squares to identify non-squares
    squares = set()
    for e in all_elements:
        squares.add(field_mul(e, e))
    
    zero = (0, 0)
    one = (1, 0)

    # Generate the parabola S1 = {(x, x^2) : x in F_49}
    S1 = []
    for x in all_elements:
        x2 = field_mul(x, x)
        S1.append((x, x2))
    
    S1_set = set(S1)

    # Find a point P not in S1 such that for all x in F_49,
    # the equation (px - x)^2 = py - x^2 has no solution
    # i.e., (px - x)^2 - (py - x^2) is a non-square for all x
    
    # Actually, the condition for P to be a "valid" extra point is:
    # For every x in F_49, the line through (x, x^2) and P must not be a tangent to the parabola at x.
    # The tangent at x has slope 2x. The line through P and (x, x^2) has slope m = (py - x^2)/(px - x) (if px != x).
    # We need m != 2x for all x where px != x.
    # And if px == x, then we need py != x^2 (which is true since P not in S1).
    
    # So condition: for all x in F_49, if px != x, then (py - x^2)/(px - x) != 2x.
    # Equivalent to: (py - x^2) != 2x*(px - x) = 2*px*x - 2*x^2
    # py - x^2 != 2*px*x - 2*x^2
    # py + x^2 != 2*px*x
    # This must hold for all x.
    
    # Let's check candidates for P.
    # We iterate over all points not in S1.
    
    def is_valid_extra_point(P):
        px, py = P
        for x in all_elements:
            x2 = field_mul(x, x)
            if field_sub(px, x) == zero:
                # px == x, then we need py != x^2, which is true since P not in S1
                continue
            # Check if (py - x^2) == 2*px*x - 2*x^2
            # LHS: py - x^2
            lhs = field_sub(py, x2)
            # RHS: 2*px*x - 2*x^2 = 2*(px*x - x^2)
            px_x = field_mul(px, x)
            px_x_minus_x2 = field_sub(px_x, x2)
            rhs = field_mul((2 % p, 0), px_x_minus_x2)
            if lhs == rhs:
                return False
        return True

    # Find P
    P = None
    for candidate in all_elements:
        if candidate in S1_set:
            continue
        if is_valid_extra_point(candidate):
            P = candidate
            break

    if P is None:
        # Fallback: just use the parabola (size q)
        S = S1
    else:
        S = S1 + [P]

    S_set = set(S)

    # Build N = F_49^2 \ S
    N_points = []
    for x in all_elements:
        for y in all_elements:
            if (x, y) not in S_set:
                N_points.append([list(x), list(y)])

    N_array = np.array(N_points, dtype=np.int64)
    
    # Ensure shape is (k, 2, 2)
    if len(N_points) == 0:
        # This shouldn't happen, but just in case
        N_array = np.zeros((1, 2, 2), dtype=np.int64)

    return N_array
