"""Rewritten program: search for 6-point integral sets in Q(sqrt(D)) with rational distances."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from itertools import combinations

def _squarefree(n):
    """Return the squarefree part of n and the square factor: n = sf * k^2."""
    if n == 0:
        return 0, 0
    sf = 1
    k = 1
    m = abs(n)
    p = 2
    while p * p <= m:
        while m % (p * p) == 0:
            m //= (p * p)
            k *= p
        p += 1
    sf = m
    return sf, k

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _check_validity(points, D):
    """
    points: list of (x_num, x_den, y_num, y_den)
    D: int
    Check:
      - all pairwise distances rational (i.e., squared distance is a rational square)
      - no three collinear
      - no four concyclic
    Returns True if valid.
    """
    n = len(points)
    if n < 3:
        return False

    # Represent each point as (x, y) in Q(sqrt(D)): x = a/b, y = c/d
    # Point i: (xi, yi) where xi = x_num_i / x_den_i, yi = y_num_i / y_den_i
    # Distance squared between i and j:
    #   (xi - xj)^2 + D * (yi - yj)^2
    # This must be a rational square.

    # Precompute rational coordinates
    xs = [Fraction(p[0], p[1]) for p in points]
    ys = [Fraction(p[2], p[3]) for p in points]

    # Check pairwise distances
    for i, j in combinations(range(n), 2):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        d2 = dx * dx + D * dy * dy
        if d2 < 0:
            return False
        # d2 is a Fraction; check if it's a perfect square in Q
        # d2 = p/q in lowest terms; need p and q to be perfect squares
        p = d2.numerator
        q = d2.denominator
        if p == 0:
            continue  # same point, but we check distinctness separately
        if not _is_square(p) or not _is_square(q):
            return False

    # Check distinctness
    for i, j in combinations(range(n), 2):
        if xs[i] == xs[j] and ys[i] == ys[j]:
            return False

    # Check no three collinear
    for i, j, k in combinations(range(n), 3):
        # Collinear if (xj - xi)*(yk - yi) == (xk - xi)*(yj - yi)
        # All in Q, so exact
        if (xs[j] - xs[i]) * (ys[k] - ys[i]) == (xs[k] - xs[i]) * (ys[j] - ys[i]):
            return False

    # Check no four concyclic
    # Four points are concyclic iff the determinant of the circle equation is zero.
    # Circle through (x1,y1),(x2,y2),(x3,y3): x^2 + y^2 + Dx + Ey + F = 0
    # For point (x,y) in our space: x^2 + D*y^2 + Dx + Ey + F = 0
    # (Note: the "y^2" term becomes D*y^2 because y is coefficient of sqrt(D))
    # Actually, let's be careful. The points are (x, y*sqrt(D)) in the plane.
    # A circle in the plane: (X - h)^2 + (Y - k)^2 = r^2
    # Expanding: X^2 + Y^2 - 2hX - 2kY + (h^2 + k^2 - r^2) = 0
    # With X = x, Y = y*sqrt(D): x^2 + D*y^2 - 2h*x - 2k*y*sqrt(D) + C = 0
    # For this to hold for rational x, y, the sqrt(D) term must vanish or be handled.
    # Actually, a circle in the Euclidean plane with rational center and radius won't necessarily
    # have the form above with rational coefficients in x, y.
    # 
    # Better approach: four points (x1,y1),...,(x4,y4) in the plane are concyclic iff
    # the determinant of the 4x4 matrix with rows [x_i^2 + y_i^2, x_i, y_i, 1] is zero,
    # where (x_i, y_i) are the actual plane coordinates.
    # Here plane coordinates are (x_i, y_i * sqrt(D)).
    # So x_i^2 + y_i^2 in plane = x_i^2 + D*y_i^2 (since (y_i*sqrt(D))^2 = D*y_i^2)
    # But the x_i and y_i in the determinant should be the plane coordinates:
    #   row i: [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
    # The determinant of this matrix must be zero for concyclicity.
    # However, y_i*sqrt(D) is irrational in general. The determinant will be in Q(sqrt(D)).
    # For it to be zero, both the rational and sqrt(D) parts must be zero.
    #
    # Alternative: use the condition that four points are concyclic iff
    # the power of each point with respect to the circle through the other three is zero.
    # 
    # Simpler: use the cross-ratio or the determinant approach with exact arithmetic.
    # Let's use the determinant of:
    # | x_i^2 + D*y_i^2, x_i, y_i, 1 |
    # Wait, that's not right either because the plane coordinates involve sqrt(D).
    #
    # Let me think again. The plane coordinates are P_i = (x_i, y_i * sqrt(D)).
    # A circle is defined by: |P - C|^2 = r^2 where C = (h, k) and r are real.
    # Expanding: x^2 + D*y^2 - 2h*x - 2k*y*sqrt(D) + (h^2 + k^2 - r^2) = 0
    # For this to be satisfied by points with rational x, y, we can write:
    # x^2 + D*y^2 + A*x + B*y + C = 0  where A, B, C are real.
    # But actually, for four points to be concyclic, there exists (h,k,r) such that all four satisfy it.
    # This gives a system of 4 equations in 3 unknowns (h, k, h^2+k^2-r^2).
    # Subtracting pairs, we get linear equations in h and k.
    #
    # Let's use the standard determinant test for concyclicity in the plane.
    # Four points (X_i, Y_i) are concyclic iff:
    # det [[X_i^2 + Y_i^2, X_i, Y_i, 1] for i=1..4] = 0
    # Here X_i = x_i, Y_i = y_i * sqrt(D).
    # So the matrix is:
    # [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
    # The determinant is in Q(sqrt(D)). Let's compute it symbolically.
    # Let M be the 4x4 matrix. det(M) = a + b*sqrt(D) for some a, b in Q.
    # det(M) = 0 iff a = 0 and b = 0.
    #
    # To compute det(M) = a + b*sqrt(D), we can use the fact that:
    # det(M) = det(M_conj) where M_conj replaces y_i*sqrt(D) with -y_i*sqrt(D).
    # Actually, det(M) + det(M_conj) = 2a and det(M) - det(M_conj) = 2b*sqrt(D).
    # But computing two determinants is expensive. Instead, let's just compute the determinant
    # as an element of Q(sqrt(D)) using 2x2 block representation.
    #
    # Element of Q(sqrt(D)): (a, b) representing a + b*sqrt(D)
    # Addition: (a1+b1*sqrt(D)) + (a2+b2*sqrt(D)) = (a1+a2, b1+b2)
    # Multiplication: (a1,b1)*(a2,b2) = (a1*a2 + D*b1*b2, a1*b2 + a2*b1)
    #
    # Let's implement a simple 4x4 determinant over Q(sqrt(D)) using Gaussian elimination.

    def add_qsd(a1, b1, a2, b2):
        return (a1 + a2, b1 + b2)

    def sub_qsd(a1, b1, a2, b2):
        return (a1 - a2, b1 - b2)

    def mul_qsd(a1, b1, a2, b2):
        return (a1 * a2 + D * b1 * b2, a1 * b2 + a2 * b1)

    def div_qsd(a1, b1, a2, b2):
        # Divide (a1 + b1*sqrt(D)) / (a2 + b2*sqrt(D))
        # = (a1 + b1*sqrt(D)) * (a2 - b2*sqrt(D)) / (a2^2 - D*b2^2)
        denom = a2 * a2 - D * b2 * b2
        if denom == 0:
            raise ZeroDivisionError("Division by zero in Q(sqrt(D))")
        na, nb = mul_qsd(a1, b1, a2, -b2)
        return (na / denom, nb / denom)

    # Build the 4x4 matrix with entries in Q(sqrt(D))
    # Row i: [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
    # Entry (i,0): (x_i^2 + D*y_i^2, 0)
    # Entry (i,1): (x_i, 0)
    # Entry (i,2): (0, y_i)
    # Entry (i,3): (1, 0)

    for i, j, k, l in combinations(range(n), 4):
        # Build 4x4 matrix
        pts = [i, j, k, l]
        # Matrix as list of 4 rows, each row is 4 entries, each entry is (a, b)
        M = []
        for p in pts:
            row = [
                (xs[p] * xs[p] + D * ys[p] * ys[p], Fraction(0)),
                (xs[p], Fraction(0)),
                (Fraction(0), ys[p]),
                (Fraction(1), Fraction(0)),
            ]
            M.append(row)

        # Gaussian elimination to compute determinant
        # Work on a copy
        det = (Fraction(1), Fraction(0))  # 1 + 0*sqrt(D)
        n_mat = 4
        for col in range(n_mat):
            # Find pivot
            pivot_row = None
            for r in range(col, n_mat):
                a, b = M[r][col]
                if a != 0 or b != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                det = (Fraction(0), Fraction(0))
                break
            if pivot_row != col:
                M[col], M[pivot_row] = M[pivot_row], M[col]
                det = div_qsd(*sub_qsd(0, 0, *det))  # negate det
            pivot_a, pivot_b = M[col][col]
            det = mul_qsd(*det, pivot_a, pivot_b)
            # Eliminate below
            for r in range(col + 1, n_mat):
                ra, rb = M[r][col]
                if ra != 0 or rb != 0:
                    factor_a, factor_b = div_qsd(ra, rb, pivot_a, pivot_b)
                    for c in range(col, n_mat):
                        M[r][c] = sub_qsd(M[r][c][0], M[r][c][1],
                                          *mul_qsd(factor_a, factor_b, M[col][c][0], M[col][c][1]))

        da, db = det
        if da != 0 or db != 0:
            continue  # not concyclic
        else:
            return False

    return True

def _search_6points(deadline):
    """Search for a 6-point integral set in Q(sqrt(D)) with rational distances."""
    # Strategy: search over small D and small rational coordinates.
    # For D=1, we need integer or rational points with integer distances.
    # Let's try to find 6 points by extending the known 5-point set.
    
    # Known 5-point set:
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try adding a 6th point near the existing ones
    # Search in a box around the existing points
    # The existing points span x in [-56, 16], y in [0, 63]
    
    best = None
    base_set = set(base)
    
    # Try various D values
    for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95, 97, 101, 102, 103, 105, 106, 107, 109, 110, 111, 113, 114, 115, 118, 119, 121, 122, 123, 127, 129, 130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 149, 150, 151, 153, 154, 155, 157, 158, 159, 161, 162, 163, 165, 166, 167, 169, 170, 171, 173, 174, 175, 177, 178, 179, 181, 182, 183, 185, 186, 187, 190, 191, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 209, 210, 211, 213, 214, 215, 217, 218, 219, 221, 222, 223, 226, 227, 229, 230, 231, 233, 234, 235, 237, 238, 239, 241, 242, 243, 245, 246, 247, 249, 250, 251, 253, 254, 255, 257, 258, 259, 261, 262, 263, 265, 266, 267, 269, 270, 271, 273, 274, 275, 277, 278, 279, 281, 282, 283, 285, 286, 287, 289, 290, 291, 293, 294, 295, 297, 298, 299, 301, 302, 303, 305, 306, 307, 309, 310, 311, 313, 314, 315, 317, 318, 319, 321, 322, 323, 325, 326, 327, 329, 330, 331, 333, 334, 335, 337, 338, 339, 341, 342, 343, 345, 346, 347, 349, 350, 351, 353, 354, 355, 357, 358, 359, 361, 362, 363, 365, 366, 367, 369, 370, 371, 373, 374, 375, 377, 378, 379, 381, 382, 383, 385, 386, 387, 389, 390, 391, 393, 394, 395, 397, 398, 399, 401, 402, 403, 405, 406, 407, 409, 410, 411, 413, 414, 415, 417, 418, 419, 421, 422, 423, 425, 426, 427, 429, 430, 431, 433, 434, 435, 437, 438, 439, 441, 442, 443, 445, 446, 447, 449, 450, 451, 453, 454, 455, 457, 458, 459, 461, 462, 463, 465, 466, 467, 469, 470, 471, 473, 474, 475, 477, 478, 479, 481, 482, 483, 485, 486, 487, 489, 490, 491, 493, 494, 495, 497, 498, 499, 500]:
        if time.time() > deadline:
            break
        if D == 1:
            # For D=1, try integer points
            # Search for a 6th point that works with the base 5
            for x in range(-100, 101):
                if time.time() > deadline:
                    break
                for y in range(-100, 101):
                    if (x, y) in base_set:
                        continue
                    candidate = base + [(x, y)]
                    if _check_validity(candidate, 1):
                        return {'D': 1, 'points': [[[p[0], 1], [p[1], 1]] for p in candidate]}
        else:
            # For D > 1, try small rational points
            # Keep it simple: try integer x, y first
            for x in range(-20, 21):
                if time.time() > deadline:
                    break
                for y in range(-20, 21):
                    candidate = [(0,0), (x, y)]
                    # Just check if these two have rational distance
                    d2 = x*x + D*y*y
                    if _is_square(d2):
                        # Try to build up from here, but this is too slow
                        pass
    return None

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin

    # First try to find a 6-point set
    result = _search_6points(deadline)
    if result is not None:
        return result

    # Fall back to the known 5-point set
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
