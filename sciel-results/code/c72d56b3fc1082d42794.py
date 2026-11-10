# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 595.0  # safety margin before the 600s wall-clock kill


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_q^2 with q = 49 = 7^2, as small as possible
    (equivalently, make its complement C as large as possible).

    A set N is a Nikodym set if, for every point x in F_q^2, there exists a
    direction v != 0 such that the punctured line {x + t*v : t in F_q, t != 0}
    is fully contained in N. Equivalently, for every line L, the number of
    points of L that lie in the complement C = F_q^2 \\ N is at most 1.

    Construction: let C be the graph of a perfect nonlinear (PN) function
        f : F_q -> F_q,   C = { (x, f(x)) : x in F_q }.
    A function f is PN iff for every h != 0 the map
        x -> f(x) - f(x+h)
    is a permutation. For such an f, every line of slope m
        L_m,b = { (x, m*x + b) : x in F_q }
    intersects C in exactly one point, because the equation
        f(x) = m*x + b
    has a unique solution: subtract the solution at x=0 (f(0)=b) to get
        f(x) - f(0) = m*x,  and the PN property makes x -> f(x) - f(0) - m*x
    a permutation. Vertical lines intersect C in exactly one point trivially.
    Hence every line meets C in exactly one point, so the punctured line of any
    x is fully in N.  N = F_q^2 \\ C is a valid Nikodym set of size q^2 - q.

    Field model: F_q = F_p[alpha], alpha^2 = k with k a quadratic non-residue
    mod p (k = 3 for p = 7).  An element is the pair (a, b) for a + b*alpha.

    We use f(x) = x^2, which is PN for q = 49 (since gcd(2, q-1) = 1, so x->x^2
    is a permutation and x -> (x+h)^2 - x^2 - m*h = h*(2x + h - m) is a
    permutation in x for every h != 0).

    Returns:
        points: np.ndarray of shape (k, 2, 2), dtype int64.  Each point is
        [[x_a, x_b], [y_a, y_b]] with x = (x_a, x_b), y = (y_a, y_b) in [0, p).
    """
    if time.time() > _DEADLINE:
        raise RuntimeError("deadline exceeded before construction")

    p = 7
    q = p * p

    # Smallest quadratic non-residue mod p -> alpha^2 = k.
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    if k is None:
        k = 3  # fallback (3 is a non-residue mod 7)

    # Field addition and multiplication in F_q, element = (a, b) for a + b*alpha.
    def add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*alpha)(a2 + b2*alpha) = (a1 a2 + b1 b2 k) + (a1 b2 + a2 b1) alpha
        ra = (a1 * a2 + b1 * b2 * k) % p
        rb = (a1 * b2 + a2 * b1) % p
        return (ra, rb)

    def sq(e):
        # e^2 = a^2 + 2ab alpha + b^2 k = (a^2 + b^2 k) + (2ab) alpha
        a, b = e
        ra = (a * a + b * b * k) % p
        rb = (2 * a * b) % p
        return (ra, rb)

    # Build the complement C = graph of f(x) = x^2.
    complement = set()
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            ya, yb = sq(x)
            complement.add((xa, xb, ya, yb))

    # N = all points not in C.
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) in complement:
                        continue
                    points.append(((xa, xb), (ya, yb)))

    if time.time() > _DEADLINE:
        raise RuntimeError("deadline exceeded during construction")

    arr = np.array(points, dtype=np.int64)
    # Reduce mod p (already in range, but be safe) and drop any non-finite.
    arr = np.mod(arr, p)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("non-finite entries in output")
    return arr


# EVOLVE-BLOCK-END
