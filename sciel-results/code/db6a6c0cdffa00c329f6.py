# EVOLVE-BLOCK-START
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    The field is F_49 = F_7[alpha] with alpha^2 = 3.
    We use a construction based on the Hermitian unital and a blocking set.
    
    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array.
    """
    p = 7
    k_const = 3  # alpha^2 = 3

    # Special y-coordinate set Y = {(u^2 mod p, v)}: y_a must be a square mod p.
    # Squares mod 7: 0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
    # So squares are {0, 1, 2, 4}
    squares = {0, 1, 2, 4}
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    # Check if in unital: Trace(x) + Norm(y) = 0
                    # Trace((xa, xb)) = xa + xa*alpha^2 = xa + 3*xa = 4*xa (since alpha^2=3)
                    # Wait, the trace from F_49 to F_7 of (a + b*alpha) is a + (a*alpha)^7
                    # (a + b*alpha)^7 = a^7 + b^7*alpha^7 = a + b*alpha^7
                    # alpha^2 = 3 => alpha^4 = 9 = 2, alpha^7 = alpha^4 * alpha^2 = 2*3 = 6 = -1
                    # So (a + b*alpha)^7 = a - b*alpha
                    # Trace(a + b*alpha) = (a + b*alpha) + (a - b*alpha) = 2a
                    # So Trace(x) = 2*xa
                    
                    # Norm(y) = y * y^7 = (ya + yb*alpha)*(ya - yb*alpha) = ya^2 - yb^2*alpha^2 = ya^2 - 3*yb^2
                    trace_x = (2 * xa) % p
                    norm_y = (ya * ya - yb * yb * k_const) % p
                    in_unital = (trace_x + norm_y) % p == 0
                    
                    # Check if y is in special set Y: ya must be a square mod 7
                    in_y_special = ya in squares
                    
                    if in_unital and not in_y_special:
                        continue  # excluded point
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
