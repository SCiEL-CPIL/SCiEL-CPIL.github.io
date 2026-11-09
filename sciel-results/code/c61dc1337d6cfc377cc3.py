import math
import time

def construct():
    # Use the 12 smallest positive zeros associated with Bessel J_{1/2}:
    # z_i = (i * pi)^2 for i = 1, 2, ..., 12.
    # This exploits the asymptotic orthogonality of Laguerre polynomials
    # to the Bessel measure, confining the quotient q(x)'s largest root B.
    deadline = time.time() + 600.0
    z = [(i * math.pi) ** 2 for i in range(1, 13)]
    result = sorted([float(zi) for zi in z])
    # Safety: verify constraints
    if any(not (math.isfinite(zi) and zi > 0) for zi in result):
        raise ValueError("Invalid nodes")
    if len(set(result)) != 12:
        raise ValueError("Nodes not pairwise distinct")
    return result
