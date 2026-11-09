"""Symbolic regression with 3 inputs – orthonormal basis via shifted Legendre polynomials."""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression to recover an unknown function of 3 inputs.
    Uses a linear-in-params model with a basis of shifted Legendre polynomials.
    This basis is well-conditioned over [0.5, 2], aiding BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Normalize inputs to [0, 1] for standard Legendre polynomial scaling, 
    # then map to [0.5, 2].
    # Actually, simpler: map x, y, z from [0.5, 2] to t in [-1, 1] for standard Legendre P_n(t)
    # t = (2*u - (2+0.5)) / (2-0.5) = (2*u - 2.5) / 1.5
    
    def map_to_unit_interval(u):
        # Maps [0.5, 2] to [0, 1]
        return (u - 0.5) / 1.5

    def legendre_poly(u, deg):
        """
        Compute shifted Legendre polynomial of given degree on [0, 1].
        Standard Legendre P_n(t) is on [-1, 1].
        Shifted: P_n(2u - 1) where u in [0, 1].
        """
        t = 2.0 * u - 1.0
        if deg == 0:
            return np.ones_like(u)
        elif deg == 1:
            return t
        elif deg == 2:
            return 0.5 * (3.0 * t * t - 1.0)
        else:
            # General recurrence if needed, but max deg here is 2
            p0 = np.ones_like(u)
            p1 = t
            for k in range(2, deg + 1):
                pk = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k
                p0 = p1
                p1 = pk
            return p1

    ux = map_to_unit_interval(x)
    uy = map_to_unit_interval(y)
    uz = map_to_unit_interval(z)

    # Calculate basis functions
    # P_0(u) = 1
    # P_1(u) = 2u - 1
    # P_2(u) = 6u^2 - 6u + 1  (which is 0.5*(3*(2u-1)^2 - 1))

    # Let's explicitly calculate the 10 features
    # B0: Constant
    f0 = np.ones_like(x)
    
    # B1-B3: Linear in each var
    f1 = legendre_poly(ux, 1)
    f2 = legendre_poly(uy, 1)
    f3 = legendre_poly(uz, 1)
    
    # B4-B6: Quadratic in each var
    f4 = legendre_poly(ux, 2)
    f5 = legendre_poly(uy, 2)
    f6 = legendre_poly(uz, 2)
    
    # B7-B9: Interactions (Linear x Linear)
    f7 = f1 * f2
    f8 = f1 * f3
    f9 = f2 * f3

    # Combine
    pred = (params[0] * f0 +
            params[1] * f1 +
            params[2] * f2 +
            params[3] * f3 +
            params[4] * f4 +
            params[5] * f5 +
            params[6] * f6 +
            params[7] * f7 +
            params[8] * f8 +
            params[9] * f9)

    return pred
