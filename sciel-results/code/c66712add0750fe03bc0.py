import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1]
    # center = 1.25, half_range = 0.75
    # u = (x - 1.25) / 0.75 = (2x - 2.5) / 1.5
    ux = (2.0 * x - 2.5) / 1.5
    uy = (2.0 * y - 2.5) / 1.5
    uz = (2.0 * z - 2.5) / 1.5
    
    # Chebyshev polynomials of first kind
    # T0 = 1
    # T1 = u
    # T2 = 2u^2 - 1
    # T3 = 4u^3 - 3u
    
    # Basis functions:
    # We construct 10 basis terms.
    # We include main effects up to degree 2 and pairwise interactions of degree 1.
    # This allows capturing curvature and non-additive effects.
    
    t0x = 1.0
    t1x = ux
    t2x = 2.0 * ux * ux - 1.0
    
    t0y = 1.0
    t1y = uy
    t2y = 2.0 * uy * uy - 1.0
    
    t0z = 1.0
    t1z = uz
    t2z = 2.0 * uz * uz - 1.0
    
    # Linear combination of basis functions
    # p[0] * T0(x)T0(y)T0(z) + p[1] * T1(x)T0(y)T0(z) + p[2] * T0(x)T1(y)T0(z) + p[3] * T0(x)T0(y)T1(z)
    # + p[4] * T2(x)T0(y)T0(z) + p[5] * T0(x)T2(y)T0(z) + p[6] * T0(x)T0(y)T2(z)
    # + p[7] * T1(x)T1(y)T0(z) + p[8] * T1(x)T0(y)T1(z) + p[9] * T0(x)T1(y)T1(z)
    
    result = (
        p[0] * t0x * t0y * t0z +
        p[1] * t1x * t0y * t0z +
        p[2] * t0x * t1y * t0z +
        p[3] * t0x * t0y * t1z +
        p[4] * t2x * t0y * t0z +
        p[5] * t0x * t2y * t0z +
        p[6] * t0x * t0y * t2z +
        p[7] * t1x * t1y * t0z +
        p[8] * t1x * t0y * t1z +
        p[9] * t0x * t1y * t1z
    )
    
    return result
