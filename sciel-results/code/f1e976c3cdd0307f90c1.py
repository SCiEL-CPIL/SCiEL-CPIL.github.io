"""
Strategy: Chebyshev Polynomial Basis (Degree 2 Tensor Product)

The previous approach used a rational function which can suffer from instability
near poles and may not efficiently use the 10-parameter budget for smooth functions.
This strategy uses a basis of Chebyshev polynomials of the first kind, T_n(t).

1. Map inputs x, y, z from [0.5, 2] to t in [-1, 1].
   t = (2*v - (min+max)) / (max-min) = (2*v - 2.5) / 1.5 = (4*v - 5) / 3

2. Compute Chebyshev polynomials up to degree 2 for each input:
   T_0(t) = 1
   T_1(t) = t
   T_2(t) = 2*t^2 - 1

3. Construct the 10-dimensional basis using a tensor product of these polynomials:
   The basis vectors are:
   1. T_0(x)T_0(y)T_0(z) = 1
   2. T_1(x)T_0(y)T_0(z) = t_x
   3. T_0(x)T_1(y)T_0(z) = t_y
   4. T_0(x)T_0(y)T_1(z) = t_z
   5. T_2(x)T_0(y)T_0(z) = 2*t_x^2 - 1
   6. T_0(x)T_2(y)T_0(z) = 2*t_y^2 - 1
   7. T_0(x)T_0(y)T_2(z) = 2*t_z^2 - 1
   8. T_1(x)T_1(y)T_0(z) = t_x * t_y
   9. T_1(x)T_0(y)T_1(z) = t_x * t_z
   10. T_0(x)T_1(y)T_1(z) = t_y * t_z

This basis is:
- Linear in the parameters (coefficients), so BFGS will converge in 1 step (or very close).
- Orthogonal (in the continuous sense) over the domain [-1, 1], leading to a well-conditioned
  Gram matrix. This avoids the numerical instability of monomial bases (like x^2, x^3) or
  rational functions which can have ill-conditioned Hessians.
- The minimax property of Chebyshev polynomials ensures that the basis functions are
  well-distributed across the domain, providing good approximation properties for
  smooth functions.
"""
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map inputs from [0.5, 2] to [-1, 1]
    # t = (4*v - 5) / 3
    tx = (4.0 * x - 5.0) / 3.0
    ty = (4.0 * y - 5.0) / 3.0
    tz = (4.0 * z - 5.0) / 3.0
    
    # Chebyshev polynomials of the first kind up to degree 2
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    
    t0x = 1.0
    t1x = tx
    t2x = 2.0 * tx * tx - 1.0
    
    t0y = 1.0
    t1y = ty
    t2y = 2.0 * ty * ty - 1.0
    
    t0z = 1.0
    t1z = tz
    t2z = 2.0 * tz * tz - 1.0
    
    # Construct the 10 basis functions
    # The order matches the params array
    # p0 * B0 + p1 * B1 + ... + p9 * B9
    
    B0 = t0x * t0y * t0z
    B1 = t1x * t0y * t0z
    B2 = t0x * t1y * t0z
    B3 = t0x * t0y * t1z
    B4 = t2x * t0y * t0z
    B5 = t0x * t2y * t0z
    B6 = t0x * t0y * t2z
    B7 = t1x * t1y * t0z
    B8 = t1x * t0y * t1z
    B9 = t0x * t1y * t1z
    
    # Linear combination
    pred = (params[0] * B0 + 
            params[1] * B1 + 
            params[2] * B2 + 
            params[3] * B3 + 
            params[4] * B4 + 
            params[5] * B5 + 
            params[6] * B6 + 
            params[7] * B7 + 
            params[8] * B8 + 
            params[9] * B9)
            
    return pred
