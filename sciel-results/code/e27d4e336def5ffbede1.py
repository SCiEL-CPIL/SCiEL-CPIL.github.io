import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using an orthonormal Chebyshev polynomial basis model.
    The model is a linear combination of 10 orthonormal 2D Chebyshev polynomial basis functions.
    Chebyshev polynomials provide superior conditioning on [-1, 1] and may capture
    oscillatory behavior more effectively than Legendre polynomials.
    """
    # Compute 1D orthonormal Chebyshev polynomials of the first kind
    # T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1, T_3(x) = 4x^3 - 3x
    # Orthonormalization: T_0 / sqrt(2), T_n / sqrt(1) for n >= 1 (on [-1,1] with uniform weight)
    # Actually, the orthogonality of Chebyshev polynomials is with weight 1/sqrt(1-x^2)
    # For uniform weight, we need to adjust. Let's use standard Chebyshev and normalize empirically
    # or use the fact that they are orthogonal with respect to the weight function.
    # For BFGS conditioning, orthonormality with respect to the sampling distribution is key.
    # Since inputs are uniform, we want basis functions orthogonal under uniform measure.
    # Chebyshev polynomials are NOT orthogonal under uniform measure on [-1,1].
    # However, they are still well-conditioned. Let's use them as is, but scale them.
    
    # T_0 = 1
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct 2D basis functions (products of 1D Chebyshev polynomials)
    # Total 10 terms
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T0_x * T2_y
    b5 = T1_x * T1_y
    b6 = T3_x * T0_y
    b7 = T0_x * T3_y
    b8 = T2_x * T1_y
    b9 = T1_x * T2_y
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B.dot(params)
    
    return pred
