import numpy as np
from scipy.linalg import eigh

def construct():
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_0(x) = 1
    # H_4(x) = x^4 - 6x^2 + 3
    # H_8(x) = x^8 - 28x^6 + 420x^4 - 1680x^2 + 105
    # H_12(x) = x^12 - 84x^10 + 2520x^8 - 27720x^6 + 83160x^4 - 55440x^2 + 10395
    
    # We need integrals of form int x^2 H_i(x) H_j(x) e^{-x^2} dx and int H_i(x) H_j(x) e^{-x^2} dx
    # Using Gauss-Hermite quadrature for high precision
    # The integrals are exact for polynomials of degree up to 2*n-1.
    # Max degree in product is 12+12+2 = 26.
    # We need n such that 2n-1 >= 26 => n >= 13.5 => n=14.
    # Let's use n=50 to be safe and fast.
    x, w = np.polynomial.hermite.hermgauss(50)
    
    # Evaluate H_k at nodes
    H = np.zeros((4, 50))
    H[0, :] = 1
    H[1, :] = x**4 - 6*x**2 + 3
    H[2, :] = x**8 - 28*x**6 + 420*x**4 - 1680*x**2 + 105
    H[3, :] = x**12 - 84*x**10 + 2520*x**8 - 27720*x**6 + 83160*x**4 - 55440*x**2 + 10395
    
    # Compute M_ij = int x^2 H_i H_j e^{-x^2} dx
    # Compute N_ij = int H_i H_j e^{-x^2} dx
    M = np.zeros((4, 4))
    N = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
            # int x^2 H_i H_j e^{-x^2} dx approx sum w_k x_k^2 H_i(x_k) H_j(x_k)
            M[i, j] = np.sum(w * x**2 * H[i, :] * H[j, :])
            N[i, j] = np.sum(w * H[i, :] * H[j, :])
            
    # Constraint: g(0) = 0
    # c0 H0(0) + c4 H4(0) + c8 H8(0) + c12 H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    # c0 + 3c4 + 105c8 + 10395c12 = 0  (Wait, H4(0)=3? Let's check)
    # H4(x) = x^4 - 6x^2 + 3. H4(0) = 3.
    # H8(x) ... + 105. H8(0) = 105.
    # H12(x) ... + 10395. H12(0) = 10395.
    # So c12 = -(c0 + 3c4 + 105c8) / 10395
    
    # Note: The seed program used H_n from a specific convention?
    # The seed coefficients were -1.13, 0.04, 0.0003.
    # Let's check if my H_n signs match.
    # Standard probabilists' vs physicists'?
    # The Gaussian is e^{-x^2/2}. The weight in integral is e^{-x^2}.
    # This corresponds to physicists' Hermite polynomials $H_n$ with $e^{-x^2}$.
    # $H_4(0) = 3$.
    # In the seed code comment: "Goncalves et al. 2017".
    # Let's assume standard physicists' Hermite.
    
    # Matrix L mapping u=[c0, c4, c8] to c=[c0, c4, c8, c12]
    L = np.zeros((4, 3))
    L[0, 0] = 1
    L[1, 1] = 1
    L[2, 2] = 1
    L[3, 0] = -1 / 10395
    L[3, 1] = -3 / 10395
    L[3, 2] = -105 / 10395
    
    # A = L^T M L
    # B = L^T N L
    A = L.T @ M @ L
    B = L.T @ N @ L
    
    # Solve generalized eigenvalue problem A v = lambda B v
    # We want the smallest eigenvalue.
    vals, vecs = eigh(A, B)
    
    # The optimal coefficients are the eigenvector corresponding to the smallest eigenvalue
    # The value of the objective is vals[0]
    # We return the coefficients u
    u_opt = vecs[:, 0]
    
    # Ensure the function is not zero (it won't be)
    # Return the three coefficients
    return np.array(u_opt)
