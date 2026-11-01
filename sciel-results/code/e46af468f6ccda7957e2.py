import numpy as np
from scipy.linalg import eig
from scipy.integrate import quad
import time

# Define Probabilist's Hermite Polynomials
def He(n, x):
    """Compute the n-th probabilist's Hermite polynomial at x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3*x
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 12:
        return x**12 - 54*x**10 + 1260*x**8 - 12600*x**6 + 37800*x**4 - 30240*x**2 + 945
    else:
        # Fallback for other degrees if needed, though we only use 0, 4, 8, 12
        # Recurrence: He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)
        if n < 0:
            return 0
        h0 = 1.0
        h1 = x
        if n == 0: return h0
        if n == 1: return h1
        for k in range(1, n):
            h2 = x*h1 - k*h0
            h0 = h1
            h1 = h2
        return h1

def construct():
    start_time = time.time()
    deadline = start_time + 180.0 # Safety margin
    
    # Coefficients for H12 to ensure g(0)=0
    # g(0) = c0*He0(0) + c4*He4(0) + c8*He8(0) + c12*He12(0) = 0
    # He0(0)=1, He4(0)=3, He8(0)=105, He12(0)=945
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    
    # Basis functions phi_j(x) = He_j(x) + k_j * He_12(x)
    # where k_j is the coefficient of c_j in c12 expression, which is -He_j(0)/He_12(0)
    # So phi_j(x) = He_j(x) - (He_j(0)/He_12(0)) * He_12(x)
    
    k0 = -1.0 / 945.0
    k4 = -3.0 / 945.0
    k8 = -105.0 / 945.0
    
    def phi0(x):
        return He(0, x) + k0 * He(12, x)
    
    def phi4(x):
        return He(4, x) + k4 * He(12, x)
    
    def phi8(x):
        return He(8, x) + k8 * He(12, x)
    
    basis = [phi0, phi4, phi8]
    
    # Compute Matrix A (Norm) and B (Moment)
    # A_ij = integral exp(-x^2) phi_i(x) phi_j(x) dx
    # B_ij = integral x^2 exp(-x^2) phi_i(x) phi_j(x) dx
    
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(i, 3):
            # Integrate from -inf to inf
            # Using quad for accuracy
            integrand_A = lambda x: np.exp(-x**2) * basis[i](x) * basis[j](x)
            integrand_B = lambda x: x**2 * np.exp(-x**2) * basis[i](x) * basis[j](x)
            
            val_A, _ = quad(integrand_A, -np.inf, np.inf, limit=200)
            val_B, _ = quad(integrand_B, -np.inf, np.inf, limit=200)
            
            A[i, j] = val_A
            A[j, i] = val_A
            B[i, j] = val_B
            B[j, i] = val_B
            
            if time.time() > deadline - 10:
                # If running out of time, break and return best guess or seed
                # But quad is fast, this is unlikely
                break

    # Solve Generalized Eigenvalue Problem: B v = lambda A v
    # We want the smallest eigenvalue lambda
    eigenvalues, eigenvectors = eig(B, A)
    
    # Filter real eigenvalues
    real_eigenvalues = eigenvalues.real
    real_eigenvectors = eigenvectors.real
    
    # Find index of minimum real eigenvalue
    min_idx = np.argmin(real_eigenvalues)
    
    # The optimal coefficients are the eigenvector corresponding to min eigenvalue
    opt_vec = real_eigenvectors[:, min_idx]
    
    # Normalize? The problem doesn't specify normalization, but returning the vector is standard.
    # The seed returns specific numbers. I will return the vector as is.
    # Note: eigenvectors from scipy.linalg.eig are normalized such that the largest component has magnitude 1?
    # No, they are normalized to have unit Euclidean norm.
    
    return opt_vec
