import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space, eigh

def construct():
    # Indices of Hermite polynomials used
    K = np.array([0, 4, 8, 12])
    
    # Eigenvalues of x^2 and -d^2/dx^2 in the Hermite basis
    # For psi_n, <psi_n, x^2 psi_n> = n + 0.5
    # <psi_n, -d^2/dx^2 psi_n> = n + 0.5
    # Off-diagonal terms are zero for this specific subspace {0, 4, 8, 12}
    # because the operators connect n to n +/- 2, and the difference between
    # elements in K is 4.
    lambdas = K + 0.5
    
    # Coefficients for the constraint g(0) = 0
    # g(x) = sum c_k H_k(x) exp(-x^2/2)
    # g(0) = sum c_k H_k(0) = 0
    # c_k = d_k / sqrt(sqrt(pi) 2^k k!)
    # sum d_k * H_k(0) / sqrt(sqrt(pi) 2^k k!) = 0
    # Let L_k = H_k(0) / sqrt(2^k k!)
    # Then sum d_k L_k = 0 (ignoring the common factor 1/sqrt(sqrt(pi)))
    
    def factorial(n):
        return int(np.prod(np.arange(1, n+1))) if n > 0 else 1
    
    L = np.zeros(4)
    for i, k in enumerate(K):
        # H_k(0)
        if k == 0:
            H0 = 1
        elif k == 4:
            H0 = -12
        elif k == 8:
            H0 = 1680
        elif k == 12:
            H0 = -145152
        else:
            H0 = 0 # Should not happen
        
        L[i] = H0 / np.sqrt(2**k * factorial(k))
        
    # Find null space of L
    # We want d such that L @ d = 0
    U = null_space(L)
    
    # Project Lambda onto the null space
    # M = U^T diag(lambdas) U
    M = U.T @ np.diag(lambdas) @ U
    
    # Find smallest eigenvalue and eigenvector
    # The minimum of y^T M y subject to ||y||=1 is the smallest eigenvalue
    evals, evecs = eigh(M)
    
    # Smallest eigenvalue is evals[0], corresponding eigenvector is evecs[:, 0]
    y_opt = evecs[:, 0]
    
    # Recover d
    d_opt = U @ y_opt
    
    # Ensure d is normalized (it should be, but just in case)
    d_opt = d_opt / np.linalg.norm(d_opt)
    
    # Calculate c_k
    # c_k = d_k / sqrt(sqrt(pi) 2^k k!)
    # The normalization of g requires sum c_k^2 sqrt(pi) 2^k k! = 1
    # sum (d_k^2 / (sqrt(pi) 2^k k!)) * sqrt(pi) 2^k k! = sum d_k^2 = 1
    # So the formula is correct.
    
    c = np.zeros(4)
    for i, k in enumerate(K):
        c[i] = d_opt[i] / np.sqrt(np.sqrt(np.pi) * 2**k * factorial(k))
        
    # The function returns c_0, c_4, c_8
    # c[0] is c_0, c[1] is c_4, c[2] is c_8
    return np.array([c[0], c[1], c[2]])
