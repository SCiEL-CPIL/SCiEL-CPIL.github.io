"""Improved program: Analytical solution via generalized eigenvalue decomposition."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Hermite polynomial values at 0 (physicists' convention)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 1039500
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Variance (effective n) for each Hermite mode: <x^2> = n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    # Coefficients for the constraint c12 = -k * (c0*h0_0 + c4*h4_0 + c8*h8_0)
    k = 1.0 / h12_0
    a = np.array([h0_0, h4_0, h8_0])

    # Matrix A for the numerator of the variance
    # N = c^T D c + n12 * (k * a^T c)^2 = c^T (D + n12 * k^2 * a a^T) c
    D = np.diag([n0, n4, n8])
    A = D + n12 * (k**2) * np.outer(a, a)

    # Matrix B for the denominator of the variance
    # M = c^T I c + (k * a^T c)^2 = c^T (I + k^2 * a a^T) c
    I = np.eye(3)
    B = I + (k**2) * np.outer(a, a)

    # Solve the generalized eigenvalue problem A v = lambda B v
    # The minimum variance is the smallest eigenvalue
    try:
        eigvals, eigvecs = np.linalg.eigh(A, B)
        # eigh returns eigenvalues in ascending order
        idx_min = np.argmin(eigvals)
        v_min = eigvecs[:, idx_min]
        
        # The objective is scale-invariant. We can return the eigenvector directly.
        # To match the "style" of the seed (non-trivial values), we might normalize it.
        # However, any non-zero scalar multiple is valid.
        # Let's normalize it so that the norm is 1, or just return it.
        # The seed had c0 ~ -1.13.
        # Let's check if the sign matters. No, squares are used.
        # We will return the eigenvector.
        
        # Optional: Scale to match seed magnitude roughly if needed, but not required.
        # Let's just return the eigenvector.
        return np.array(v_min)
    except Exception:
        # Fallback to seed if linear algebra fails (should not happen)
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
