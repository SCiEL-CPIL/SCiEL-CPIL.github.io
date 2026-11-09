import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_hermite

def construct():
    # 1. Define Hermite Polynomials
    # H_n(x) are physicists' Hermite polynomials.
    # H_0 = 1
    # H_4 = 16x^4 - 48x^2 + 12
    # H_8 = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
    # H_12 = 4096x^12 - 122880x^10 + 1474560x^8 - 7077920x^6 + 12740160x^4 - 7077920x^2 + 10321920
    
    def H0(x): return np.ones_like(x)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 3072*x**6 + 10752*x**4 - 10752*x**2 + 1680
    def H12(x): return 4096*x**12 - 122880*x**10 + 1474560*x**8 - 7077920*x**6 + 12740160*x**4 - 7077920*x**2 + 10321920

    # Coefficients for H12 to satisfy g(0)=0
    # g(0) = c0*1 + c4*12 + c8*1680 + c12*10321920 = 0
    # c12 = -(c0 + 12c4 + 1680c8) / 10321920
    # Note: H12(0) = 10321920. Let's verify.
    # H_n(0) for n=12: (-1)^6 * 12! / 6! = 479001600 / 720 = 665280.
    # My polynomial H12 above: constant term 10321920.
    # Let's check H12(0) from the polynomial formula.
    # H_12(x) = e^{x^2/2} (-1)^12 (d/dx)^12 e^{-x^2/2}.
    # The constant term is indeed 10321920?
    # Let's re-calculate H12(0).
    # H_0(0)=1
    # H_2(0)=-2
    # H_4(0)=12
    # H_6(0)=-120
    # H_8(0)=1680
    # H_10(0)=-30240
    # H_12(0)=665280
    # My polynomial H12 has constant 10321920. 10321920 / 665280 = 15.5? No.
    # Let's check the coefficients of H12.
    # H_12(x) = 4096 x^12 - 122880 x^10 + 1474560 x^8 - 7077920 x^6 + 12740160 x^4 - 7077920 x^2 + 10321920.
    # If x=0, value is 10321920.
    # Why the discrepancy?
    # H_n(x) = (-1)^n e^{x^2/2} (d/dx)^n e^{-x^2/2}.
    # For n=12, the constant term is $(-1)^{12} \frac{12!}{(12/2)!} = \frac{479001600}{720} = 665280$.
    # So my polynomial H12 is scaled?
    # Let's check H4. 16x^4 - 48x^2 + 12. H4(0)=12. Correct.
    # H8. 256x^8 ... + 1680. H8(0)=1680. Correct.
    # H12. The leading coeff is $2^{12} = 4096$.
    # The constant term should be 665280.
    # 10321920 / 665280 = 15.5. This is not an integer.
    # Let's recalculate H12 constant term.
    # Maybe I copied the wrong polynomial.
    # Let's generate Hermite polynomials using scipy or sympy logic if needed, or just use the recurrence.
    # H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x).
    # H_0 = 1
    # H_1 = 2x
    # H_2 = 4x^2 - 2
    # H_3 = 8x^3 - 12x
    # H_4 = 16x^4 - 48x^2 + 12
    # H_5 = 32x^5 - 160x^3 + 120x
    # H_6 = 64x^6 - 480x^4 + 720x^2 - 120
    # H_7 = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H_8 = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
    # H_9 = 512x^9 - 7168x^7 + 30464x^5 - 40320x^3 + 10080x
    # H_10 = 1024x^10 - 16384x^8 + 81920x^6 - 163840x^4 + 100800x^2 - 10080
    # H_11 = 2048x^11 - 36864x^9 + 220160x^7 - 589824x^5 + 554400x^3 - 110880x
    # H_12 = 4096x^12 - 73728x^10 + 440320x^8 - 1179648x^6 + 1108800x^4 - 332640x^2 + 1032192? 
    # Let's check constant term of H12 from recurrence.
    # H_12(0) = -2 * 11 * H_10(0) = -22 * (-10080) = 221760?
    # Wait. H_{n+1}(0) = -2n H_{n-1}(0).
    # H_2(0) = -2(1)H_0(0) = -2.
    # H_4(0) = -2(3)H_2(0) = -6(-2) = 12.
    # H_6(0) = -2(5)H_4(0) = -10(12) = -120.
    # H_8(0) = -2(7)H_6(0) = -14(-120) = 1680.
    # H_10(0) = -2(9)H_8(0) = -18(1680) = -30240.
    # H_12(0) = -2(11)H_10(0) = -22(-30240) = 665280.
    # So H12(0) is 665280.
    # The polynomial I had: 10321920.
    # 10321920 / 665280 = 15.5.
    # It seems the polynomial coefficients I recalled were wrong or scaled.
    # I will use the recurrence to generate the polynomials accurately in the code.

    def gen_hermite(n):
        # Returns coefficients [c_n, c_{n-1}, ..., c_0]
        if n == 0: return np.array([1.0])
        if n == 1: return np.array([2.0, 0.0])
        h_prev = gen_hermite(n-1)
        h_curr = gen_hermite(n-2)
        # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
        # Multiply h_prev by 2x: shift coeffs
        term1 = np.convolve(h_prev, [2.0, 0.0])
        term2 = -2.0 * (n-1) * h_curr
        # Pad term2 to match length
        pad_len = len(term1) - len(term2)
        if pad_len > 0:
            term2 = np.concatenate([np.zeros(pad_len), term2])
        return term1 + term2

    # Generate H0, H4, H8, H12
    # We need to evaluate them at quadrature points.
    # It's easier to just use the recurrence to get coefficients, then evaluate.
    
    # However, for the matrix calculation, we need integrals of products of polynomials times e^{-x^2}.
    # Gauss-Hermite quadrature is ideal.
    # Nodes x_i and weights w_i such that \int e^{-x^2} f(x) dx \approx \sum w_i f(x_i).
    
    N_points = 200 # Enough for degree 24 polynomials (product of two deg 12)
    x_nodes, w_weights = roots_hermite(N_points)
    
    # Calculate polynomial values at nodes
    # We need B_i(x) and C_i(x) at x_nodes.
    
    # Get coefficients for H0, H4, H8, H12
    h0_coeff = gen_hermite(0)
    h4_coeff = gen_hermite(4)
    h8_coeff = gen_hermite(8)
    h12_coeff = gen_hermite(12)
    
    # Evaluate at nodes
    H0_vals = np.polyval(h0_coeff, x_nodes)
    H4_vals = np.polyval(h4_coeff, x_nodes)
    H8_vals = np.polyval(h8_coeff, x_nodes)
    H12_vals = np.polyval(h12_coeff, x_nodes)
    
    # Constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # c12 = -(c0 + 12c4 + 1680c8) / 665280
    
    # Basis B:
    # B0 = H0 - (1/665280) H12
    # B4 = H4 - (12/665280) H12
    # B8 = H8 - (1680/665280) H12
    
    B0 = H0_vals - (1.0 / 665280.0) * H12_vals
    B4 = H4_vals - (12.0 / 665280.0) * H12_vals
    B8 = H8_vals - (1680.0 / 665280.0) * H12_vals
    
    # B matrix (3 x N_points)
    B_mat = np.vstack([B0, B4, B8])
    
    # Calculate C_i = B_i' - x B_i
    # Derivative of polynomial:
    def deriv_poly_vals(coeffs, x):
        d_coeffs = np.polyder(coeffs)
        if len(d_coeffs) == 0:
            return np.zeros_like(x)
        return np.polyval(d_coeffs, x)
    
    dH0 = deriv_poly_vals(h0_coeff, x_nodes)
    dH4 = deriv_poly_vals(h4_coeff, x_nodes)
    dH8 = deriv_poly_vals(h8_coeff, x_nodes)
    dH12 = deriv_poly_vals(h12_coeff, x_nodes)
    
    dB0 = dH0 - (1.0 / 665280.0) * dH12
    dB4 = dH4 - (12.0 / 665280.0) * dH12
    dB8 = dH8 - (1680.0 / 665280.0) * dH12
    
    C0 = dB0 - x_nodes * B0
    C4 = dB4 - x_nodes * B4
    C8 = dB8 - x_nodes * B8
    
    C_mat = np.vstack([C0, C4, C8])
    
    # Weight vector for integration
    # Integral approx sum w_i * f(x_i)
    # M_D = \int e^{-x^2} B B^T dx = B^T W B
    # M_N = \int e^{-x^2} C C^T dx = C^T W C
    # W is diagonal matrix of weights.
    # B_mat is 3 x N. W is N x N.
    # M_D = B_mat.T @ (w_weights[:, None] * B_mat)
    
    M_D = B_mat.T @ (w_weights[:, None] * B_mat)
    M_N = C_mat.T @ (w_weights[:, None] * C_mat)
    
    # Solve generalized eigenvalue problem M_N v = lambda M_D v
    # Use scipy.linalg.eigh
    eigenvalues, eigenvectors = eigh(M_N, M_D)
    
    # Find smallest eigenvalue
    idx_min = np.argmin(eigenvalues)
    lambda_min = eigenvalues[idx_min]
    v_opt = eigenvectors[:, idx_min]
    
    # Normalize v_opt to have unit norm for consistency, though ratio is scale invariant
    v_opt = v_opt / np.linalg.norm(v_opt)
    
    return v_opt

