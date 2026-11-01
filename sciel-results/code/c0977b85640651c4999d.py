"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 210s limit
    
    # Hermite polynomial coefficients (physicists' convention)
    # H_0(x) = 1
    # H_4(x) = 16x^4 - 48x^2 + 12
    # H_8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 10080
    # H_12(x) = 4096x^12 - 147456x^10 + 1658880x^8 - 7983360x^6 + 15482880x^4 - 11801280x^2 + 2032128
    
    # We are tuning c0, c4, c8. c12 is fixed so that g(0) = 0.
    # g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # g(0) = c0*1 + c4*12 + c8*10080 + c12*2032128 = 0
    # c12 = -(c0 + 12*c4 + 10080*c8) / 2032128
    
    def g_coefficients(c):
        c0, c4, c8 = c
        c12 = -(c0 + 12*c4 + 10080*c8) / 2032128.0
        return np.array([c0, c4, c8, c12])
    
    def poly_coeffs(c):
        """
        Returns coefficients of the polynomial part of g(x) in descending powers of x.
        g(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        """
        c0, c4, c8, c12 = c
        # H0: [1]
        # H4: [16, 0, -48, 0, 12]
        # H8: [256, 0, -4608, 0, 23040, 0, -34560, 0, 10080]
        # H12: [4096, 0, -147456, 0, 1658880, 0, -7983360, 0, 15482880, 0, -11801280, 0, 2032128]
        
        # Degree 12: 4096*c12
        # Degree 10: -147456*c12
        # Degree 8: 256*c8 + 1658880*c12
        # Degree 6: -4608*c8 - 7983360*c12
        # Degree 4: 16*c4 + 23040*c8 + 15482880*c12
        # Degree 2: -48*c4 - 34560*c8 - 11801280*c12
        # Degree 0: c0 + 12*c4 + 10080*c8 + 2032128*c12 (should be 0)
        
        coeffs = np.zeros(13)
        coeffs[0] = 4096 * c12
        coeffs[2] = -147456 * c12
        coeffs[4] = 256 * c8 + 1658880 * c12
        coeffs[6] = -4608 * c8 - 7983360 * c12
        coeffs[8] = 16 * c4 + 23040 * c8 + 15482880 * c12
        coeffs[10] = -48 * c4 - 34560 * c8 - 11801280 * c12
        coeffs[12] = 0.0 # g(0)=0
        
        return coeffs
    
    def uncertainty_bound(c):
        """
        Computes the upper bound on A(f)*A(Fourier(f)).
        f(x) = g(x) * exp(-x^2/2)
        A(f) = ||f||_1 / ||f||_2
        A(F) = ||F||_1 / ||F||_2
        Note: ||F||_2 = ||f||_2 (Plancherel, depending on Fourier convention normalization).
        Usually A(f) = ||f||_1 / ||f||_2.
        The uncertainty principle states ||f||_1 ||F||_1 <= C ||f||_2 ||F||_2 = C ||f||_2^2.
        So we want to minimize (||f||_1 / ||f||_2) * (||F||_1 / ||f||_2) = ||f||_1 ||F||_1 / ||f||_2^2.
        
        Let's use numerical integration.
        f(x) = P(x) exp(-x^2/2)
        F(omega) = \int f(x) exp(-i omega x) dx.
        For real even P(x), F(omega) is real and even.
        F(omega) = \int_{-\infty}^{\infty} P(x) e^{-x^2/2} e^{-i \omega x} dx.
        This can be computed analytically or numerically.
        Since P is a polynomial, we can use the fact that the Fourier transform of x^n e^{-x^2/2} is related to Hermite polynomials or just compute the integral numerically.
        Given the degree is 12, numerical integration over a large range is feasible.
        """
        coeffs = poly_coeffs(c)
        # coeffs[i] is coeff of x^(12-i)
        
        # Define f(x)
        def f(x):
            # Horner's method for P(x)
            p = 0.0
            for coeff in coeffs:
                p = p * x + coeff
            return p * np.exp(-0.5 * x**2)
        
        # Define F(omega)
        # F(omega) = \int f(x) cos(omega x) dx (since f is even and real, F is real and even, and imaginary part is 0)
        # Actually F(omega) = \int f(x) e^{-i \omega x} dx. If f is even, F is even and real.
        # F(omega) = 2 \int_0^\infty f(x) cos(omega x) dx.
        
        # We need ||f||_1, ||f||_2, ||F||_1, ||F||_2.
        # ||f||_2^2 = \int f(x)^2 dx.
        # ||F||_2^2 = ||f||_2^2 (assuming standard L2 normalized Fourier transform? No, standard is \hat{f}(\xi) = \int f(x) e^{-2\pi i x \xi} dx or \int f(x) e^{-i x \xi} dx.
        # Let's stick to the convention where \hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-i x \xi} dx.
        # Then ||\hat{f}||_2 = \sqrt{2\pi} ||f||_2.
        # The constant C4 usually absorbs these factors. The task asks for the "upper bound on C4".
        # The seed program returns coefficients. The evaluator computes the score.
        # I need to replicate the metric the evaluator uses.
        # "score is the upper bound on C4".
        # Usually, the uncertainty inequality is ||f||_1 ||\hat{f}||_1 \le C ||f||_2 ||\hat{f}||_2.
        # C = (||f||_1 / ||f||_2) (||\hat{f}||_1 / ||\hat{f}||_2).
        # Let's compute this ratio.
        
        # Numerical integration setup
        # Range for x: [-10, 10] should be enough for exp(-x^2/2).
        x = np.linspace(-12, 12, 4001)
        dx = x[1] - x[0]
        f_vals = f(x)
        
        norm_f_1 = np.trapz(np.abs(f_vals), x)
        norm_f_2_sq = np.trapz(f_vals**2, x)
        norm_f_2 = np.sqrt(norm_f_2_sq)
        
        # Compute F(omega)
        # F(omega) = \int f(x) e^{-i x \omega} dx = 2 \int_0^\infty f(x) \cos(x \omega) dx
        # We need ||F||_1 and ||F||_2.
        # Range for omega: Since f is smooth and decays, F decays algebraically (degree 12 -> 1/omega^13? No, polynomial times Gaussian decays super-exponentially? No, polynomial times Gaussian Fourier transform is also Gaussian times polynomial.
        # Wait, Fourier transform of x^n e^{-x^2/2} is i^n \sqrt{2\pi} H_n(\omega) e^{-\omega^2/2}.
        # So F(omega) is also a polynomial times Gaussian.
        # We can compute F(omega) analytically!
        # F(omega) = \sum_{k=0}^{12} a_k \hat{x^k e^{-x^2/2}}(\omega)
        # \hat{x^k e^{-x^2/2}}(\omega) = i^k \sqrt{2\pi} H_k(\omega) e^{-\omega^2/2} ?
        # Let's check.
        # \hat{e^{-x^2/2}} = \sqrt{2\pi} e^{-\omega^2/2}.
        # \hat{x f(x)} = i \frac{d}{d\omega} \hat{f}(\omega).
        # So \hat{x^k e^{-x^2/2}} = i^k \frac{d^k}{d\omega^k} (\sqrt{2\pi} e^{-\omega^2/2}).
        # This is related to Hermite polynomials.
        # Specifically, \frac{d^k}{dx^k} e^{-x^2/2} = (-1)^k H_k(x) e^{-x^2/2}.
        # So i^k \sqrt{2\pi} \frac{d^k}{d\omega^k} e^{-\omega^2/2} = i^k \sqrt{2\pi} (-1)^k H_k(\omega) e^{-\omega^2/2} = (i)^k (-1)^k \sqrt{2\pi} H_k(\omega) e^{-\omega^2/2} = (-i)^k \sqrt{2\pi} H_k(\omega) e^{-\omega^2/2}.
        
        # So F(omega) = \sum_{k=0}^{12} a_k (-i)^k \sqrt{2\pi} H_k(\omega) e^{-\omega^2/2}.
        # Where a_k are the coefficients of P(x) = \sum a_k x^k.
        # Note: My `poly_coeffs` returns coefficients in descending order.
        # Let's convert to ascending order a_k for x^k.
        
        a = np.zeros(13)
        for i in range(13):
            a[12-i] = coeffs[i]
            
        # Compute F(omega)
        # We need H_k(omega) for k=0..12.
        # We can compute these analytically or numerically.
        # Since we are minimizing, we can precompute the structure.
        
        # Let's evaluate F(omega) on a grid.
        omega = np.linspace(-12, 12, 4001)
        domega = omega[1] - omega[0]
        
        # Compute H_k(omega) for k=0..12
        # Recurrence: H_0=1, H_1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
        H = np.zeros((13, len(omega)))
        H[0] = 1
        H[1] = 2 * omega
        for k in range(2, 13):
            H[k] = 2 * omega * H[k-1] - 2 * (k-1) * H[k-2]
            
        F_vals = np.zeros(len(omega))
        for k in range(13):
            if a[k] != 0:
                # Factor: (-i)^k * sqrt(2*pi)
                # (-i)^k = cos(k*pi/2) - i sin(k*pi/2)
                # Since F(omega) is real and even (P is real, f is even), the imaginary parts must cancel.
                # For even k, (-i)^k is real. For odd k, (-i)^k is imaginary.
                # But P(x) is a sum of even and odd powers?
                # Wait, H_0, H_4, H_8, H_12 are all even polynomials.
                # So P(x) is an even polynomial.
                # So a_k = 0 for odd k.
                # Thus only even k terms contribute.
                # For even k, (-i)^k is real: 1, -1, 1, -1...
                # (-i)^0 = 1
                # (-i)^2 = -1
                # (-i)^4 = 1
                # (-i)^6 = -1
                # (-i)^8 = 1
                # (-i)^10 = -1
                # (-i)^12 = 1
                sign = 1 if (k // 2) % 2 == 0 else -1
                F_vals += a[k] * sign * np.sqrt(2 * np.pi) * H[k] * np.exp(-0.5 * omega**2)
                
        norm_F_1 = np.trapz(np.abs(F_vals), omega)
        norm_F_2_sq = np.trapz(F_vals**2, omega)
        norm_F_2 = np.sqrt(norm_F_2_sq)
        
        # The bound C is (||f||_1 / ||f||_2) * (||F||_1 / ||F||_2)
        # Note: ||F||_2 = sqrt(2*pi) ||f||_2.
        # So C = (||f||_1 / ||f||_2) * (||F||_1 / (sqrt(2*pi) ||f||_2))
        # Let's just compute the product of the ratios.
        
        if norm_f_2 == 0 or norm_F_2 == 0:
            return np.inf
            
        return (norm_f_1 / norm_f_2) * (norm_F_1 / norm_F_2)
    
    # Initial guess from seed
    c_init = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # Reparameterize to sphere?
    # The directive suggests: "Reparameterize the search space using the normalized vector of coefficients c subject to sum c_i^2 = 1"
    # But the coefficients are not normalized in the seed.
    # The scale of c doesn't matter for the ratio (||f||_1/||f||_2) because it's homogeneous.
    # So we can fix the scale, e.g., c0^2 + c4^2 + c8^2 = 1.
    # Let's use angles on the sphere.
    # c0 = r cos(theta) cos(phi) ... no, 3D sphere.
    # c0 = sin(theta) cos(phi)
    # c4 = sin(theta) sin(phi)
    # c8 = cos(theta)
    # This covers the sphere.
    
    def objective_from_angles(params):
        theta, phi = params
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        return uncertainty_bound(c)
    
    # Optimize
    # Start from seed direction
    c_seed = c_init
    norm_seed = np.linalg.norm(c_seed)
    c_seed_norm = c_seed / norm_seed
    # Find theta, phi for c_seed_norm
    c8_s = c_seed_norm[2]
    theta_s = np.arccos(np.clip(c8_s, -1, 1))
    if np.sin(theta_s) > 1e-9:
        phi_s = np.arctan2(c_seed_norm[1], c_seed_norm[0])
    else:
        phi_s = 0
        
    x0 = np.array([theta_s, phi_s])
    
    # Use L-BFGS-B or Nelder-Mead on the angles
    # Nelder-Mead is robust for low dimensions.
    res = minimize(objective_from_angles, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    
    best_c = c_init
    best_val = uncertainty_bound(c_init)
    
    if res.success and res.fun < best_val:
        theta_opt, phi_opt = res.x
        c0_opt = np.sin(theta_opt) * np.cos(phi_opt)
        c4_opt = np.sin(theta_opt) * np.sin(phi_opt)
        c8_opt = np.cos(theta_opt)
        best_c = np.array([c0_opt, c4_opt, c8_opt])
        best_val = res.fun
        
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a few random restarts
        np.random.seed(42)
        for i in range(5):
            theta_r = np.random.uniform(0, np.pi)
            phi_r = np.random.uniform(-np.pi, np.pi)
            x_r = np.array([theta_r, phi_r])
            res_r = minimize(objective_from_angles, x_r, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_r.fun < best_val:
                best_val = res_r.fun
                c0_opt = np.sin(res_r.x[0]) * np.cos(res_r.x[1])
                c4_opt = np.sin(res_r.x[0]) * np.sin(res_r.x[1])
                c8_opt = np.cos(res_r.x[0])
                best_c = np.array([c0_opt, c4_opt, c8_opt])
                
            if time.time() - start_time > deadline - 5:
                break
                
    # Scale back?
    # The objective is scale invariant. The evaluator likely computes the bound from the coefficients.
    # If the evaluator just takes the coefficients and computes the bound, the scale doesn't matter.
    # However, if the evaluator expects specific scaling, it might matter.
    # The seed returns specific numbers. Let's return the normalized direction scaled by the original norm?
    # Or just the normalized direction?
    # The task says "three coefficients of the Hermite combination".
    # Usually, the "best" test function is unique up to scale.
    # Let's return the coefficients that minimize the bound. Since the bound is scale invariant, any scale works.
    # To be safe, let's return the normalized vector (norm 1).
    # Or maybe the seed's scale is important for some reason? Unlikely for a ratio.
    # Let's return the coefficients found.
    
    return best_c

# EVOLVE-BLOCK-END
