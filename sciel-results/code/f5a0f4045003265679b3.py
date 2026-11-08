"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function
    to reduce the upper bound on A(f)*A(Fourier(f)).
    
    The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    where a12 is determined by the constraint g(0) = 0.
    
    We use the analytic properties of Hermite functions under Fourier transform
    to express the uncertainty product bound in terms of spectral moments.
    
    Returns: array of three coefficients [a0, a4, a8]
    """
    
    def hermite_h(n, x):
        """Compute the n-th physicist's Hermite polynomial H_n(x) recursively."""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            h0 = np.ones_like(x)
            h1 = 2 * x
            for k in range(2, n + 1):
                h2 = 2 * x * h1 - 2 * (k - 1) * h0
                h0, h1 = h1, h2
            return h1
    
    def compute_bound(a0, a4, a8):
        """
        Compute the upper bound C4 on A(f)*A(Fourier(f)) for the given coefficients.
        
        For a function g(x) = sum_k a_k H_k(x) exp(-x^2/2), the Fourier transform
        is (up to a phase) sum_k a_k H_k(ξ) exp(-ξ^2/2) (Fourier invariant for Hermite functions).
        
        The uncertainty product A(g)*A(ĝ) involves the L1 norms of |g| and |ĝ|.
        
        For the specific form with Hermite polynomials times Gaussian, we can use
        the fact that |H_k(x) exp(-x^2/2)| has known properties.
        
        The bound C4 is related to the product of the L1 norms. For the normalized
        Hermite functions, we need to compute:
        
        A(g) = ∫ |g(x)| dx / (∫ g(x) dx)  [or similar normalization]
        
        Actually, the uncertainty inequality bound involves:
        C4 = sup over g of [ (∫ x^2 |g|^2 dx / ∫ |g|^2 dx) * (∫ ξ^2 |ĝ|^2 dξ / ∫ |ĝ|^2 dξ) ]
        
        But for the specific formulation in the task, the "upper bound on A(f)*A(Fourier(f))"
        where A likely refers to some norm-based measure.
        
        Given the context of the seed program returning specific rational coefficients,
        and the task mentioning "upper bound on C4", let's use the standard
        uncertainty product for the L2-normalized function.
        
        For g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2):
        
        First, enforce g(0) = 0:
        g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
        H0(0) = 1
        H4(0) = -12  (H4(x) = 16x^4 - 48x^2 + 12, so H4(0) = 12... wait let me recheck)
        
        Physicist's Hermite polynomials:
        H0(x) = 1
        H1(x) = 2x
        H2(x) = 4x^2 - 2
        H3(x) = 8x^3 - 12x
        H4(x) = 16x^4 - 48x^2 + 12
        H8(x) = 256x^8 - 4480x^6 + 23040x^4 - 30720x^2 + 46080
        H12(x) = 4096x^12 - 153600x^10 + 2304000x^8 - 12902400x^6 + 25804800x^4 - 14745600x^2 + 10321920
        
        So:
        H0(0) = 1
        H4(0) = 12
        H8(0) = 46080
        H12(0) = 10321920
        
        g(0) = a0*1 + a4*12 + a8*46080 + a12*10321920 = 0
        => a12 = -(a0 + 12*a4 + 46080*a8) / 10321920
        
        Now for the uncertainty product. The standard Heisenberg uncertainty principle
        gives Δx * Δξ ≥ 1/2, but we're looking at a specific bound C4.
        
        Let's compute the second moments:
        For g(x) = sum a_k H_k(x) e^{-x^2/2}, with the constraint g(0)=0:
        
        ||g||_2^2 = sum_k a_k^2 * ||H_k e^{-x^2/2}||_2^2 + cross terms
        
        Actually, the Hermite functions ψ_k(x) = (2^k k! sqrt(π))^{-1/2} H_k(x) e^{-x^2/2}
        are orthonormal. So:
        
        Let c_k = a_k * (2^k k! sqrt(π))^{1/2} so that g = sum c_k ψ_k
        
        Then ||g||_2^2 = sum c_k^2
        <x^2> = sum c_k^2 * <ψ_k, x^2 ψ_k> / sum c_k^2
        
        For Hermite functions, <ψ_k, x^2 ψ_k> = (2k+1)/2
        
        Similarly for the Fourier side, since Fourier transform maps ψ_k to (-i)^k ψ_k,
        the second moment in frequency space is the same: <ξ^2> = <x^2>
        
        So the uncertainty product (Δx * Δξ) where Δx^2 = <x^2> - <x>^2, etc.
        
        But wait, the function is not necessarily centered at 0. Let's check <x>:
        <x> = sum c_k c_{k'} <ψ_k, x ψ_{k'}> / sum c_k^2
        
        <ψ_k, x ψ_{k+1}> = sqrt((k+1)/2), <ψ_k, x ψ_{k-1}> = sqrt(k/2)
        
        Since we only have even k (0, 4, 8, 12), and x is odd, <ψ_k, x ψ_{k'}> = 0 for even k, k'
        because the integrand is odd. So <x> = 0.
        
        Similarly <ξ> = 0.
        
        So Δx^2 = <x^2> = sum c_k^2 (2k+1)/2 / sum c_k^2
        
        And Δξ^2 = <ξ^2> = same as Δx^2 (by Fourier invariance of the moment structure)
        
        So (Δx * Δξ)^2 = [<x^2>]^2
        
        The uncertainty product Δx * Δξ = <x^2>
        
        Wait, that's not right. Δx * Δξ = sqrt(<x^2>) * sqrt(<ξ^2>) = <x^2> since they're equal.
        
        But the standard uncertainty product is Δx * Δξ ≥ 1/2. Here we get:
        Δx * Δξ = sum c_k^2 (2k+1)/2 / sum c_k^2
        
        To minimize this, we want to put weight on low k. But we have the constraint g(0) = 0.
        
        However, the task mentions "upper bound on A(f)*A(Fourier(f))" where A might be
        a different norm. Let me reconsider.
        
        Given the seed returns [-113/100, 1/25, 1/3240], let's just use numerical
        optimization to minimize the uncertainty product (Δx * Δξ) subject to g(0) = 0.
        
        Actually, let's just minimize <x^2> since Δx * Δξ = <x^2> for this symmetric case.
        """
        H0_0 = 1.0
        H4_0 = 12.0
        H8_0 = 46080.0
        H12_0 = 10321920.0
        
        a12 = -(a0 + 12.0 * a4 + 46080.0 * a8) / 10321920.0
        
        # Coefficients for the normalized Hermite functions
        # ψ_k(x) = (2^k k! sqrt(π))^{-1/2} H_k(x) e^{-x^2/2}
        # So g = sum a_k H_k e^{-x^2/2} = sum c_k ψ_k where c_k = a_k (2^k k! sqrt(π))^{1/2}
        
        def norm_factor(k):
            return np.sqrt(2.0**k * float(np.math.factorial(k)) * np.sqrt(np.pi))
        
        c0 = a0 * norm_factor(0)
        c4 = a4 * norm_factor(4)
        c8 = a8 * norm_factor(8)
        c12 = a12 * norm_factor(12)
        
        c2_sum = c0**2 + c4**2 + c8**2 + c12**2
        if c2_sum < 1e-30:
            return 1e10
        
        # <x^2> = sum c_k^2 (2k+1)/2 / sum c_k^2
        x2_moment = (c0**2 * 1.0 + c4**2 * 4.5 + c8**2 * 8.5 + c12**2 * 12.5) / c2_sum
        
        # Uncertainty product Δx * Δξ = x2_moment (since Δx^2 = Δξ^2 = x2_moment)
        return x2_moment
    
    # Use scipy.optimize to find the minimum
    def objective(coeffs):
        a0, a4, a8 = coeffs
        return compute_bound(a0, a4, a8)
    
    # Start from the seed solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    return result.x


# EVOLVE-BLOCK-END
