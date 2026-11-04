"""Rewritten program: Analytic Gram matrix + multi-start optimization for uncertainty inequality."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _hermite_val(n, x):
    """Compute physicist's Hermite polynomial Hn(x) at scalar x."""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    h0 = 1.0
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _hermite_coeffs(n):
    """Return list of coefficients [c_0, c_1, ..., c_n] for H_n(x) = sum c_k x^k."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    # Recurrence: H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
    prev2 = [1.0]  # H_0
    prev1 = [0.0, 2.0]  # H_1
    for k in range(2, n + 1):
        curr = [0.0] * (k + 1)
        # 2x * prev1: shift prev1 by 1, multiply by 2
        for j in range(len(prev1)):
            curr[j + 1] += 2.0 * prev1[j]
        # -2(k-1) * prev2
        for j in range(len(prev2)):
            curr[j] -= 2.0 * (k - 1) * prev2[j]
        prev2, prev1 = prev1, curr
    return prev1


def construct():
    start_time = time.time()
    deadline = 195  # safety margin before 210s kill

    # Hermite polynomial coefficients
    H0_c = _hermite_coeffs(0)   # [1]
    H4_c = _hermite_coeffs(4)   # [0, 0, 0, 0, 16] -> wait let me check
    H8_c = _hermite_coeffs(8)
    H12_c = _hermite_coeffs(12)

    # H_n(0) values: H_0(0)=1, H_4(0)=12, H_8(0)=1680, H_12(0)=40320
    H0_at0 = _hermite_val(0, 0.0)   # 1
    H4_at0 = _hermite_val(4, 0.0)   # 12
    H8_at0 = _hermite_val(8, 0.0)   # 1680
    H12_at0 = _hermite_val(12, 0.0)  # 40320

    # Constraint: g(0) = 0 => c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c12 = -(c0*1 + c4*12 + c8*1680) / 40320

    # Analytic Gram matrices using Hermite-Gaussian orthogonality
    # v_i(x) = [H_{n_i}(x) - alpha_i * H_12(x)] * exp(-x^2/2)
    # where alpha_0 = 1/40320, alpha_4 = 12/40320, alpha_8 = 1680/40320

    # Key integral: ∫ H_m(x) H_n(x) e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_{mn}
    # ∫ x^2 H_m(x) H_n(x) e^{-x^2} dx: use x^2 = (1/2)(H_2(x)/4 + 2x H_1(x)/2 + ... )
    # Actually: x^2 e^{-x^2} = (1/4)H_2(x) e^{-x^2} + (1/2) e^{-x^2}  ... let me use the relation:
    # x H_n = (1/2)H_{n+1} + n H_{n-1}
    # x^2 H_n = (1/4)H_{n+2} + (n + 1/2) H_n + (n(n-1)/4) H_{n-2}

    def x2_hermite_integral(m, n):
        """Compute ∫ x^2 H_m(x) H_n(x) e^{-x^2} dx analytically."""
        # x^2 H_m = (1/4)H_{m+2} + (m + 1/2)H_m + (m(m-1)/4)H_{m-2} (for m>=2)
        # For m=0: x^2 * 1 = (1/4)H_2 + (1/2)H_0
        # For m=1: x^2 * 2x = 2x^3 = 2*(1/2 H_3 + 1*H_1) = H_3 + 2H_1... let me just use the general formula
        # x^2 H_m(x) = (1/4)H_{m+2}(x) + (m + 1/2)H_m(x) + (m(m-1)/4)H_{m-2}(x)
        # This works for m >= 2. For m=0: x^2 = (1/4)H_2 + (1/2)H_0
        # For m=1: x^2*2x = 2x^3. x^3 = (1/2)H_3 + (3/2)H_1... wait
        # x^3 = x*x^2. Let me use: x H_n = (1/2)H_{n+1} + n H_{n-1}
        # x^2 H_m = x * (x H_m) = x * [(1/2)H_{m+1} + m H_{m-1}]
        # = (1/2)[(1/2)H_{m+2} + (m+1)H_m] + m[(1/2)H_m + (m-1)H_{m-2}]
        # = (1/4)H_{m+2} + (m+1)/2 H_m + m/2 H_m + m(m-1)/2 H_{m-2}
        # = (1/4)H_{m+2} + (m + 1/2) H_m + (m(m-1)/2) H_{m-2}
        # Wait, that doesn't match. Let me recheck: x H_n = (1/2)H_{n+1} + n H_{n-1}
        # So x^2 H_m = x * [(1/2)H_{m+1} + m H_{m-1}]
        # = (1/2) x H_{m+1} + m x H_{m-1}
        # = (1/2)[(1/2)H_{m+2} + (m+1)H_m] + m[(1/2)H_m + (m-1)H_{m-2}]
        # = (1/4)H_{m+2} + (m+1)/2 H_m + m/2 H_m + m(m-1)/2 H_{m-2}
        # = (1/4)H_{m+2} + (2m+1)/2 H_m + m(m-1)/2 H_{m-2}

        sqrt_pi = np.sqrt(np.pi)

        def fact(k):
            result = 1
            for i in range(2, k + 1):
                result *= i
            return result

        # Orthogonality: ∫ H_a H_b e^{-x^2} dx = sqrt(pi) * 2^a * a! * delta_{ab} (for a<=b)
        # Actually it's symmetric: ∫ H_m H_n e^{-x^2} dx = sqrt(pi) * 2^{min(m,n)} * min(m,n)! * delta_{mn}
        # No wait: ∫ H_m(x) H_n(x) e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_{mn} where n = m
        # More precisely: = sqrt(pi) * 2^m * m! * delta_{mn}

        def ortho(a, b):
            if a == b:
                return sqrt_pi * (2.0 ** a) * float(fact(a))
            return 0.0

        if m == 0:
            # x^2 * H_0 = x^2
            # x^2 = (1/4)H_2 + (1/2)H_0
            result = (1.0/4.0) * ortho(2, n) + (1.0/2.0) * ortho(0, n)
            return result
        elif m == 1:
            # x^2 * H_1 = x^2 * 2x = 2x^3
            # x^3 = (1/2)H_3 + (3/2)H_1  [from x H_2 = (1/2)H_3 + 2H_1, and H_2=4x^2-2, so x*H_2=4x^3-2x, thus 4x^3 = 2x H_2 + 2x = 2[(1/2)H_3+2H_1]+2H_1 = H_3+4H_1+2H_1=H_3+6H_1, so x^3=(H_3+6H_1)/4]
            # Let me just use: x^3 = (1/4)H_3 + (3/2)H_1
            # So 2x^3 = (1/2)H_3 + 3H_1
            result = (1.0/2.0) * ortho(3, n) + 3.0 * ortho(1, n)
            return result
        else:
            # x^2 H_m = (1/4)H_{m+2} + (2m+1)/2 H_m + m(m-1)/2 H_{m-2}
            result = (1.0/4.0) * ortho(m + 2, n) + (2*m + 1)/2.0 * ortho(m, n) + m*(m-1)/2.0 * ortho(m - 2, n)
            return result

    def hermite_ortho(m, n):
        """∫ H_m(x) H_n(x) e^{-x^2} dx"""
        sqrt_pi = np.sqrt(np.pi)
        if m == n:
            def fact(k):
                result = 1
                for i in range(2, k + 1):
                    result *= i
                return result
            return sqrt_pi * (2.0 ** m) * float(fact(m))
        return 0.0

    # Effective basis: v_i = H_{n_i} - alpha_i * H_12, all times e^{-x^2/2}
    # M[i,j] = ∫ v_i v_j dx = ∫ (H_{n_i} - alpha_i H_12)(H_{n_j} - alpha_j H_12) e^{-x^2} dx
    #        = ∫ H_{n_i} H_{n_j} e^{-x^2} dx - alpha_j ∫ H_{n_i} H_12 e^{-x^2} dx - alpha_i ∫ H_{n_j} H_12 e^{-x^2} dx + alpha_i alpha_j ∫ H_12^2 e^{-x^2} dx

    alphas = [1.0 / 40320.0, 12.0 / 40320.0, 1680.0 / 40320.0]
    degrees = [0, 4, 8]

    # Compute M, Mx analytically
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            ni, nj = degrees[i], degrees[j]
            ai, aj = alphas[i], alphas[j]
            # M[i,j]
            M[i, j] = (hermite_ortho(ni, nj) - aj * hermite_ortho(ni, 12)
                       - ai * hermite_ortho(nj, 12) + ai * aj * hermite_ortho(12, 12))
            # Mx[i,j]
            Mx[i, j] = (x2_hermite_integral(ni, nj) - aj * x2_hermite_integral(ni, 12)
                        - ai * x2_hermite_integral(nj, 12) + ai * aj * x2_hermite_integral(12, 12))

    # Fourier transform: The FT of H_n(x) e^{-x^2/2} is (-i)^n H_n(ξ) e^{-ξ^2/2} (with convention F[f](ξ) = ∫ f(x) e^{-iξx} dx)
    # So |FT|^2 = H_n(ξ)^2 e^{-ξ^2}, and ∫ ξ^2 |FT|^2 dξ = ∫ ξ^2 H_n(ξ)^2 e^{-ξ^2} dξ
    # By symmetry, this equals ∫ x^2 H_n(x)^2 e^{-x^2} dx = x2_hermite_integral(n, n)
    # For cross terms: ∫ ξ^2 Re(FT(H_m e^{-x^2/2}) * conj(FT(H_n e^{-x^2/2}))) dξ
    # = ∫ ξ^2 Re((-i)^m H_m(ξ) e^{-ξ^2/2} * (-i)^{-n} H_n(ξ) e^{-ξ^2/2}) dξ
    # = Re((-i)^{m-n}) ∫ ξ^2 H_m(ξ) H_n(ξ) e^{-ξ^2} dξ
    # Since m,n are both even (0,4,8), m-n is even, so (-i)^{m-n} = (-1)^{(m-n)/2} which is real
    # For m=n: (-i)^0 = 1
    # For m≠n with both even: the integral is 0 by orthogonality anyway

    # Actually for the cross terms in Mf:
    # Mf[i,j] = ∫ ξ^2 Re(FT(v_i) conj(FT(v_j))) dξ
    # FT(v_i) = FT(H_{n_i} e^{-x^2/2}) - alpha_i FT(H_12 e^{-x^2/2})
    # = (-i)^{n_i} H_{n_i}(ξ) e^{-ξ^2/2} - alpha_i (-i)^{12} H_12(ξ) e^{-ξ^2/2}
    # = [(-i)^{n_i} H_{n_i}(ξ) - alpha_i (-i)^{12} H_12(ξ)] e^{-ξ^2/2}
    # Since all degrees are even, (-i)^{n_i} = (-1)^{n_i/2} which is real
    # So FT(v_i) is real-valued (up to the exponential), and the cross terms simplify

    # For even n: (-i)^n = (-1)^{n/2}
    # n=0: (-i)^0 = 1
    # n=4: (-i)^4 = 1
    # n=8: (-i)^8 = 1
    # n=12: (-i)^12 = 1
    # All are 1! So FT(v_i) = [H_{n_i}(ξ) - alpha_i H_12(ξ)] e^{-ξ^2/2}
    # This is the same form as v_i but in frequency space!

    # Therefore Mf[i,j] = ∫ ξ^2 [H_{n_i}(ξ) - alpha_i H_12(ξ)][H_{n_j}(ξ) - alpha_j H_12(ξ)] e^{-ξ^2} dξ
    # = x2_hermite_integral(ni, nj) - alpha_j x2_hermite_integral(ni, 12) - alpha_i x2_hermite_integral(nj, 12) + alpha_i alpha_j x2_hermite_integral(12, 12)
    # = Mx[i,j] !

    # Wait, that means Mf = Mx? Let me verify...
    # Actually the Fourier transform convention matters. If F[f](ξ) = ∫ f(x) e^{-2πiξx} dx, then
    # the relationship is different. But with the convention F[f](ξ) = ∫ f(x) e^{-iξx} dx,
    # and for the uncertainty product, we need ∫ x^2 |f(x)|^2 dx * ∫ ξ^2 |F[f](ξ)|^2 dξ
    # By Plancherel and the specific form, for Hermite-Gaussians this gives the same structure.

    # Let me verify: for a single Hermite-Gaussian H_n e^{-x^2/2}:
    # ∫ x^2 H_n^2 e^{-x^2} dx = x2_hermite_integral(n,n)
    # ∫ ξ^2 |FT|^2 dξ = ∫ ξ^2 H_n^2(ξ) e^{-ξ^2} dξ = x2_hermite_integral(n,n)
    # So yes, they're equal for each basis element. And by the same argument for cross terms.

    Mf = Mx.copy()  # They are identical for even-degree Hermite-Gaussians with this FT convention

    def compute_bound_from_coeffs(c):
        """Compute bound for a specific coefficient vector c."""
        c = np.array(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
        x2_norm = c @ Mx @ c
        freq2_norm = c @ Mf @ c
        if not np.isfinite(freq2_norm) or freq2_norm < 0 or not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
        delta_x = np.sqrt(x2_norm / norm2)
        delta_xi = np.sqrt(freq2_norm / norm2)
        return delta_x * delta_xi

    # Since Mf = Mx, the objective simplifies to (c^T Mx c) / (c^T M c)
    # This is a generalized Rayleigh quotient. The minimum is the smallest
    # eigenvalue of M^{-1} Mx.
    try:
        eigvals = np.linalg.eigvals(np.linalg.inv(M) @ Mx)
        eigvals_real = np.real(eigvals)
        eigvals_real = eigvals_real[eigvals_real > 0]
        if len(eigvals_real) > 0:
            analytic_min = np.min(eigvals_real)
            # Get the eigenvector for the smallest eigenvalue
            eigvals_full, eigvecs = np.linalg.eig(np.linalg.inv(M) @ Mx)
            idx = np.argmin(np.real(eigvals_full))
            c_analytic = np.real(eigvecs[:, idx])
            c_analytic = c_analytic / np.linalg.norm(c_analytic)
            analytic_bound = compute_bound_from_coeffs(c_analytic)
            if analytic_bound < best_bound and np.isfinite(analytic_bound):
                best_bound = analytic_bound
                best_coeffs = c_analytic.copy()
    except Exception:
        pass

    # Spherical parameterization for optimization
    def compute_bound_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        c = np.array([c0, c4, c8])
        return compute_bound_from_coeffs(c)

    def compute_bound_unconstrained(c):
        """Optimize over unconstrained 3D vector, normalize internally."""
        c = np.array(c, dtype=float)
        norm = np.linalg.norm(c)
        if norm < 1e-15:
            return np.inf
        c = c / norm
        return compute_bound_from_coeffs(c)

    # Start with seed
    best_bound = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = compute_bound_from_coeffs(best_coeffs)

    # Analytic solution via generalized eigenvalue problem (Mf = Mx case)
    try:
        eigvals_full, eigvecs = np.linalg.eig(np.linalg.inv(M) @ Mx)
        for idx in range(len(eigvals_full)):
            c_candidate = np.real(eigvecs[:, idx])
            c_norm = np.linalg.norm(c_candidate)
            if c_norm > 1e-15:
                c_candidate = c_candidate / c_norm
                bound = compute_bound_from_coeffs(c_candidate)
                if bound < best_bound and np.isfinite(bound):
                    best_bound = bound
                    best_coeffs = c_candidate.copy()
    except Exception:
        pass

    # Phase 1: Differential Evolution for global search
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(-1, 1), (-1, 1), (-1, 1)]
            de_result = differential_evolution(
                compute_bound_unconstrained,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=30,
                tol=1e-12,
                polish=True,
                workers=1,
                updating='immediate'
            )
            if de_result.fun < best_bound and np.isfinite(de_result.fun):
                best_coeffs = de_result.x.copy()
                best_bound = de_result.fun
        except Exception:
            pass

    # Phase 2: Multi-start local optimization
    if time.time() - start_time < deadline - 20:
        rng = np.random.RandomState(123)
        n_starts = 200
        for i in range(n_starts):
            if time.time() - start_time > deadline - 10:
                break
            # Random direction on sphere
            z = 2 * rng.random() - 1
            theta = 2 * np.pi * rng.random()
            r = np.sqrt(max(0, 1 - z**2))
            c_init = np.array([r * np.cos(theta), r * np.sin(theta), z])

            try:
                res = minimize(
                    compute_bound_unconstrained,
                    x0=c_init,
                    method='Nelder-Mead',
                    options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 3000, 'adaptive': True}
                )
                if res.fun < best_bound and np.isfinite(res.fun):
                    best_coeffs = res.x.copy()
                    best_bound = res.fun
            except Exception:
                pass

    # Phase 3: Final high-precision polish
    if time.time() - start_time < deadline - 3:
        try:
            c_norm = best_coeffs / np.linalg.norm(best_coeffs)
            res = minimize(
                compute_bound_unconstrained,
                x0=c_norm,
                method='Nelder-Mead',
                options={'xatol': 1e-16, 'fatol': 1e-18, 'maxiter': 10000, 'adaptive': True}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
                best_coeffs = res.x.copy()
                best_bound = res.fun
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
