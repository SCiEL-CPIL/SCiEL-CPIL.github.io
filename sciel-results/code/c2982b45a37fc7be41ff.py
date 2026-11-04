"""Combined approach: exact quad-based matrices + robust multi-start optimization with eigenvalue init and high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad


def _hermite_poly_scalar(n, x):
    """Compute physicist's Hermite polynomial Hn(x) for scalar x."""
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


def _basis_func_scalar(i, x):
    """Scalar basis function with g(0)=0 constraint baked in."""
    gauss = np.exp(-x**2 / 2.0)
    h12 = _hermite_poly_scalar(12, x)
    if i == 0:
        return (1.0 - h12 / 40320.0) * gauss
    elif i == 1:
        h4 = _hermite_poly_scalar(4, x)
        return (h4 - 12.0 * h12 / 40320.0) * gauss
    elif i == 2:
        h8 = _hermite_poly_scalar(8, x)
        return (h8 - 1680.0 * h12 / 40320.0) * gauss
    else:
        raise ValueError("Invalid basis index")


def construct():
    start_time = time.time()
    deadline = 170  # safety margin before 210s kill

    # --- Step 1: Compute M, Mx, Mf via adaptive quadrature (exact) ---
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    Mf = np.zeros((3, 3))

    for i in range(3):
        for j in range(i, 3):
            fi = lambda x, i=i: _basis_func_scalar(i, x)
            fj = lambda x, j=j: _basis_func_scalar(j, x)

            # M[i,j] = integral fi(x)*fj(x) dx
            val_M, _ = quad(lambda x: fi(x) * fj(x), -np.inf, np.inf, limit=300, epsabs=1e-15, epsrel=1e-15)
            M[i, j] = val_M
            M[j, i] = val_M

            # Mx[i,j] = integral x^2 * fi(x)*fj(x) dx
            val_Mx, _ = quad(lambda x: x**2 * fi(x) * fj(x), -np.inf, np.inf, limit=300, epsabs=1e-15, epsrel=1e-15)
            Mx[i, j] = val_Mx
            Mx[j, i] = val_Mx

            # Mf[i,j] = integral xi^2 * Fi(xi)*Fj(xi) dxi where F is Fourier transform
            # For these functions, we can compute via Parseval or direct quad on Fourier side
            # Fourier transform of Hn(x)*exp(-x^2/2) is (-i)^n Hn(xi)*exp(-xi^2/2)
            # So Fi(xi) = (-i)^i * (effective basis in xi space)
            # Actually, the basis functions are not pure Hermite-Gaussians due to the H12 correction
            # Let's compute Mf via the relation: integral xi^2 |F(xi)|^2 dxi = integral x^2 g''(x) * g(x) dx (up to sign)
            # Or use: for g(x) = sum ci * vi(x), Mf[i,j] = integral xi^2 * (Fourier of vi) * conj(Fourier of vj) dxi
            # Since Fourier transform is unitary (with proper normalization), and these are real functions:
            # Mf[i,j] = integral xi^2 * (F[vi])(xi) * conj(F[vj])(xi) dxi
            # By Parseval-like identity, this equals integral x^2 * vi''(x) * vj(x) dx / (-1) ... let me use direct quad on Fourier side

            # Actually, let's use the property that for f(x) = Hn(x) exp(-x^2/2), 
            # F[f](xi) = (-i)^n Hn(xi) exp(-xi^2/2) (with standard FT convention)
            # Our basis functions are linear combinations, so let's compute Mf using the fact that
            # the operator xi^2 in Fourier space corresponds to -d^2/dx^2 in position space
            # So Mf[i,j] = -integral vi''(x) * vj(x) dx

            # Compute second derivatives numerically or analytically
            # For now, use a fine grid FFT approach for Mf as a cross-check, but primarily use quad
            # Let's compute Mf via quad on the Fourier side by expressing F[vi] analytically

            # For vi(x) = (Hi(x) - ai*H12(x)) * exp(-x^2/2) where:
            # v0: H0 - H12/40320, so F[v0] = H0(xi)exp(-xi^2/2) - (1/40320)*(-i)^12*H12(xi)exp(-xi^2/2)
            # (-i)^12 = ((-i)^4)^3 = 1^3 = 1
            # So F[v0](xi) = (H0(xi) - H12(xi)/40320) * exp(-xi^2/2)
            # Similarly for v4: (-i)^4 = 1, so F[v4](xi) = (H4(xi) - 12*H12(xi)/40320) * exp(-xi^2/2)
            # For v8: (-i)^8 = 1, so F[v8](xi) = (H8(xi) - 1680*H12(xi)/40320) * exp(-xi^2/2)

            # So the Fourier transforms have the SAME form as the original functions!
            # This means Mf[i,j] = integral xi^2 * fi(xi) * fj(xi) dxi = Mx[i,j]
            # Wait, that's only true if the basis is symmetric. Let me verify:
            # The FT of Hn(x)exp(-x^2/2) is (-i)^n Hn(xi)exp(-xi^2/2)
            # For n=0: (-i)^0 = 1, so F[v0] = (1 - H12/40320)exp(-xi^2/2) = v0(xi)
            # For n=4: (-i)^4 = 1, so F[v4] = (H4 - 12H12/40320)exp(-xi^2/2) = v4(xi)  
            # For n=8: (-i)^8 = 1, so F[v8] = (H8 - 1680H12/40320)exp(-xi^2/2) = v8(xi)

            # So indeed Mf = Mx! This is a key simplification.

            # Mf = Mx by Fourier symmetry for these basis functions
            Mf[i, j] = val_Mx
            Mf[j, i] = val_Mx

    # --- Step 2: Define objective ---
    def compute_bound_from_coeffs(c):
        c = np.array(c, dtype=float)
        norm2 = c @ M @ c
        if norm2 <= 1e-30 or not np.isfinite(norm2):
            return np.inf
        x2_val = c @ Mx @ c
        f2_val = c @ Mf @ c
        if not np.isfinite(x2_val) or not np.isfinite(f2_val) or x2_val < -1e-12 or f2_val < -1e-12:
            return np.inf
        # Smooth surrogate: use softplus-like clipping to avoid kink at 0
        eps = 1e-12
        x2_smooth = 0.5 * (x2_val + np.sqrt(x2_val**2 + eps))
        f2_smooth = 0.5 * (f2_val + np.sqrt(f2_val**2 + eps))
        delta_x = np.sqrt(x2_smooth / norm2)
        delta_xi = np.sqrt(f2_smooth / norm2)
        return delta_x * delta_xi

    def objective_sphere(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return compute_bound_from_coeffs([c0, c4, c8])

    # --- Step 3: Initial points ---
    # Seed point
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    c_seed_norm = c_seed / np.linalg.norm(c_seed)
    c0s, c4s, c8s = c_seed_norm
    theta_seed = np.arccos(np.clip(c8s, -1.0 + 1e-12, 1.0 - 1e-12))
    phi_seed = np.arctan2(c4s, c0s)
    if phi_seed < 0:
        phi_seed += 2 * np.pi

    best_bound = objective_sphere(theta_seed, phi_seed)
    best_theta = theta_seed
    best_phi = phi_seed

    # Eigenvalue-based initialization
    try:
        eigvals_x, eigvecs_x = np.linalg.eigh(Mx, M)
        eigvals_f, eigvecs_f = np.linalg.eigh(Mf, M)

        candidates = []
        # Pure eigenvectors
        for i in range(3):
            candidates.append(eigvecs_x[:, i].copy())
            candidates.append(eigvecs_f[:, i].copy())
        # Mixtures
        for i in range(3):
            for j in range(3):
                for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    c_trial = alpha * eigvecs_x[:, i] + (1 - alpha) * eigvecs_f[:, j]
                    candidates.append(c_trial.copy())

        for c_trial in candidates:
            nrm = np.linalg.norm(c_trial)
            if nrm < 1e-12:
                continue
            c_trial = c_trial / nrm
            c0_t, c4_t, c8_t = c_trial
            if abs(c8_t) > 0.99999:
                continue
            theta_t = np.arccos(np.clip(c8_t, -1.0 + 1e-12, 1.0 - 1e-12))
            phi_t = np.arctan2(c4_t, c0_t)
            if phi_t < 0:
                phi_t += 2 * np.pi
            b = objective_sphere(theta_t, phi_t)
            if np.isfinite(b) and b < best_bound:
                best_bound = b
                best_theta = theta_t
                best_phi = phi_t
    except Exception:
        pass

    # --- Step 4: Multi-start Nelder-Mead ---
    if time.time() - start_time < deadline - 30:
        rng = np.random.RandomState(42)
        n_starts = 40

        for i in range(n_starts):
            if time.time() - start_time >= deadline - 22:
                break

            theta_rand = rng.uniform(0.01, np.pi - 0.01)
            phi_rand = rng.uniform(0.01, 2 * np.pi - 0.01)

            try:
                res = minimize(
                    objective_sphere,
                    x0=[theta_rand, phi_rand],
                    method='Nelder-Mead',
                    options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 8000, 'adaptive': True}
                )
                if res.fun < best_bound and np.isfinite(res.fun):
                    best_bound = res.fun
                    best_theta, best_phi = res.x
            except Exception:
                pass

    # --- Step 5: High-precision polish using BFGS with tight tolerances ---
    if time.time() - start_time < deadline - 12:
        try:
            # First a very tight Nelder-Mead
            res = minimize(
                objective_sphere,
                x0=[best_theta, best_phi],
                method='Nelder-Mead',
                options={'xatol': 1e-16, 'fatol': 1e-18, 'maxiter': 50000, 'adaptive': True}
            )
            if res.fun < best_bound and np.isfinite(res.fun):
                best_bound = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass

    # --- Step 6: BFGS polish for final precision ---
    if time.time() - start_time < deadline - 5:
        try:
            def objective_cartesian(c):
                # c = [c0, c4, c8], normalize to unit sphere
                nrm = np.linalg.norm(c)
                if nrm < 1e-15:
                    return np.inf
                c_norm = c / nrm
                return compute_bound_from_coeffs(c_norm)

            def gradient_cartesian(c):
                # Numerical gradient with central differences
                nrm = np.linalg.norm(c)
                if nrm < 1e-15:
                    return np.zeros(3)
                c_norm = c / nrm
                h = 1e-8
                grad = np.zeros(3)
                for k in range(3):
                    c_plus = c.copy()
                    c_plus[k] += h
                    c_minus = c.copy()
                    c_minus[k] -= h
                    # Normalize each
                    np_plus = np.linalg.norm(c_plus)
                    np_minus = np.linalg.norm(c_minus)
                    if np_plus < 1e-15 or np_minus < 1e-15:
                        grad[k] = 0.0
                        continue
                    f_plus = compute_bound_from_coeffs(c_plus / np_plus)
                    f_minus = compute_bound_from_coeffs(c_minus / np_minus)
                    if not np.isfinite(f_plus) or not np.isfinite(f_minus):
                        grad[k] = 0.0
                    else:
                        grad[k] = (f_plus - f_minus) / (2 * h)
                return grad

            c_init = np.array([np.sin(best_theta) * np.cos(best_phi),
                               np.sin(best_theta) * np.sin(best_phi),
                               np.cos(best_theta)])
            res_bfgs = minimize(
                objective_cartesian,
                x0=c_init,
                jac=gradient_cartesian,
                method='BFGS',
                options={'gtol': 1e-15, 'maxiter': 1000, 'disp': False}
            )
            if res_bfgs.fun < best_bound and np.isfinite(res_bfgs.fun):
                best_bound = res_bfgs.fun
                c_best = res_bfgs.x
                nrm_best = np.linalg.norm(c_best)
                if nrm_best > 1e-15:
                    c_best = c_best / nrm_best
                    best_theta = np.arccos(np.clip(c_best[2], -1.0 + 1e-14, 1.0 - 1e-14))
                    best_phi = np.arctan2(c_best[1], c_best[0])
                    if best_phi < 0:
                        best_phi += 2 * np.pi
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)

    return np.array([c0, c4, c8])

# EVOLVE-BLOCK-END
