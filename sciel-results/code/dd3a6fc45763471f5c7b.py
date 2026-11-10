# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
import numpy.polynomial.legendre as npl


def _build_basis():
    """Build real orthonormal degree-10 spherical harmonics on a quadrature grid."""
    n_theta = 70
    n_phi = 130

    mu, w = npl.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi

    basis = np.zeros((21, n_theta, n_phi), dtype=np.float64)

    for m_idx in range(21):
        m = m_idx - 10
        abs_m = abs(m)

        # Compute N_m = sqrt(21/(4*pi) * (10-abs_m)!/(10+abs_m)!)
        num = 1.0
        for i in range(10 - abs_m + 1, 11):
            num *= i
        den = 1.0
        for i in range(10 - abs_m + 1, 10 + abs_m + 1):
            den *= i
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * num / den)

        # Associated Legendre with Condon-Shortley phase
        P = lpmv(abs_m, 10, mu)

        if m == 0:
            basis[m_idx] = N_m * P[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]

    weights = np.outer(w, np.full(n_phi, dphi))

    return basis, weights


def _get_axisymmetric_indices():
    """Return indices for axisymmetric modes: m=0,±2,±4,±6,±8,±10 (11 indices)."""
    return [10, 8, 12, 6, 14, 4, 16, 2, 18, 0, 20]


def _get_axisymmetric_indices():
    """Return indices into the 21-coeff array for axisymmetric modes m=0,±2,±4,±6,±8,±10."""
    # m values: -10,-9,...,10; axisymmetric means m even (including 0)
    # m=0: idx 10
    # m=±2: idx 8, 12
    # m=±4: idx 6, 14
    # m=±6: idx 4, 16
    # m=±8: idx 2, 18
    # m=±10: idx 0, 20
    return [10, 8, 12, 6, 14, 4, 16, 2, 18, 0, 20]


def _compute_R_and_grad(c, basis, weights):
    """Compute R = ||f||_4/||f||_2 and its gradient in one pass."""
    f = np.tensordot(c, basis, axes=(0, 0))

    f2 = f * f
    norm2_sq = np.sum(f2 * weights)

    if norm2_sq <= 0:
        return np.inf, np.zeros_like(c)

    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)

    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    R = norm4 / norm2

    # Gradient computation
    f3 = f2 * f
    int_f3_Y = np.einsum('jkl,kl->j', basis, f3 * weights)
    int_f_Y = np.einsum('jkl,kl->j', basis, f * weights)

    d_norm4_dc = int_f3_Y / (norm4 ** 3)
    d_norm2_dc = int_f_Y / norm2

    grad = d_norm4_dc / norm2 - (norm4 / norm2 ** 2) * d_norm2_dc

    return R, grad


def _compute_R(c, basis, weights):
    """Compute R = ||f||_4/||f||_2 only (faster, no gradient)."""
    f = np.tensordot(c, basis, axes=(0, 0))
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    if norm2_sq <= 0:
        return np.inf
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2


def _adam_step(c, grad, state, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer step."""
    t = state['t'] + 1
    state['t'] = t

    m = state['m']
    v = state['v']

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    c_new = c - lr * m_hat / (np.sqrt(v_hat) + eps)

    state['m'] = m
    state['v'] = v

    return c_new


def _gradient_descent(c, basis, weights, deadline, max_iter=5000):
    """Gradient descent with adaptive learning rate and momentum."""
    state = {'t': 0, 'm': np.zeros_like(c), 'v': np.zeros_like(c)}
    lr = 0.015
    best_R = _compute_R(c, basis, weights)
    best_c = c.copy()
    no_improve = 0

    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break

        R, grad = _compute_R_and_grad(c, basis, weights)
        if R < best_R:
            best_R = R
            best_c = c.copy()
            no_improve = 0

        c_new = _adam_step(c, grad, state, lr)

        if not np.all(np.isfinite(c_new)):
            lr *= 0.5
            state['t'] = 0
            state['m'] = np.zeros_like(c)
            state['v'] = np.zeros_like(c)
            continue

        R_new = _compute_R(c_new, basis, weights)

        if R_new < R - 1e-14:
            c = c_new
            lr = min(lr * 1.03, 0.15)
            no_improve = 0
        else:
            no_improve += 1
            lr *= 0.85
            state['t'] = max(0, state['t'] - 2)
            if no_improve > 500:
                break

    return best_c, best_R


def _newton_like_refinement(c, basis, weights, deadline, max_iter=100):
    """Newton-like refinement using Hessian approximation."""
    best_R = _compute_R(c, basis, weights)
    best_c = c.copy()

    lr = 0.005
    momentum = np.zeros_like(c)
    beta = 0.8

    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break

        R, grad = _compute_R_and_grad(c, basis, weights)
        if R < best_R:
            best_R = R
            best_c = c.copy()

        momentum = beta * momentum - lr * grad
        c_new = c + momentum

        if not np.all(np.isfinite(c_new)):
            lr *= 0.5
            momentum *= 0.5
            continue

        R_new = _compute_R(c_new, basis, weights)

        if R_new < R:
            c = c_new
            if R_new < best_R * 0.999:
                lr = min(lr * 1.05, 0.05)
        else:
            lr *= 0.9
            momentum *= 0.5

    return best_c, best_R


def _simulated_annealing(c, basis, weights, deadline, max_iter=2000, T0=0.1, T1=0.001):
    """Simulated annealing for escaping local minima."""
    best_R = _compute_R(c, basis, weights)
    best_c = c.copy()
    current_R = best_R
    current_c = c.copy()

    for i in range(max_iter):
        if time.time() > deadline - 5:
            break

        T = T0 * (T1 / T0) ** (i / max_iter)

        # Random perturbation
        scale = 0.05 * (1 + T / T0)
        c_trial = current_c + np.random.standard_normal(len(c)) * scale

        R_trial = _compute_R(c_trial, basis, weights)

        if R_trial < current_R or np.random.random() < np.exp(-(R_trial - current_R) / max(T, 1e-10)):
            current_c = c_trial
            current_R = R_trial

            if current_R < best_R:
                best_R = current_R
                best_c = current_c.copy()

    return best_c, best_R


def _optimize_axisymmetric(c_full, basis, weights, deadline, max_iter=5000):
    """Optimize within the axisymmetric subspace (11 coefficients: m=0,±2,±4,±6,±8,±10)."""
    ax_indices = _get_axisymmetric_indices()
    n_ax = len(ax_indices)

    # Extract axisymmetric coefficients
    c_ax = c_full[ax_indices].copy()

    # Build reduced basis (only axisymmetric modes)
    basis_ax = basis[ax_indices]  # shape (11, n_theta, n_phi)

    state = {'t': 0, 'm': np.zeros(n_ax), 'v': np.zeros(n_ax)}
    lr = 0.01
    best_R = _compute_R(c_ax, basis_ax, weights)
    best_c_ax = c_ax.copy()

    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break

        R, grad_ax = _compute_R_and_grad(c_ax, basis_ax, weights)
        if R < best_R:
            best_R = R
            best_c_ax = c_ax.copy()

        c_ax_new = _adam_step(c_ax, grad_ax, state, lr)

        if not np.all(np.isfinite(c_ax_new)):
            lr *= 0.5
            state['t'] = 0
            state['m'] = np.zeros(n_ax)
            state['v'] = np.zeros(n_ax)
            continue

        R_new = _compute_R(c_ax_new, basis_ax, weights)

        if R_new < R:
            c_ax = c_ax_new
            lr = min(lr * 1.02, 0.1)
        else:
            lr *= 0.9
            state['t'] = max(0, state['t'] - 1)

    # Map back to full 21-dim
    c_full_new = c_full.copy()
    c_full_new[ax_indices] = best_c_ax
    return c_full_new, best_R


def _high_precision_polish(c, basis, weights, deadline, max_iter=2000):
    """Final high-precision polish with very small steps and tight tolerance."""
    best_R = _compute_R(c, basis, weights)
    best_c = c.copy()

    lr = 0.002
    state = {'t': 0, 'm': np.zeros_like(c), 'v': np.zeros_like(c)}
    no_improve_count = 0

    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break

        R, grad = _compute_R_and_grad(c, basis, weights)
        if R < best_R:
            best_R = R
            best_c = c.copy()
            no_improve_count = 0

        # Use small learning rate for precision
        c_new = _adam_step(c, grad, state, lr)

        if not np.all(np.isfinite(c_new)):
            lr *= 0.5
            continue

        R_new = _compute_R(c_new, basis, weights)

        if R_new < R - 1e-13:
            c = c_new
            lr = min(lr * 1.02, 0.01)
            no_improve_count = 0
        else:
            no_improve_count += 1
            lr *= 0.9
            if lr < 1e-7 or no_improve_count > 100:
                break

    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonics.
    """
    start_time = time.time()
    deadline = start_time + 620.0

    # Build basis
    basis, weights = _build_basis()

    rng = np.random.default_rng(42)

    # Generate diverse initial guesses
    initial_guesses = []

    # Single m modes (often good starting points for flat harmonics)
    for m in range(-10, 11):
        guess = np.zeros(21, dtype=np.float64)
        guess[m + 10] = 1.0
        initial_guesses.append(guess)

    # Pairs of m modes with various phase combinations
    for m1 in range(-10, 11, 2):
        for m2 in range(m1 + 1, 11, 2):
            for sign1 in [1.0, -1.0]:
                for sign2 in [1.0, -1.0]:
                    guess = np.zeros(21, dtype=np.float64)
                    guess[m1 + 10] = sign1
                    guess[m2 + 10] = sign2
                    initial_guesses.append(guess)

    # Triple combinations
    for m1 in range(-10, 11, 3):
        for m2 in range(m1 + 1, 11, 3):
            for m3 in range(m2 + 1, 11, 3):
                guess = np.zeros(21, dtype=np.float64)
                guess[m1 + 10] = 1.0
                guess[m2 + 10] = 1.0
                guess[m3 + 10] = 1.0
                initial_guesses.append(guess)

    # Random guesses with various scales
    for scale in [0.1, 0.3, 0.5, 1.0]:
        for _ in range(5):
            initial_guesses.append(rng.standard_normal(21) * scale)

    # Structured: alternating signs on even m
    for pattern in [
        [1.0 if (i % 4 == 0) else (-1.0 if (i % 4 == 2) else 0.0) for i in range(21)],
        [1.0 if (i % 2 == 0) else 0.0 for i in range(21)],
        [(-1.0) ** (i // 2) if (i % 2 == 0) else 0.0 for i in range(21)],
    ]:
        initial_guesses.append(np.array(pattern, dtype=np.float64))

    best_c = None
    best_R = np.inf

    # Phase 1: Try each initial guess with Adam, limited iterations per guess
    max_iter_per_guess = 1500
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break

        c = guess.copy()
        c, R = _gradient_descent(c, basis, weights, deadline, max_iter=max_iter_per_guess)

        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Phase 2: Refine the best result with Newton-like method
    if best_c is not None and time.time() < deadline - 30:
        c, R = _newton_like_refinement(best_c, basis, weights, deadline, max_iter=800)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Phase 3: Simulated annealing from the best point
    if best_c is not None and time.time() < deadline - 40:
        c, R = _simulated_annealing(best_c, basis, weights, deadline, max_iter=2000)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Phase 4: More random restarts if time permits
    if best_c is not None and time.time() < deadline - 25:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            guess = rng.standard_normal(21) * 0.3
            c, R = _gradient_descent(c, basis, weights, deadline, max_iter=800)
            if R < best_R:
                best_R = R
                best_c = c.copy()

    # Phase 5: Final high-precision polish
    if best_c is not None and time.time() < deadline - 10:
        c, R = _high_precision_polish(best_c, basis, weights, deadline, max_iter=3000)
        if R < best_R:
            best_R = R
            best_c = c.copy()

    # Final validation
    if best_c is None or not np.all(np.isfinite(best_c)):
        return np.ones(21, dtype=np.float64)

    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1.0 / max(1e-3, max_abs))
    elif max_abs > 1e5:
        best_c = best_c * (1e5 / max_abs)

    return best_c


# EVOLVE-BLOCK-END
