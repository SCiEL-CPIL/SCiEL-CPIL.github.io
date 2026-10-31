"""
LJ38 cluster global minimum search.

Combined approach:
1. Build the FCC truncated-octahedron geometry explicitly (known global minimum candidate).
2. Generate diverse icosahedral-motif starting configurations to probe that basin.
3. Basin hopping with adaptive step size to escape local minima.
4. 1D scale optimization to fine-tune the overall cluster size.
5. Final high-precision L-BFGS polish.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero

    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)

    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)

    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale

    return pts


def _make_icosahedral_structure(rng=None):
    """
    Icosahedral-motif starting geometry with random shell radii and angular jitter.
    1 centre + 12 + 12 + 12 + 1 = 38.
    """
    if rng is None:
        rng = np.random.default_rng()

    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    # Random shell radii around typical values
    base_radii = [1.0, 1.7, 2.4]
    radii = [r * (1.0 + 0.1 * rng.standard_normal()) for r in base_radii]

    # Angular jitter: small random rotations
    def random_rotation(rng, angle_max=0.2):
        # Random axis
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = angle_max * rng.standard_normal()
        # Rodrigues' rotation formula
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    pts = [np.zeros(3)]
    for i, r in enumerate(radii):
        R = random_rotation(rng)
        layer = (ico @ R.T) * r
        # Add small per-atom jitter
        jitter = 0.05 * rng.standard_normal(layer.shape)
        layer += jitter
        pts.extend(layer)
    # 1 + 12*3 = 37, add one more
    extra = 0.25 * rng.standard_normal(3)
    pts.append(extra)
    pts = np.array(pts[:N], dtype=np.float64)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _relax(pos, max_iter=500, gtol=1e-8, ftol=1e-12):
    """L-BFGS-B local relaxation with analytic gradient."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        e, _ = _lj_energy_and_grad(pos)
        return pos, e


def _optimize_scale(pos, n_iter=50):
    """
    1D global search over uniform scale factor s applied to centered positions.
    Finds the s that minimizes total LJ energy.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0

    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s

    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s

    return centered * best_s + center, best_e


def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    rng = np.random.default_rng(12345)

    best_pos = None
    best_energy = np.inf

    # ---- Phase 1: Direct FCC construction and relaxation ----
    try:
        fcc = _make_fcc_cluster()
        p, e = _relax(fcc, max_iter=500, gtol=1e-8)
        p = _ensure_min_sep(p)
        e, _ = _lj_energy_and_grad(p)
        best_pos = p.copy()
        best_energy = e
    except Exception:
        fcc = None

    # ---- Phase 2: 1D scale optimization on FCC ----
    if best_pos is not None and time.time() < deadline - 30.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=30)
            p_scaled = _ensure_min_sep(p_scaled)
            e_scaled, _ = _lj_energy_and_grad(p_scaled)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled.copy()
        except Exception:
            pass

    # ---- Phase 3: Basin hopping with diverse starts ----
    if time.time() < deadline - 20.0:
        # Generate diverse icosahedral starts
        ico_starts = []
        for _ in range(3):
            if time.time() > deadline - 25.0:
                break
            try:
                ico = _make_icosahedral_structure(rng)
                ico = _ensure_min_sep(ico)
                ico_starts.append(ico)
            except Exception:
                pass

        # Also add perturbed FCC variants
        if fcc is not None:
            for _ in range(3):
                if time.time() > deadline - 25.0:
                    break
                perturbed = fcc + 0.05 * rng.standard_normal(fcc.shape)
                perturbed = _ensure_min_sep(perturbed)
                ico_starts.append(perturbed)

        # Basin hopping from best known structure
        current_pos = best_pos.copy()
        current_energy = best_energy
        step_size = 0.5
        T = 1.0
        accept_count = 0
        total_count = 0

        for start in ico_starts:
            if time.time() > deadline - 15.0:
                break
            try:
                p, e = _relax(start, max_iter=400, gtol=1e-6)
                p = _ensure_min_sep(p)
                e, _ = _lj_energy_and_grad(p)
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            except Exception:
                continue

        # Main basin hopping loop
        while time.time() < deadline - 10.0:
            # Perturb
            noise = rng.normal(0, step_size, size=current_pos.shape)
            perturbed = current_pos + noise
            perturbed = _ensure_min_sep(perturbed)

            # Relax
            new_pos, new_energy = _relax(perturbed, max_iter=400, gtol=1e-6)
            new_pos = _ensure_min_sep(new_pos)
            new_energy, _ = _lj_energy_and_grad(new_pos)

            total_count += 1

            # Acceptance Criterion
            delta_e = new_energy - current_energy

            if delta_e < 0:
                # Better energy, accept
                current_pos = new_pos
                current_energy = new_energy
                accept_count += 1

                # Update global best
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Worse energy, accept with probability exp(-delta_e / T)
                if rng.random() < np.exp(-delta_e / T):
                    current_pos = new_pos
                    current_energy = new_energy
                    accept_count += 1

            # Adaptive step size
            if total_count > 20:
                accept_rate = accept_count / total_count
                if accept_rate < 0.2:
                    step_size *= 0.8
                elif accept_rate > 0.8:
                    step_size *= 1.2
                step_size = max(0.1, min(2.0, step_size))

    # ---- Phase 4: Final high-precision polish ----
    if best_pos is not None and time.time() < deadline - 3.0:
        try:
            # Scale optimization
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            p_scaled = _ensure_min_sep(p_scaled)
            e_scaled, _ = _lj_energy_and_grad(p_scaled)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled.copy()

            # Tight L-BFGS polish
            p_final, e_final = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-14)
            p_final = _ensure_min_sep(p_final)
            e_final, _ = _lj_energy_and_grad(p_final)
            if e_final < best_energy:
                best_energy = e_final
                best_pos = p_final.copy()
        except Exception:
            pass

    # Fallback if all failed
    if best_pos is None:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        best_pos = np.array(pts[:N], dtype=np.float64)
        best_pos -= np.mean(best_pos, axis=0)
        best_pos *= 1.12
        best_pos = _ensure_min_sep(best_pos)

    return best_pos
