"""LJ38 global minimum search: symmetry-adapted FCC construction with
topology-forcing relaxation, multi-scale perturbation, and high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE_S = 120.0
MARGIN = 15.0


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _topology_loss_and_grad(pos, target_pos):
    """
    Smooth differentiable topology loss: sum of squared deviations in pairwise distances.
    Returns (loss, gradient w.r.t. pos).
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2) + 1e-12
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    
    # Target pairwise distances
    tdiff = target_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :]
    tr2 = np.sum(tdiff * tdiff, axis=2) + 1e-12
    tr = np.sqrt(tr2)
    np.fill_diagonal(tr, 1.0)
    
    # Only upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_diff = r[i_idx, j_idx] - tr[i_idx, j_idx]
    loss = np.sum(d_diff ** 2)
    
    # Gradient: d_loss/d_pos_i = 2 * sum_{j!=i} (d_ij - d_tgt_ij) * (pos_i - pos_j) / d_ij
    grad = np.zeros_like(pos)
    coeff = 2.0 * (r - tr) / r
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return loss, grad


def _topology_forcing_relax(pos, target_pos, deadline=None, alpha=0.5, maxiter=2000):
    """
    Hybrid relaxation: minimize LJ energy + alpha * topology loss.
    This guides the structure toward the FCC target while still minimizing LJ energy.
    """
    pos = np.array(pos, dtype=float).copy()
    
    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e_lj, g_lj = _lj_energy_and_grad(p)
        e_top, g_top = _topology_loss_and_grad(p, target_pos)
        return e_lj + alpha * e_top, (g_lj + alpha * g_top).ravel()
    
    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14,
                                'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _relax(pos, maxiter=20000, deadline=None, gtol=1e-10, ftol=1e-16):
    """L-BFGS-B relaxation with optional deadline and precision control."""
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                                'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _get_fcc_shell_structure(n_atoms=38):
    """Return base FCC positions and shell assignments for LJ38 truncated octahedron."""
    r0 = 2.0 ** (1.0 / 6.0)
    
    scale = 1.0 / np.sqrt(2.0)
    max_coord = 6
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x * x + y * y + z * z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (round(p[3], 6), abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    radii = np.linalg.norm(coords, axis=1)
    unique_r = []
    for r in sorted(radii):
        if not unique_r or abs(r - unique_r[-1]) > 0.02:
            unique_r.append(r)
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.03:
                shell_ids[i] = sid
                break
    
    return coords, shell_ids, unique_r


def _make_fcc_truncated_octahedron():
    """
    Construct the LJ38 global-minimum truncated-octahedron geometry using
    symmetry-adapted parameter optimization.
    """
    r0 = 2.0 ** (1.0 / 6.0)
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    x0 = np.array([r0] + [0.0] * n_shells)
    
    def sym_energy_and_grad(params):
        n_atoms = len(base_coords)
        global_scale = params[0]
        shell_params = params[1:]
        
        positions = np.zeros((n_atoms, 3))
        breath = np.ones(n_atoms)
        for i in range(n_atoms):
            s = shell_ids[i]
            if s < len(shell_params):
                breath[i] = 1.0 + shell_params[s]
        positions = base_coords * breath[:, np.newaxis] * global_scale
        
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        r2 = np.maximum(r2, 1e-12)
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
        
        coeff = 4.0 * (-6.0 * r2 ** (-7) + 3.0 * r2 ** (-4))
        np.fill_diagonal(coeff, 0.0)
        grad_pos = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        grad_global = np.sum(grad_pos * (base_coords * breath[:, np.newaxis]))
        
        grad_shell = np.zeros(len(shell_params))
        for s in range(len(shell_params)):
            mask = (shell_ids == s)
            grad_shell[s] = np.sum(grad_pos[mask] * positions[mask])
        
        grad = np.concatenate([[grad_global], grad_shell])
        
        return energy, grad
    
    try:
        res = minimize(sym_energy_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 200})
        e_opt = res.fun
        e_init = sym_energy_and_grad(x0)[0]
        if e_opt < e_init:
            opt_params = res.x
        else:
            opt_params = x0
    except Exception:
        opt_params = x0
    
    global_scale = opt_params[0]
    shell_params = opt_params[1:]
    
    positions = np.zeros((N, 3))
    breath = np.ones(N)
    for i in range(N):
        s = shell_ids[i]
        if s < len(shell_params):
            breath[i] = 1.0 + shell_params[s]
    positions = base_coords * breath[:, np.newaxis] * global_scale
    
    positions -= positions.mean(axis=0)
    
    return positions


def _make_icosahedral():
    """Construct an icosahedral arrangement for 38 atoms (1+12+25)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    r0 = 2.0 ** (1.0 / 6.0)

    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * r0

    pos_list = [np.zeros(3)]
    pos_list.extend(list(verts))

    n_outer = N - len(pos_list)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        radius = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden
        v = 2.0 * r0 * np.array([radius * np.cos(theta),
                                  radius * np.sin(theta),
                                  z])
        pos_list.append(v)

    pos = np.array(pos_list[:N])
    pos -= pos.mean(axis=0)
    return pos


def _perturb(pos, sigma, rng):
    """Add Gaussian noise to positions."""
    return pos + rng.randn(*pos.shape) * sigma


def _basin_hop(pos, rng, n_hops=10, max_sigma=0.5, deadline=None, target_fcc=None):
    """Basin hopping with topology-forcing strategy."""
    best_pos = pos.copy()
    best_e = _lj_energy_and_grad(pos)[0]
    
    for hop in range(n_hops):
        if deadline is not None and time.time() > deadline:
            break
        
        # Use topology-forcing relaxation with random perturbation
        sigma = rng.uniform(0.05, max_sigma)
        perturbed = _perturb(best_pos, sigma, rng)
        
        if target_fcc is not None:
            # Try topology-forcing relaxation
            alpha = rng.uniform(0.1, 0.5)
            relaxed, e = _topology_forcing_relax(perturbed, target_fcc, deadline=deadline,
                                                  alpha=alpha, maxiter=2000)
        else:
            relaxed, e = _relax(perturbed, maxiter=5000, deadline=deadline,
                                gtol=1e-10, ftol=1e-14)
        
        if e < best_e:
            best_e = e
            best_pos = relaxed.copy()
    
    return best_pos, best_e


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) FCC truncated octahedron with symmetry-adapted optimization
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
        # High-precision polish pass
        if time.time() < deadline - 5.0:
            pos_r2, e2 = _relax(pos_r, maxiter=10000, deadline=deadline,
                                gtol=1e-12, ftol=1e-16)
            if e2 < best_e:
                best_e = e2
                best_pos = pos_r2.copy()
    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return best_pos

    # 2) Icosahedral start
    try:
        pos_ico = _make_icosahedral()
        pos_r, e = _relax(pos_ico, maxiter=15000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    if time.time() > deadline:
        if best_pos is not None:
            return best_pos

    # 3) Topology-forcing relaxation from multiple perturbed FCC starts
    rng = np.random.RandomState(12345)
    fcc_target = _make_fcc_truncated_octahedron()
    
    if time.time() < deadline - 15.0:
        # Try topology-forcing with various alphas and perturbations
        for alpha in [0.3, 0.5, 0.8]:
            if time.time() > deadline:
                break
            for sigma in [0.1, 0.2, 0.3]:
                if time.time() > deadline:
                    break
                pos_p = _perturb(fcc_target, sigma, rng)
                pos_r, e = _topology_forcing_relax(pos_p, fcc_target, deadline=deadline,
                                                   alpha=alpha, maxiter=3000)
                # Then polish with pure LJ relaxation
                if time.time() < deadline - 2.0:
                    pos_r2, e2 = _relax(pos_r, maxiter=5000, deadline=deadline,
                                        gtol=1e-10, ftol=1e-16)
                    if e2 < best_e:
                        best_e = e2
                        best_pos = pos_r2.copy()
                elif e < best_e:
                    best_e = e
                    best_pos = pos_r.copy()

    # 4) Basin hopping with topology-forcing
    if time.time() < deadline - 20.0:
        try:
            best_pos_bh, best_e_bh = _basin_hop(best_pos, rng, n_hops=15,
                                                 max_sigma=0.4, deadline=deadline,
                                                 target_fcc=fcc_target)
            if best_e_bh < best_e:
                best_e = best_e_bh
                best_pos = best_pos_bh.copy()
        except Exception:
            pass

    # 5) Multi-scale perturbations around the best FCC result
    base = _make_fcc_truncated_octahedron()
    
    remaining = deadline - time.time()
    if remaining > 30:
        perturb_schedule = [
            (0.01, 6),
            (0.03, 6),
            (0.05, 6),
            (0.08, 5),
            (0.12, 4),
            (0.18, 4),
            (0.25, 3),
            (0.35, 3)
        ]
    elif remaining > 15:
        perturb_schedule = [
            (0.02, 4),
            (0.05, 4),
            (0.08, 4),
            (0.12, 3),
            (0.18, 3),
            (0.25, 2)
        ]
    else:
        perturb_schedule = [
            (0.03, 3),
            (0.08, 3),
            (0.12, 2)
        ]
    
    for sigma, count in perturb_schedule:
        if time.time() > deadline:
            break
        for _ in range(count):
            if time.time() > deadline:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=8000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 6) Random starts (catch-all)
    if time.time() < deadline - 10.0:
        for _ in range(2):
            if time.time() > deadline:
                break
            pos_rand = rng.randn(N, 3) * 2.5
            pos_r, e = _relax(pos_rand, maxiter=5000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    if best_pos is None:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        return np.array(pts[:N], dtype=float) * 1.12

    # Final high-precision polish
    if time.time() < deadline - 3.0:
        pos_r, e = _relax(best_pos, maxiter=20000, deadline=deadline,
                          gtol=1e-14, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
            
    # Ultra-high precision pass
    if time.time() < deadline - 1.0:
        pos_r, e = _relax(best_pos, maxiter=5000, deadline=deadline,
                          gtol=1e-15, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # Ensure minimum pairwise distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
