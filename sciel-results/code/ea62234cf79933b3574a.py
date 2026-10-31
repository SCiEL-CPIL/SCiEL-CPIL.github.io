"""LJ38 global minimum search: symmetry-adapted FCC construction with
multi-scale perturbation, basin hopping, and high-precision L-BFGS polish."""
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


def _detect_icosahedral_motifs(pos, cutoff=1.5):
    """Detect atoms with icosahedral-like coordination (12 nearest neighbors).
    
    Returns a list of indices of atoms that appear to be in icosahedral
    coordination environments.
    """
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    
    ico_candidates = []
    for i in range(n):
        # Find neighbors within cutoff
        neighbor_mask = r2[i, :] < cutoff ** 2
        n_neighbors = np.sum(neighbor_mask)
        
        # Icosahedral coordination: ~12 neighbors
        if 10 <= n_neighbors <= 14:
            ico_candidates.append(i)
    
    return ico_candidates


def _topology_swap_displacement(pos, ico_atoms, target_fcc, rng):
    """Generate a targeted displacement to break icosahedral motifs.
    
    For each atom in an icosahedral environment, compute the direction toward
    the corresponding FCC position and apply a partial displacement.
    """
    n = pos.shape[0]
    displacement = np.zeros_like(pos)
    
    if len(ico_atoms) == 0:
        return displacement
    
    # For each icosahedral atom, find the closest FCC target
    for i in ico_atoms:
        # Find the FCC position closest to current position
        dists = np.linalg.norm(target_fcc - pos[i], axis=1)
        target_idx = np.argmin(dists)
        
        # Compute displacement direction
        direction = target_fcc[target_idx] - pos[i]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        
        # Apply partial displacement with some randomness
        scale = rng.uniform(0.1, 0.3)
        noise = rng.randn(3) * 0.05
        displacement[i] = direction * scale + noise
    
    return displacement


def _get_fcc_shell_structure(n_atoms=38):
    """Return base FCC positions and shell assignments for LJ38 truncated octahedron.
    
    Uses the known FCC lattice structure: 1 center + 12 first-shell + 14 second-shell + 11 third-shell
    The truncated octahedron has specific symmetry properties.
    """
    r0 = 2.0 ** (1.0 / 6.0)
    
    # Generate FCC lattice points
    # FCC: (i,j,k) where i+j+k is even, scaled by 1/sqrt(2)
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
    
    # Sort by radius, then by symmetry (abs values)
    points.sort(key=lambda p: (round(p[3], 6), abs(p[0]), abs(p[1]), abs(p[2])))
    
    # Select the n_atoms closest points
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Assign shells based on radial distance with tighter clustering
    radii = np.linalg.norm(coords, axis=1)
    # Use a tolerance-based clustering
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


def _symmetry_adapted_energy(params, base_coords, shell_ids):
    """Compute energy for symmetry-adapted parameters."""
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0


def _symmetry_adapted_energy_and_grad(params, base_coords, shell_ids):
    """
    params: [global_scale, shell_breathing_0, shell_breathing_1, ...]
    Returns energy and gradient w.r.t. params using analytical differentiation.
    """
    n_atoms = len(base_coords)
    global_scale = params[0]
    shell_params = params[1:]
    
    # Build positions
    positions = np.zeros((n_atoms, 3))
    breath = np.ones(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath[i] = 1.0 + shell_params[s]
    positions = base_coords * breath[:, np.newaxis] * global_scale
    
    # Compute energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Analytical gradient
    # dE/d(positions[i]) = sum_j 4*(-12*rij^-14 + 6*rij^-8) * (positions[i] - positions[j])
    # Using r2: dE/dr2 = 4*(-6*r2^-7 + 3*r2^-4)
    # dE/dxi = 2 * sum_j dE/dr2 * (xi - xj)
    
    coeff = 4.0 * (-6.0 * r2 ** (-7) + 3.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad_pos = 2.0 * np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    # Chain rule: dE/d(global_scale) = sum_i grad_pos[i] . (positions[i] / global_scale)
    # Since positions[i] = base_coords[i] * breath[i] * global_scale
    # d(positions[i])/d(global_scale) = base_coords[i] * breath[i]
    grad_global = np.sum(grad_pos * (base_coords * breath[:, np.newaxis]))
    
    # dE/d(shell_breathing[s]) = sum_{i in shell s} grad_pos[i] . positions[i]
    grad_shell = np.zeros(len(shell_params))
    for s in range(len(shell_params)):
        mask = (shell_ids == s)
        grad_shell[s] = np.sum(grad_pos[mask] * positions[mask])
    
    grad = np.concatenate([[grad_global], grad_shell])
    
    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Construct the LJ38 global-minimum truncated-octahedron geometry using
    symmetry-adapted parameter optimization (global scale + shell breathing).
    """
    r0 = 2.0 ** (1.0 / 6.0)
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    # Initial guess: scale = r0, all breathing = 0
    x0 = np.array([r0] + [0.0] * n_shells)
    
    # Optimize symmetry-adapted parameters with analytical gradient
    try:
        res = minimize(
            lambda p: _symmetry_adapted_energy_and_grad(p, base_coords, shell_ids),
            x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16, 'maxls': 200}
        )
        e_opt = res.fun
        e_init = _symmetry_adapted_energy(x0, base_coords, shell_ids)
        if e_opt < e_init:
            opt_params = res.x
        else:
            opt_params = x0
    except Exception:
        opt_params = x0
    
    # Build final positions from optimized parameters
    global_scale = opt_params[0]
    shell_params = opt_params[1:]
    
    positions = np.zeros((N, 3))
    breath = np.ones(N)
    for i in range(N):
        s = shell_ids[i]
        if s < len(shell_params):
            breath[i] = 1.0 + shell_params[s]
    positions = base_coords * breath[:, np.newaxis] * global_scale
    
    # Re-center
    positions -= positions.mean(axis=0)
    
    return positions


def _make_icosahedral():
    """Construct an icosahedral arrangement for 38 atoms (1+12+25)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    r0 = 2.0 ** (1.0 / 6.0)

    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * r0

    pos_list = [np.zeros(3)]
    pos_list.extend(list(verts))

    # Add 25 more atoms on a second shell
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
    """Basin hopping with topology-swap strategy for icosahedral motifs."""
    best_pos = pos.copy()
    best_e = _lj_energy_and_grad(pos)[0]
    
    for hop in range(n_hops):
        if deadline is not None and time.time() > deadline:
            break
        
        # Detect icosahedral motifs
        ico_atoms = _detect_icosahedral_motifs(best_pos)
        
        if len(ico_atoms) > 0 and target_fcc is not None:
            # Apply topology-swap displacement
            disp = _topology_swap_displacement(best_pos, ico_atoms, target_fcc, rng)
            perturbed = best_pos + disp
        else:
            # Standard random perturbation
            sigma = rng.uniform(0.05, max_sigma)
            perturbed = _perturb(best_pos, sigma, rng)
        
        # Relax
        relaxed, e = _relax(perturbed, maxiter=5000, deadline=deadline,
                           gtol=1e-10, ftol=1e-14)
        
        # Accept if lower energy
        if e < best_e:
            best_e = e
            best_pos = relaxed.copy()
    
    return best_pos, best_e


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) FCC truncated octahedron with symmetry-adapted optimization: relax + polish
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

    # 3) Basin hopping from the best result with topology-swap
    rng = np.random.RandomState(12345)
    if time.time() < deadline - 20.0:
        try:
            # Get FCC target for topology-swap
            fcc_target = _make_fcc_truncated_octahedron()
            best_pos_bh, best_e_bh = _basin_hop(best_pos, rng, n_hops=25,
                                                 max_sigma=0.4, deadline=deadline,
                                                 target_fcc=fcc_target)
            if best_e_bh < best_e:
                best_e = best_e_bh
                best_pos = best_pos_bh.copy()
        except Exception:
            pass

    # 4) Multi-scale perturbations around the best FCC result
    base = _make_fcc_truncated_octahedron()
    
    # Adaptive perturbation schedule based on remaining time
    remaining = deadline - time.time()
    if remaining > 30:
        perturb_schedule = [
            (0.01, 8),
            (0.03, 8),
            (0.05, 8),
            (0.08, 8),
            (0.12, 6),
            (0.18, 6),
            (0.25, 4),
            (0.35, 4)
        ]
    elif remaining > 15:
        perturb_schedule = [
            (0.02, 5),
            (0.05, 5),
            (0.08, 5),
            (0.12, 4),
            (0.18, 4),
            (0.25, 3)
        ]
    else:
        perturb_schedule = [
            (0.03, 3),
            (0.08, 3),
            (0.12, 3)
        ]
    
    for sigma, count in perturb_schedule:
        if time.time() > deadline:
            break
        for _ in range(count):
            if time.time() > deadline:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=10000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 5) Random starts (catch-all) - only if time remains
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline:
                break
            pos_rand = rng.randn(N, 3) * 2.5
            pos_r, e = _relax(pos_rand, maxiter=5000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        return np.array(pts[:N], dtype=float) * 1.12

    # Final high-precision polish if time remains
    if time.time() < deadline - 3.0:
        pos_r, e = _relax(best_pos, maxiter=20000, deadline=deadline,
                          gtol=1e-14, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
            
    # One more ultra-high precision pass if time allows
    if time.time() < deadline - 1.0:
        pos_r, e = _relax(best_pos, maxiter=5000, deadline=deadline,
                          gtol=1e-15, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # Additional topology-swap passes if time remains
    if time.time() < deadline - 5.0:
        try:
            fcc_target = _make_fcc_truncated_octahedron()
            ico_atoms = _detect_icosahedral_motifs(best_pos)
            if len(ico_atoms) > 0:
                disp = _topology_swap_displacement(best_pos, ico_atoms, fcc_target, rng)
                perturbed = best_pos + disp
                pos_r, e = _relax(perturbed, maxiter=10000, deadline=deadline,
                                  gtol=1e-12, ftol=1e-16)
                if e < best_e:
                    best_e = e
                    best_pos = pos_r.copy()
        except Exception:
            pass

    # Ensure minimum pairwise distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
