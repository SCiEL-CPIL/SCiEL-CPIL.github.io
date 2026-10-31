"""LJ38 cluster: FCC core+outer shell construction with basin hopping and polish."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    e, _ = _lj_energy_and_grad(pos)
    return e

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    _, g = _lj_energy_and_grad(pos)
    return g

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    Using proper FCC primitive vectors: a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)
    """
    # FCC primitive vectors
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 6.5:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    # Remove duplicates (FCC lattice can have coincident points from different representations)
    # Sort and deduplicate
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    core_pts = pts[pts[:, 3] <= 4.1]
    outer_pts = pts[(pts[:, 3] > 4.1) & (pts[:, 3] <= 6.1)]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    pts.append(np.array([0.25, 0.25, 0.25]))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _optimize_scale(pos):
    """
    Optimize the global scale of the cluster to minimize energy.
    This is a 1D optimization problem.
    """
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    # Initial scale is 1.0. We expect the optimal scale to be close to 1.0.
    # The LJ minimum is at r = 2^(1/6) ~ 1.122.
    # If the initial structure is too compressed, scale > 1.
    # If too expanded, scale < 1.
    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-8})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairwise distances are at least min_sep."""
    for _ in range(100):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=1)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        mask = r < min_sep
        if not np.any(mask):
            break
        # Push apart atoms that are too close
        for i in range(N):
            for j in range(i+1, N):
                if mask[i, j]:
                    d = pos[j] - pos[i]
                    r = np.linalg.norm(d)
                    if r < 1e-10:
                        d = np.array([0.01, 0.01, 0.01])
                        r = np.linalg.norm(d)
                    d = d / r
                    push = (min_sep - r) / 2.0
                    pos[i] -= push * d
                    pos[j] += push * d
    return pos

def _get_fcc_target_radii():
    """
    Get the target shell radii for FCC truncated octahedron with 38 atoms.
    Shell 0: 1 atom at r=0
    Shell 1: 12 atoms at r=sqrt(2) (110 type)
    Shell 2: 6 atoms at r=2 (200 type)
    Shell 3: 19 atoms at r=sqrt(6) (211 type)
    Total: 38 atoms
    """
    # In reduced LJ units, the FCC lattice parameter a is such that
    # nearest neighbor distance = a/sqrt(2) = 2^(1/6)
    # So a = sqrt(2) * 2^(1/6)
    a = np.sqrt(2.0) * (2.0 ** (1.0/6.0))
    
    # Shell radii in terms of a:
    # r0 = 0
    # r1 = a*sqrt(2)/2 = a/sqrt(2) = 2^(1/6)  (12 atoms)
    # r2 = a  (6 atoms)
    # r3 = a*sqrt(6)/2 = a*sqrt(6)/2  (19 atoms)
    
    r0 = 0.0
    r1 = a / np.sqrt(2.0)  # = 2^(1/6)
    r2 = a
    r3 = a * np.sqrt(6.0) / 2.0
    
    return [r0, r1, r2, r3], [1, 12, 6, 19]

def _geometric_objective(pos):
    """
    Smooth geometric objective that penalizes deviations from FCC shell structure.
    This helps navigate to the vicinity of the global minimum without getting
    trapped in the icosahedral funnel.
    """
    target_radii, target_counts = _get_fcc_target_radii()
    
    # Compute distances from centroid
    centroid = np.mean(pos, axis=0)
    dists = np.linalg.norm(pos - centroid, axis=1)
    
    # For each atom, find the closest target shell and compute penalty
    # Use a soft assignment: each atom contributes to all shells with weight based on distance
    penalty = 0.0
    for r_target, count_target in zip(target_radii, target_counts):
        # Distance of each atom from this shell radius
        d = dists - r_target
        # Soft assignment: atoms closer to this shell contribute more
        # Use a Gaussian kernel with width ~0.1
        width = 0.1
        weights = np.exp(-0.5 * (d / width) ** 2)
        
        # The shell should have exactly count_target atoms
        # Penalize if the "effective count" (sum of weights) deviates from target
        effective_count = np.sum(weights)
        count_penalty = (effective_count - count_target) ** 2
        
        # Also penalize atoms that are far from all shells
        penalty += count_penalty * 0.1
    
    # Additional penalty: encourage atoms to be near one of the target radii
    for i in range(len(pos)):
        min_d2 = min((dists[i] - r) ** 2 for r in target_radii)
        penalty += 0.5 * min_d2
    
    return penalty

def _geometric_gradient(pos):
    """
    Gradient of the geometric objective.
    """
    target_radii, target_counts = _get_fcc_target_radii()
    
    centroid = np.mean(pos, axis=0)
    dists = np.linalg.norm(pos - centroid, axis=1)
    
    grad = np.zeros_like(pos)
    
    # Gradient of the shell count penalty
    for r_target, count_target in zip(target_radii, target_counts):
        d = dists - r_target
        width = 0.1
        weights = np.exp(-0.5 * (d / width) ** 2)
        effective_count = np.sum(weights)
        
        # d(effective_count)/d(d_i) = -d_i/width^2 * weights_i
        d_eff_count_dd = -d / (width ** 2) * weights
        
        # d(penalty)/d(d_i) = 2 * (effective_count - count_target) * d_eff_count_dd_i
        # d(penalty)/d(pos_i) = d(penalty)/d(d_i) * d(d_i)/d(pos_i)
        # d(d_i)/d(pos_i) = (pos_i - centroid) / d_i
        
        for i in range(len(pos)):
            if dists[i] > 1e-10:
                d_pos = (pos[i] - centroid) / dists[i]
                grad[i] += 0.1 * 2 * (effective_count - count_target) * d_eff_count_dd[i] * d_pos
    
    # Gradient of the "near a shell" penalty
    for i in range(len(pos)):
        # Find closest shell
        min_d = min(abs(dists[i] - r) for r in target_radii)
        closest_r = min(target_radii, key=lambda r: abs(dists[i] - r))
        
        if dists[i] > 1e-10:
            d_pos = (pos[i] - centroid) / dists[i]
            # d(min_d2)/d(d_i) = 2 * (d_i - closest_r)
            grad[i] += 0.5 * 2 * (dists[i] - closest_r) * d_pos
    
    # Subtract centroid gradient contribution
    grad -= np.sum(grad, axis=0, keepdims=True) / len(pos)
    
    return grad

def _topology_forcing_relax(pos, max_iter=1000, gtol=1e-10):
    """
    First phase: minimize the geometric objective to get close to FCC structure.
    """
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e = _geometric_objective(p)
        g = _geometric_gradient(p)
        return e, g.ravel()
    
    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    return p_out

def _make_truncated_octahedron_start():
    """
    Create a starting geometry that closely resembles the FCC truncated octahedron.
    Uses proper FCC lattice generation.
    """
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 20.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    # Sort by distance from origin
    pts = pts[np.argsort(pts[:, 3])]
    
    # Take the 38 closest points
    selected = pts[:38, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    # Scale so that nearest neighbor distance is ~2^(1/6)
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
    
    return selected

def _basin_hop(pos, rng, max_iters=100, T=0.1, t0=1.0, t1=0.01):
    """
    Basin hopping: perturb, relax, accept/reject by Metropolis criterion.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    for i in range(max_iters):
        # Perturbation: add Gaussian noise
        noise_scale = T
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax
        relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-9)
        e_relaxed, _ = _lj_energy_and_grad(relaxed)
        
        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            # Temperature annealing
            T_curr = t0 * (t1 / t0) ** (i / max_iters)
            accept = rng.random() < np.exp(-delta_e / T_curr)
        
        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    # Phase 1: Start with the full 38-atom FCC truncated octahedron
    best_pos = _make_truncated_octahedron_start()
    best_pos, _ = _optimize_scale(best_pos)
    best_pos = _ensure_min_sep(best_pos)
    
    # Phase 2: Topology forcing - minimize geometric objective to lock in FCC structure
    if time.time() < deadline - 60.0:
        best_pos = _topology_forcing_relax(best_pos, max_iter=500, gtol=1e-10)
        best_pos = _ensure_min_sep(best_pos)
    
    # Phase 3: Energy relaxation
    best_pos, best_e = _relax(best_pos, max_iter=500, gtol=1e-10)
    
    # Phase 4: Basin hopping with topology-aware perturbations
    if time.time() < deadline - 30.0:
        best_pos, best_e = _basin_hop(best_pos, rng, max_iters=50, T=0.1, t0=1.0, t1=0.01)
    
    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        best_pos, best_e = _relax(best_pos, max_iter=3000, gtol=1e-13)
    
    # Ensure minimum separation
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
