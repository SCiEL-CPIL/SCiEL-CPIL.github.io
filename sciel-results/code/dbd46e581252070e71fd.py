"""Morse cluster global minima: diverse lattice seeds + basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 290.0
    return _T0


def _time_left():
    global _HARD_DEADLINE
    return max(0.0, _HARD_DEADLINE - time.time())


def _enforce_min_dist(P, min_d=0.5):
    """Vectorized: ensure all pair distances >= min_d by local pushes."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    for _ in range(40):
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        mask = dists < min_d
        if not np.any(mask):
            break
        # Only process upper triangle to avoid double-counting
        iu, ju = np.where(np.triu(mask, 1))
        for i, j in zip(iu, ju):
            d = P[i] - P[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                d = np.array([1.0, 0.0, 0.0])
            else:
                d = d / dist
            push = (min_d - dist) / 2.0 + 0.005
            P[i] += push * d
            P[j] -= push * d
    return P


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice, centered."""
    pts = []
    a = spacing
    basis = [np.array([0, 0, 0]), np.array([0, a/2, a/2]),
             np.array([a/2, 0, a/2]), np.array([a/2, a/2, 0])]
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                base = np.array([i, j, k]) * a
                for b in basis:
                    pts.append(base + b)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP lattice, centered."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * (k + (1.0/3.0) * ((j % 2) + 1))
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                base = np.array([i, j, k]) * a
                pts.append(base)
                pts.append(base + np.array([a/2, a/2, a/2]))
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice, centered."""
    pts = []
    a = spacing
    R = int(spacing * (n ** (1.0/3.0) + 2)) + 3
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                pts.append(np.array([i, j, k]) * a)
    pts = np.array(pts)
    center = pts.mean(axis=0)
    d2 = np.sum((pts - center)**2, axis=1)
    idx = np.argsort(d2)[:n]
    pts = pts[idx] - pts[idx].mean(axis=0)
    return pts


def _icosahedral_shells(n):
    """Generate points on nested icosahedral shells (good for small clusters)."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    shells = [1.0, 1.45, 1.85, 2.2, 2.5, 2.8, 3.1]
    for s in shells:
        if len(pts) >= n:
            break
        shell_pts = ico * s
        # Add rotated copies for more points per shell
        rotations = [np.eye(3),
                     np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                     np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])]
        for rot in rotations:
            rotated = shell_pts @ rot.T
            for p in rotated:
                if len(pts) >= n:
                    break
                pts.append(p)
    pts = np.array(pts[:n])
    return pts


def _random_compact(n, spacing=1.0, seed=42):
    """Generate a random compact cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    max_tries = 50000
    tries = 0
    while len(points) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= 0.52 * spacing:
                points.append(p)
        tries += 1
    if len(points) < n:
        for i in range(n - len(points)):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            r = radius * 0.9
            p = np.array([r * np.sin(phi) * np.cos(theta),
                          r * np.sin(phi) * np.sin(theta),
                          r * np.cos(phi)])
            points.append(p)
    return np.array(points)


def _perturb(P, scale, rng):
    """Add Gaussian noise to positions."""
    return P + rng.normal(0, scale, P.shape)


def _relax_one(P, potential, parameter):
    """Relax a single configuration, returning (energy, positions)."""
    try:
        energy, positions = relax(P, potential, parameter)
        return energy, positions
    except Exception:
        return np.inf, P


def _langevin_refine(P, potential, parameter, n_steps=3, kick_scale=0.05, rng=None):
    """Short Langevin-style local refinement: small random kicks + relax."""
    if rng is None:
        rng = np.random.default_rng()
    best_E, best_P = np.inf, P.copy()
    for _ in range(n_steps):
        if _time_left() < 2.0:
            break
        P_kick = P + rng.normal(0, kick_scale, P.shape)
        P_kick = _enforce_min_dist(P_kick, 0.5)
        E, P_new = _relax_one(P_kick, potential, parameter)
        if E < best_E:
            best_E, best_P = E, P_new.copy()
        # Accept with probability 1 if better, else keep trying
        P = P_new
    return best_E, best_P


def _parallel_basin_hopping(chains, potential, parameter, total_time_budget):
    """Run multiple independent basin-hopping chains in parallel (sequential but diverse)."""
    n_chains = len(chains)
    # Each chain gets an equal share of the time budget
    time_per_chain = total_time_budget / n_chains
    chain_rngs = [np.random.default_rng(1000 + i * 7919) for i in range(n_chains)]
    
    # Kick schedules: each chain has a slightly different cooling
    base_schedules = [
        [0.35, 0.28, 0.22, 0.16, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01],
        [0.30, 0.25, 0.18, 0.14, 0.10, 0.07, 0.04, 0.025, 0.015, 0.01],
        [0.40, 0.30, 0.20, 0.15, 0.10, 0.06, 0.04, 0.02, 0.015, 0.01],
        [0.25, 0.20, 0.15, 0.11, 0.08, 0.05, 0.03, 0.02, 0.01, 0.008],
    ]
    
    global_best_E = np.inf
    global_best_P = None
    
    for ci, chain_init in enumerate(chains):
        if _time_left() < 5.0:
            break
        
        chain_start = time.time()
        chain_deadline = chain_start + time_per_chain
        
        # Initialize this chain
        P = _enforce_min_dist(chain_init, 0.5)
        E, P = _relax_one(P, potential, parameter)
        chain_best_E, chain_best_P = E, P.copy()
        
        if E < global_best_E:
            global_best_E, global_best_P = E, P.copy()
        
        rng = chain_rngs[ci]
        schedule = base_schedules[ci % len(base_schedules)]
        
        for scale in schedule:
            if time.time() > chain_deadline or _time_left() < 3.0:
                break
            # Number of hops at this scale
            n_hops = 3 if scale > 0.15 else (4 if scale > 0.05 else 5)
            for hop in range(n_hops):
                if time.time() > chain_deadline or _time_left() < 3.0:
                    break
                # Basin hop
                kick = scale * (0.7 + 0.6 * rng.random())
                P_kick = _perturb(chain_best_P, kick, rng)
                P_kick = _enforce_min_dist(P_kick, 0.5)
                E_kick, P_kick = _relax_one(P_kick, potential, parameter)
                
                # Langevin refinement around the new point
                E_ref, P_ref = _langevin_refine(P_kick, potential, parameter, 
                                                 n_steps=2, kick_scale=0.03, rng=rng)
                
                # Accept if better (Metropolis-like: always accept better)
                if E_ref < chain_best_E:
                    chain_best_E, chain_best_P = E_ref, P_ref.copy()
                    if E_ref < global_best_E:
                        global_best_E, global_best_P = E_ref, P_ref.copy()
        
        # End-of-chain: try a few extra large exploratory kicks if time allows
        if time.time() < chain_deadline and _time_left() > 5.0:
            for extra_scale in [0.5, 0.65]:
                if time.time() > chain_deadline or _time_left() < 4.0:
                    break
                P_kick = _perturb(chain_best_P, extra_scale, rng)
                P_kick = _enforce_min_dist(P_kick, 0.5)
                E_kick, P_kick = _relax_one(P_kick, potential, parameter)
                if E_kick < chain_best_E:
                    chain_best_E, chain_best_P = E_kick, P_kick.copy()
                    if E_kick < global_best_E:
                        global_best_E, global_best_P = E_kick, P_kick.copy()
    
    return global_best_E, global_best_P


def construct(instance):
    """Multi-chain parallel basin hopping with Langevin refinement."""
    _init_timer()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 6))

    rng = np.random.default_rng(size * 1000 + int(parameter))

    # Phase 1: diverse lattice + random initializations to seed chains
    candidates = []
    candidates.append(_fcc_points(size, 1.0))
    candidates.append(_hcp_points(size, 1.0))
    candidates.append(_bcc_points(size, 1.0))
    candidates.append(_sc_points(size, 1.0))

    # Icosahedral shells for smaller clusters
    if size <= 60:
        candidates.append(_icosahedral_shells(size))

    # Random compact with varied seeds
    for seed in [size, size + 7, size + 13, size + 23, size + 42, size + 99]:
        candidates.append(_random_compact(size, 1.0, seed))

    # Perturbed FCC variants
    for scale in [0.08, 0.15, 0.25]:
        fcc_pert = _fcc_points(size, 1.0) + rng.normal(0, scale, (size, 3))
        candidates.append(fcc_pert)

    # Perturbed HCP variants
    for scale in [0.1, 0.2]:
        hcp_pert = _hcp_points(size, 1.0) + rng.normal(0, scale, (size, 3))
        candidates.append(hcp_pert)

    # Relax all candidates to get initial energies
    relaxed_candidates = []
    best_energy = np.inf
    best_positions = None

    for i, cand in enumerate(candidates):
        if _time_left() < 5.0:
            break
        cand = _enforce_min_dist(cand, 0.5)
        energy, positions = _relax_one(cand, potential, parameter)
        relaxed_candidates.append((energy, positions))
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()

    # Select top diverse seeds for parallel chains
    # Sort by energy, pick top 4 with some diversity
    relaxed_candidates.sort(key=lambda x: x[0])
    
    # Ensure diversity: pick top 4, but if two are very close in energy, skip one
    selected_chains = []
    selected_energies = []
    for E, P in relaxed_candidates:
        if len(selected_chains) >= 4:
            break
        # Check diversity: distance from already-selected positions
        if selected_chains:
            min_dist = min(np.linalg.norm(P - sp) for sp in selected_chains)
            if min_dist < 0.3 and len(selected_chains) >= 2:
                continue
        selected_chains.append(P.copy())
        selected_energies.append(E)
    
    # If we didn't get 4 diverse ones, fill with best remaining
    while len(selected_chains) < 4 and len(relaxed_candidates) > 0:
        for E, P in relaxed_candidates:
            if P not in selected_chains and len(selected_chains) < 4:
                selected_chains.append(P.copy())
                break
        else:
            break

    # Phase 2: Parallel basin hopping with Langevin refinement
    if _time_left() > 10.0:
        time_budget = _time_left() - 8.0  # leave safety margin
        bh_best_E, bh_best_P = _parallel_basin_hopping(
            selected_chains, potential, parameter, time_budget
        )
        if bh_best_P is not None and bh_best_E < best_energy:
            best_energy = bh_best_E
            best_positions = bh_best_P

    # Phase 3: Final polish with very small kicks on the global best
    if best_positions is not None and _time_left() > 4.0:
        for _ in range(5):
            if _time_left() < 2.0:
                break
            perturbed = _perturb(best_positions, 0.01, rng)
            perturbed = _enforce_min_dist(perturbed, 0.5)
            energy, positions = _relax_one(perturbed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()

    if best_positions is None:
        best_positions = _enforce_min_dist(candidates[0], 0.5)

    return best_positions
# EVOLVE-BLOCK-END
