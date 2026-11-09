"""Sutton-Chen cluster global minima via Simulated Annealing with Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    # Vectorized distance check for speed
    # Compute pairwise distances efficiently for large n? 
    # For n<=80, O(n^2) is fine.
    d = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if d.min() < 0.35:
        return False
    return True

def _fcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]

def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 3
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]

def _icosahedral_shells(n, r1=1.0, r2=1.27, r3=1.51):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * r1)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * r2
                pts.append(mid)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * r3
                    pts.append(fc)
    if len(pts) < n:
        rng = np.random.default_rng(42)
        while len(pts) < n:
            v = verts[rng.integers(0, len(verts))]
            v2 = verts[rng.integers(0, len(verts))]
            p = (v + v2) / 2
            p = p / np.linalg.norm(p) * (r3 + 0.3)
            pts.append(p)
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]

def _random_compact(n, seed, min_dist=0.75):
    rng = np.random.default_rng(seed)
    max_radius = 0.85 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 500:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < min_dist:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    return np.array(pts)

def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))

    # 1. Generate a small set of high-quality diverse seeds
    seeds = [
        _fcc_cluster(n, lattice=0.72),
        _fcc_cluster(n, lattice=0.70),
        _hcp_cluster(n, lattice=0.72),
        _icosahedral_shells(n),
        _random_compact(n, seed=42, min_dist=0.75),
    ]

    # 2. Relax all seeds to get initial local minima
    best_e = np.inf
    best_pos = None
    current_pos = None
    current_e = np.inf

    for s in seeds:
        if time.time() > deadline:
            break
        if not _validate_positions(s):
            continue
        try:
            e, p = relax(s, potential, kind)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
                if e < current_e:
                    current_e = e
                    current_pos = p.copy()
        except Exception:
            pass

    if current_pos is None:
        # Fallback
        pos = _random_compact(n, seed=999, min_dist=0.8)
        try:
            current_e, current_pos = relax(pos, potential, kind)
            if np.isfinite(current_e) and _validate_positions(current_pos):
                best_e = current_e
                best_pos = current_pos.copy()
        except Exception:
            return pos

    # 3. Simulated Annealing / Basin Hopping Loop
    # We use a geometric cooling schedule.
    # T starts high to allow large exploratory moves, decays to 0 for fine polishing.
    
    # Estimate energy scale
    # Typical SC energies for n~50-80 are in the range of -100 to -200.
    # Let's set T_initial relative to the current energy difference or a fixed heuristic.
    T = 5.0  # Initial temperature (energy units)
    T_min = 1e-4
    alpha = 0.95  # Cooling rate per step
    
    # Perturbation magnitude
    # Start with large perturbations, decay them.
    sigma = 0.5
    sigma_min = 0.05
    sigma_decay = 0.98

    rng = np.random.default_rng(12345)

    while time.time() < deadline - 10.0:
        # Perturb current position
        # Random displacement
        disp = rng.normal(0, sigma, size=(n, 3))
        new_pos = current_pos + disp
        
        # Check validity
        if not _validate_positions(new_pos):
            # If invalid, try smaller perturbation or skip
            # For simplicity, skip and cool slightly
            T *= alpha
            sigma *= sigma_decay
            continue

        # Relax the perturbed position
        try:
            new_e, new_relaxed = relax(new_pos, potential, kind)
        except Exception:
            new_e = np.inf
            new_relaxed = new_pos

        if not np.isfinite(new_e) or not _validate_positions(new_relaxed):
            T *= alpha
            sigma *= sigma_decay
            continue

        # Metropolis Criterion
        delta_e = new_e - current_e
        if delta_e < 0:
            # Accept downhill move
            current_e = new_e
            current_pos = new_relaxed
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        else:
            # Accept uphill move with probability exp(-delta_e / T)
            if T > 1e-6:
                if rng.random() < np.exp(-delta_e / T):
                    current_e = new_e
                    current_pos = new_relaxed

        # Cooling
        T *= alpha
        sigma *= sigma_decay
        
        if T < T_min:
            # If temperature is very low, restart with a higher T to escape local minimum?
            # Or just stop if we've cooled enough?
            # Let's do a "reheat" to escape deep local minima.
            # Reheat to 50% of initial T
            T = 0.5 * 5.0 
            sigma = 0.3
            # Also, if we haven't improved in a long time, we might be stuck.
            # The reheat helps.

    # 4. Final Polish
    if time.time() < deadline - 2.0:
        try:
            final_e, final_pos = relax(best_pos, potential, kind)
            if np.isfinite(final_e) and _validate_positions(final_pos):
                best_e = final_e
                best_pos = final_pos
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
