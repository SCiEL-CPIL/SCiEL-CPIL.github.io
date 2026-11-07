"""Sutton-Chen transition-metal cluster global minima - deterministic growth algorithm."""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline: 320s (10s safety margin before 330s kill)
_DEADLINE = time.time() + 320.0


def _generate_interstitial_candidates(positions, n_candidates=200):
    """
    Generate candidate interstitial sites around existing positions.
    Returns array of shape (n_candidates, 3).
    """
    n_atoms = len(positions)
    if n_atoms == 0:
        return np.array([[0.0, 0.0, 0.0]])
    
    candidates = []
    
    # Strategy 1: Points near existing atoms (offset in various directions)
    # Use icosahedral directions for good coverage
    phi = (1 + np.sqrt(5)) / 2.0
    ico_dirs = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_dirs = ico_dirs / np.linalg.norm(ico_dirs, axis=1, keepdims=True)
    
    # Also add cardinal directions
    cardinal_dirs = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    all_dirs = np.vstack([ico_dirs, cardinal_dirs])
    
    # Distances to try for interstitial sites
    distances = [0.8, 1.0, 1.2, 1.5, 2.0]
    
    # For each existing atom, generate candidates at various distances and directions
    for i in range(n_atoms):
        for dist in distances:
            for dir_vec in all_dirs:
                candidate = positions[i] + dist * dir_vec
                candidates.append(candidate)
    
    # Strategy 2: Midpoints between pairs of nearby atoms
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 3.0:  # Only for reasonably close pairs
                midpoint = (positions[i] + positions[j]) / 2.0
                candidates.append(midpoint)
    
    # Strategy 3: Random points in a shell around the cluster
    center = np.mean(positions, axis=0)
    max_dist = np.max(np.linalg.norm(positions - center, axis=1))
    shell_radius = max_dist + 0.5
    for _ in range(50):
        # Random direction
        theta = np.random.uniform(0, 2 * np.pi)
        cos_phi = np.random.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        direction = np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        candidate = center + shell_radius * direction
        candidates.append(candidate)
    
    candidates = np.array(candidates)
    
    # Filter out candidates that are too close to existing atoms (< 0.4)
    valid_mask = np.ones(len(candidates), dtype=bool)
    for i in range(n_atoms):
        dists = np.linalg.norm(candidates - positions[i], axis=1)
        valid_mask &= (dists > 0.4)
    
    candidates = candidates[valid_mask]
    
    # If we have too many, randomly subsample
    if len(candidates) > n_candidates:
        idx = np.random.choice(len(candidates), n_candidates, replace=False)
        candidates = candidates[idx]
    
    # If we don't have enough, add more random points
    if len(candidates) < n_candidates:
        extra = np.random.uniform(-2, 2, (n_candidates - len(candidates), 3))
        candidates = np.vstack([candidates, extra])
    
    return candidates[:n_candidates]


def _evaluate_candidate(candidate, positions, potential, kind):
    """
    Evaluate the energy change of adding a candidate atom to the cluster.
    Returns (energy_change, new_positions) or (inf, None) if invalid.
    """
    new_positions = np.vstack([positions, candidate])
    n = len(new_positions)
    
    # Quick check: minimum distance
    min_dist = np.inf
    for i in range(n - 1):
        d = np.linalg.norm(new_positions[i] - new_positions[n - 1])
        if d < min_dist:
            min_dist = d
    
    if min_dist < 0.35:
        return np.inf, None
    
    # Compute full energy
    try:
        energy, _ = sutton_chen_energy_gradient(new_positions, kind)
        if not np.isfinite(energy):
            return np.inf, None
        return energy, new_positions
    except Exception:
        return np.inf, None


def _grow_cluster(n_target, potential, kind, seed_positions=None, rng=None):
    """
    Deterministically grow a cluster from seed to n_target atoms.
    At each step, find the lowest-energy interstitial site.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Start with seed positions or a single atom
    if seed_positions is not None and len(seed_positions) > 0:
        positions = seed_positions.copy()
    else:
        positions = np.array([[0.0, 0.0, 0.0]])
    
    current_n = len(positions)
    
    # Get initial energy
    try:
        current_energy, _ = sutton_chen_energy_gradient(positions, kind)
    except Exception:
        current_energy = 0.0
    
    while current_n < n_target:
        if time.time() > _DEADLINE - 15:
            break
        
        # Generate interstitial candidates
        candidates = _generate_interstitial_candidates(positions, n_candidates=150)
        
        # Evaluate each candidate
        best_energy_change = np.inf
        best_candidate = None
        best_new_positions = None
        
        for candidate in candidates:
            if time.time() > _DEADLINE - 15:
                break
            
            energy, new_positions = _evaluate_candidate(candidate, positions, potential, kind)
            
            if energy < best_energy_change:
                best_energy_change = energy
                best_candidate = candidate
                best_new_positions = new_positions
        
        if best_new_positions is None:
            # Fallback: add atom at a random position far from cluster
            center = np.mean(positions, axis=0)
            max_dist = np.max(np.linalg.norm(positions - center, axis=1))
            new_pos = center + (max_dist + 1.0) * rng.normal(0, 1, 3)
            best_new_positions = np.vstack([positions, new_pos])
        else:
            best_new_positions = best_new_positions
        
        # Relax the new configuration
        try:
            energy, relaxed_positions = relax(best_new_positions, potential, kind)
            if np.all(np.isfinite(relaxed_positions)):
                positions = relaxed_positions
                current_energy = energy
            else:
                positions = best_new_positions
        except Exception:
            positions = best_new_positions
        
        current_n = len(positions)
    
    return positions


def _seed_mackay(n):
    """Generate a Mackay icosahedral seed for small n."""
    phi = (1 + np.sqrt(5)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    R1 = 1.0 / (2.0 * np.sin(np.pi / 5.0))
    
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    elif n <= 13:
        # Center + icosahedron
        pts = np.vstack([np.array([[0.0, 0.0, 0.0]]), ico_verts * R1])
        return pts[:n]
    else:
        # Just use center + icosahedron as seed
        pts = np.vstack([np.array([[0.0, 0.0, 0.0]]), ico_verts * R1])
        return pts


def _seed_fcc(n):
    """Generate an FCC seed for small n."""
    a = np.sqrt(2.0)
    points = []
    nmax = 3
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for k in range(-nmax, nmax + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:min(n, len(points))]
    return points[idx]


def construct(instance):
    """
    Deterministic growth algorithm for Sutton-Chen clusters.
    Builds clusters atom-by-atom, selecting the lowest-energy interstitial site at each step.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate multiple seeds with different starting structures
    seeds = []
    
    # Mackay icosahedral seed
    try:
        mackay_seed = _seed_mackay(min(size, 13))
        if len(mackay_seed) > 0:
            seeds.append(mackay_seed)
    except Exception:
        pass
    
    # FCC seed
    try:
        fcc_seed = _seed_fcc(min(size, 13))
        if len(fcc_seed) > 0:
            seeds.append(fcc_seed)
    except Exception:
        pass
    
    # Single atom seed
    seeds.append(np.array([[0.0, 0.0, 0.0]]))
    
    # Grow from each seed and keep the best result
    best_energy = np.inf
    best_positions = None
    
    for seed in seeds:
        if time.time() > _DEADLINE - 20:
            break
        
        try:
            grown = _grow_cluster(size, potential, kind, seed_positions=seed, rng=rng)
            
            if len(grown) < size:
                # Pad if growth was interrupted
                while len(grown) < size:
                    center = np.mean(grown, axis=0)
                    max_dist = np.max(np.linalg.norm(grown - center, axis=1))
                    new_pos = center + (max_dist + 0.5) * rng.normal(0, 1, 3)
                    grown = np.vstack([grown, new_pos])
            
            grown = grown[:size]
            
            # Final relaxation
            energy, relaxed = relax(grown, potential, kind)
            if np.all(np.isfinite(relaxed)):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Fallback if all seeds failed
    if best_positions is None:
        # Use a simple compact random arrangement
        radius = 0.8 * size ** (1.0 / 3.0)
        best_positions = rng.uniform(-radius, radius, (size, 3))
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except Exception:
            pass
    
    # Final safety check: ensure no atoms are too close
    if best_positions is not None and len(best_positions) == size:
        n = len(best_positions)
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < min_dist:
                    min_dist = d
        
        if min_dist < 0.35:
            # Spread out atoms that are too close
            for _ in range(10):
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(best_positions[i] - best_positions[j])
                        if d < 0.35:
                            direction = best_positions[i] - best_positions[j]
                            if np.linalg.norm(direction) < 1e-10:
                                direction = rng.normal(0, 1, 3)
                            direction = direction / np.linalg.norm(direction)
                            best_positions[i] += 0.2 * direction
                            best_positions[j] -= 0.2 * direction
                
                # Re-check
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(best_positions[i] - best_positions[j])
                        if d < min_dist:
                            min_dist = d
                if min_dist >= 0.35:
                    break
    
    return best_positions
