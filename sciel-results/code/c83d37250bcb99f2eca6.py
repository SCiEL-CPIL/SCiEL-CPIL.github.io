"""Morse cluster global minima via basin hopping with time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
_DEADLINE = _START_TIME + 280.0  # 20s safety margin


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(coords, rho):
    """Compute Morse energy; return inf if any pair < 0.5 apart."""
    n = len(coords)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        return np.inf
    # Upper triangle
    iu = np.triu_indices(n, 1)
    r = dists[iu]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)


def _initial_cluster(n, rng, spacing=1.0):
    """Generate a compact initial cluster with minimum spacing."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.55 * spacing
    max_attempts = 100000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining with small random offsets
    while len(points) < n:
        base = points[rng.integers(len(points))] if points else np.zeros(3)
        p = base + rng.uniform(-0.1, 0.1, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping(n, rho, rng, time_limit, initial=None):
    """Run basin hopping for the given time limit. Returns best (energy, coords)."""
    start = time.time()
    
    if initial is not None:
        best_coords = initial.copy()
        best_energy, _ = relax(best_coords, 'morse', rho)
        if not np.isfinite(best_energy):
            best_energy = _morse_energy(best_coords, rho)
            if not np.isfinite(best_energy):
                # Try to fix by spreading out
                best_coords = _initial_cluster(n, rng)
                best_energy, _ = relax(best_coords, 'morse', rho)
    else:
        best_coords = _initial_cluster(n, rng)
        best_energy, _ = relax(best_coords, 'morse', rho)
        if not np.isfinite(best_energy):
            best_energy = _morse_energy(best_coords, rho)
    
    current_coords = best_coords.copy()
    current_energy = best_energy
    
    # Perturbation parameters
    base_step = 0.15
    step_scale = 1.0
    
    iteration = 0
    while _time_remaining() > 5.0:
        # Check per-instance time
        elapsed = time.time() - start
        if elapsed > time_limit:
            break
        
        # Perturb
        step = base_step * step_scale
        perturbed = current_coords + rng.normal(0, step, (n, 3))
        
        # Relax
        energy, relaxed = relax(perturbed, 'morse', rho)
        
        if not np.isfinite(energy):
            energy = _morse_energy(relaxed, rho)
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_coords = relaxed.copy()
            current_coords = relaxed.copy()
            current_energy = energy
            # Occasionally reduce step size
            if iteration % 20 == 0:
                step_scale *= 0.95
        else:
            # Accept with Metropolis criterion for exploration
            if np.isfinite(energy):
                delta_e = energy - current_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / (0.1 + 0.05 * step_scale)):
                    current_coords = relaxed.copy()
                    current_energy = energy
                    if energy < best_energy:
                        best_energy = energy
                        best_coords = relaxed.copy()
        
        iteration += 1
        
        # Periodically try larger perturbations to escape local minima
        if iteration % 50 == 0:
            step_scale = max(0.3, step_scale * 1.2)
    
    return best_energy, best_coords


def construct(instance):
    """Find global minimum of Morse cluster via basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Allocate time based on cluster size
    # Smaller clusters: more iterations possible; larger: fewer
    # Base allocation: larger clusters get more total time
    if n <= 40:
        time_limit = 40.0
    elif n <= 60:
        time_limit = 50.0
    else:
        time_limit = 60.0
    
    # Use a fixed seed for reproducibility but vary per instance
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    best_energy = np.inf
    best_coords = None
    
    # Multiple independent basin hopping runs
    n_runs = 3
    for run in range(n_runs):
        if _time_remaining() < 10.0:
            break
        
        # Different initial conditions per run
        if run == 0:
            initial = _initial_cluster(n, rng)
        else:
            # Perturb the best found so far for subsequent runs
            if best_coords is not None:
                initial = best_coords + rng.normal(0, 0.3, (n, 3))
            else:
                initial = _initial_cluster(n, rng)
        
        energy, coords = _basin_hopping(n, rho, rng, time_limit, initial=initial)
        
        if energy < best_energy:
            best_energy = energy
            best_coords = coords
        
        # If we found a very good result, reduce remaining runs
        if best_energy < -n * 0.95:
            break
    
    # Final polish: do a few more relaxation steps with tighter tolerance
    if best_coords is not None and _time_remaining() > 3.0:
        # Try a final high-precision relaxation
        for _ in range(3):
            if _time_remaining() < 2.0:
                break
            perturbed = best_coords + rng.normal(0, 0.05, (n, 3))
            energy, relaxed = relax(perturbed, 'morse', rho)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_coords = relaxed
    
    if best_coords is None:
        # Fallback: just do a simple relaxation
        coords = _initial_cluster(n, rng)
        _, best_coords = relax(coords, 'morse', rho)
    
    return best_coords
# EVOLVE-BLOCK-END
