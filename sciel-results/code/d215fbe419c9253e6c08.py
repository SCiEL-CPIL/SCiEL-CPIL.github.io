"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_TOTAL_TIME_LIMIT = 290.0  # safety margin before 300s kill
_START_TIME = None


def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)


def _validate(positions):
    """Check if positions are valid (finite, min distance >= 0.35)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _bcc_positions(n, scale=1.0):
    """Generate BCC lattice positions for n atoms."""
    # BCC has 2 atoms per conventional cell: (0,0,0) and (0.5,0.5,0.5)
    positions = []
    # Determine how many cells we need
    # Each cell contributes 2 atoms
    n_cells = (n + 1) // 2
    # Arrange cells in a compact arrangement
    # Use a simple cubic arrangement of cells
    side = int(np.ceil(n_cells ** (1/3)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if len(positions) >= n:
                    break
                # Corner atom
                p1 = np.array([ix, iy, iz]) * scale
                positions.append(p1)
                if len(positions) >= n:
                    break
                # Body center atom
                p2 = (np.array([ix, iy, iz]) + 0.5) * scale
                positions.append(p2)
                if len(positions) >= n:
                    break
        if len(positions) >= n:
            break
    return np.array(positions[:n])


def _fcc_positions(n, scale=1.0):
    """Generate FCC lattice positions for n atoms."""
    # FCC has 4 atoms per conventional cell
    positions = []
    n_cells = (n + 3) // 4
    side = int(np.ceil(n_cells ** (1/3)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if len(positions) >= n:
                    break
                base = np.array([ix, iy, iz]) * scale
                # FCC positions: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    if len(positions) >= n:
                        break
                    p = base + np.array([dx, dy, dz]) * scale
                    positions.append(p)
        if len(positions) >= n:
            break
    return np.array(positions[:n])


def _hcp_positions(n, scale=1.0):
    """Generate HCP lattice positions for n atoms."""
    # HCP has 6 atoms per conventional cell
    positions = []
    n_cells = (n + 5) // 6
    side = int(np.ceil(n_cells ** (1/3)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if len(positions) >= n:
                    break
                base = np.array([ix, iy, iz]) * scale
                # HCP positions (simplified)
                for dx, dy, dz in [(0, 0, 0), (0.5, 0, 0), (0.25, 0.433, 0),
                                   (0, 0, 0.816), (0.5, 0, 0.816), (0.25, 0.433, 0.816)]:
                    if len(positions) >= n:
                        break
                    p = base + np.array([dx, dy, dz]) * scale
                    positions.append(p)
        if len(positions) >= n:
            break
    return np.array(positions[:n])


def _random_compact(n, rng, scale=0.72):
    """Generate random compact cluster."""
    radius = scale * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * scale for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all atoms, add remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def construct(instance):
    """Find global minimum of Sutton-Chen cluster using multiple initialization strategies."""
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: BCC lattice initialization (measured to work well)
    if _time_left() > 60:
        bcc_pos = _bcc_positions(size, scale=0.72)
        if _validate(bcc_pos):
            energy, positions = relax(bcc_pos, potential, parameter)
            if _validate(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Strategy 2: FCC lattice initialization
    if _time_left() > 60:
        fcc_pos = _fcc_positions(size, scale=0.72)
        if _validate(fcc_pos):
            energy, positions = relax(fcc_pos, potential, parameter)
            if _validate(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Strategy 3: HCP lattice initialization
    if _time_left() > 60:
        hcp_pos = _hcp_positions(size, scale=0.72)
        if _validate(hcp_pos):
            energy, positions = relax(hcp_pos, potential, parameter)
            if _validate(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Strategy 4: Random compact clusters
    n_random = 8
    for i in range(n_random):
        if _time_left() < 15:
            break
        rand_pos = _random_compact(size, rng)
        if _validate(rand_pos):
            energy, positions = relax(rand_pos, potential, parameter)
            if _validate(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Strategy 5: Perturbed BCC - add small random perturbations to BCC
    if _time_left() > 30:
        for i in range(3):
            if _time_left() < 15:
                break
            bcc_pos = _bcc_positions(size, scale=0.72)
            perturbation = rng.normal(0, 0.1, bcc_pos.shape)
            perturbed = bcc_pos + perturbation
            if _validate(perturbed):
                energy, positions = relax(perturbed, potential, parameter)
                if _validate(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    
    # Strategy 6: Perturbed FCC
    if _time_left() > 30:
        for i in range(3):
            if _time_left() < 15:
                break
            fcc_pos = _fcc_positions(size, scale=0.72)
            perturbation = rng.normal(0, 0.1, fcc_pos.shape)
            perturbed = fcc_pos + perturbation
            if _validate(perturbed):
                energy, positions = relax(perturbed, potential, parameter)
                if _validate(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    
    # Final validation
    if best_positions is None or not _validate(best_positions):
        # Fallback: return a simple linear arrangement
        best_positions = np.array([[i, 0, 0] for i in range(size)], dtype=float)
    
    return best_positions
# EVOLVE-BLOCK-END
