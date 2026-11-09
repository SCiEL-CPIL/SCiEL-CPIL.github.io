"""
Strategy: Basin hopping with polytetrahedral (icosahedral) seed structures for Dzugutov clusters.

The Dzugutov potential favours polytetrahedral order. We generate diverse icosahedral
seeds (Mackay supericosahedra, truncated icosahedra, random PT perturbations) and run
multiple parallel basin-hopping chains with local relaxation via the provided relax function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Global deadline tracking
_START_TIME = time.time()
_HARD_DEADLINE = 290.0  # Leave 10s safety margin before 300s limit
_MAX_THREADS = 4


def _remaining_time():
    return _HARD_DEADLINE - (time.time() - _START_TIME)


def _energy(positions, potential='dzugutov'):
    """Compute Dzugutov energy directly."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            if r < b:
                E += B * np.exp(d / (r - b))
    return E


def _is_valid(positions, min_dist=0.5):
    """Check all interatomic distances >= min_dist."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _generate_icosahedron(n, rng):
    """
    Generate an icosahedral-like structure with n atoms.
    Uses concentric icosahedral shells and fills remaining atoms.
    """
    # Icosahedral vertices (12 vertices of a regular icosahedron)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # Normalize to unit radius
    
    # Face centers of icosahedron (20 faces) - approximate positions
    # For simplicity, use a second shell at ~1.9 radius with 20 points
    # Approximate: use vertices of a dodecahedron or just scaled icosahedron
    shell2_verts = verts * 1.9  # 12 more at larger radius (approximation)
    
    # Build structure
    positions = []
    # Center atom
    if n >= 1:
        positions.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: 12 atoms at radius r1
    r1 = 1.2
    if n >= 13:
        for v in verts:
            if len(positions) >= n:
                break
            positions.append(v * r1)
    
    # Second shell: 20 or more atoms
    if len(positions) < n:
        # Use a denser set of points on a sphere
        r2 = 2.2
        # Generate points on sphere using golden spiral
        count = min(20, n - len(positions))
        for k in range(count):
            t = (k + 0.5) / count
            z = 1 - 2 * t
            theta = 2 * np.pi * k / 5
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r2)
    
    # Fill remaining with random close-packed positions
    while len(positions) < n:
        # Try random position near existing structure
        for attempt in range(100):
            p = rng.uniform(-3.0, 3.0, 3)
            if all(np.linalg.norm(p - q) >= 1.0 for q in positions):
                positions.append(p)
                break
        else:
            # Fallback: place at slightly larger radius
            angle = rng.uniform(0, 2 * np.pi)
            elev = rng.uniform(-np.pi/2, np.pi/2)
            r = 2.5 + rng.uniform(0, 0.5)
            p = np.array([
                r * np.cos(elev) * np.cos(angle),
                r * np.cos(elev) * np.sin(angle),
                r * np.sin(elev)
            ])
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_mackay(n, rng):
    """
    Generate a Mackay supericosahedron-like structure.
    For n=55, this is the classic Mackay icosahedron.
    For other n, we truncate or extend.
    """
    # Mackay icosahedron has 55 atoms: 1 center + 12 (r1) + 20 (r2) + 20 (r3) + 2 (extra)
    # Actually: 1 + 12 + 20 + 20 + 2 = 55? No: 1+12=13, +20=33, +20=53, +2=55
    # Standard Mackay: 1 center, 12 at r=1, 20 at r=sqrt(2), 20 at r=sqrt(4), 2 at r=sqrt(5)? 
    # Let's use a simpler approach: concentric shells
    
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    positions = [np.array([0.0, 0.0, 0.0])]  # Center
    
    # Shell 1: 12 atoms
    r1 = 1.15
    for v in verts:
        if len(positions) < n:
            positions.append(v * r1)
    
    # Shell 2: 20 atoms (face centers of icosahedron, approximated)
    if len(positions) < n:
        r2 = 1.95
        # 20 face centers approximated by points on sphere
        for k in range(20):
            if len(positions) >= n:
                break
            t = (k + 0.5) / 20
            z = 1 - 2 * t
            theta = 2 * np.pi * (k / 5.0)
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r2)
    
    # Shell 3: 20 atoms
    if len(positions) < n:
        r3 = 2.7
        for k in range(20):
            if len(positions) >= n:
                break
            t = (k + 0.5) / 20
            z = 1 - 2 * t
            theta = 2 * np.pi * ((k + 2.5) / 5.0)
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r3)
    
    # Shell 4: fill remaining
    while len(positions) < n:
        for attempt in range(200):
            p = rng.uniform(-3.5, 3.5, 3)
            if all(np.linalg.norm(p - q) >= 0.9 for q in positions):
                positions.append(p)
                break
        else:
            angle = rng.uniform(0, 2 * np.pi)
            elev = rng.uniform(-np.pi/2, np.pi/2)
            r = 3.2 + rng.uniform(0, 0.5)
            p = np.array([
                r * np.cos(elev) * np.cos(angle),
                r * np.cos(elev) * np.sin(angle),
                r * np.sin(elev)
            ])
            positions.append(p)
    
    return np.array(positions[:n])


def _basin_hopping(seed_positions, n, max_steps=50, kick_scale=0.3, rng=None):
    """
    Basin hopping: randomly perturb, relax, accept/reject.
    Returns best energy and positions found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = seed_positions.copy()
    E_current = _energy(current)
    best = current.copy()
    E_best = E_current
    
    T = 0.5  # Initial "temperature" for acceptance
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Random kick
        kick = rng.normal(0, kick_scale, size=current.shape)
        proposed = current + kick
        
        # Relax
        try:
            E_proposed, relaxed = relax(proposed, 'dzugutov')
        except Exception:
            E_proposed = E_current + 100  # Penalize failure
            relaxed = current
        
        # Check validity
        if not _is_valid(relaxed):
            E_proposed += 50  # Penalty for invalid structure
        
        # Metropolis acceptance
        dE = E_proposed - E_current
        if dE < 0 or rng.random() < np.exp(-dE / T):
            current = relaxed
            E_current = E_proposed
            if E_current < E_best:
                E_best = E_current
                best = current.copy()
        
        # Anneal
        T *= 0.95
        # Gradually increase kick size to explore more
        kick_scale = 0.2 + 0.3 * (step / max_steps)
    
    return E_best, best


def _worker(seed_positions, n, rng_seed, max_bh_steps=40):
    """Worker thread: run basin hopping from a seed."""
    rng = np.random.default_rng(rng_seed)
    E, pos = _basin_hopping(seed_positions, n, max_steps=max_bh_steps, rng=rng)
    return E, pos


def construct(instance):
    """
    Find low-energy Dzugutov cluster using basin hopping from polytetrahedral seeds.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Generate diverse seeds
    rng = np.random.default_rng(n)
    seeds = []
    
    # Seed 1: Icosahedral
    seeds.append(_generate_icosahedron(n, rng))
    # Seed 2: Mackay-like
    seeds.append(_generate_mackay(n, rng))
    # Seed 3: Perturbed icosahedral
    s3 = _generate_icosahedron(n, rng) + rng.normal(0, 0.2, (n, 3))
    seeds.append(s3)
    # Seed 4: Perturbed Mackay
    s4 = _generate_mackay(n, rng) + rng.normal(0, 0.2, (n, 3))
    seeds.append(s4)
    
    # Relax all seeds first to get better starting points
    relaxed_seeds = []
    for s in seeds:
        if _remaining_time() < 10.0:
            relaxed_seeds.append(s)
            continue
        try:
            E, relaxed = relax(s, 'dzugutov')
            if _is_valid(relaxed):
                relaxed_seeds.append(relaxed)
            else:
                relaxed_seeds.append(s)
        except Exception:
            relaxed_seeds.append(s)
    
    # Run basin hopping from each seed in parallel
    best_E = np.inf
    best_pos = relaxed_seeds[0]
    
    # First, get energies of relaxed seeds
    for i, rs in enumerate(relaxed_seeds):
        try:
            E = _energy(rs)
            if E < best_E:
                best_E = E
                best_pos = rs.copy()
        except Exception:
            pass
    
    # Run basin hopping chains
    if _remaining_time() > 15.0:
        n_chains = min(4, len(relaxed_seeds))
        chain_results = []
        
        with ThreadPoolExecutor(max_workers=_MAX_THREADS) as executor:
            futures = []
            for i in range(n_chains):
                rng_seed = n * 1000 + i
                future = executor.submit(_worker, relaxed_seeds[i], n, rng_seed, max_bh_steps=30)
                futures.append(future)
            
            for future in as_completed(futures):
                if _remaining_time() < 5.0:
                    break
                try:
                    E, pos = future.result(timeout=10)
                    if E < best_E:
                        best_E = E
                        best_pos = pos.copy()
                except Exception:
                    pass
    
    # Final relaxation of best result
    if _remaining_time() > 5.0:
        try:
            E_final, pos_final = relax(best_pos, 'dzugutov')
            if _is_valid(pos_final):
                E_check = _energy(pos_final)
                if E_check < best_E:
                    best_E = E_check
                    best_pos = pos_final
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
