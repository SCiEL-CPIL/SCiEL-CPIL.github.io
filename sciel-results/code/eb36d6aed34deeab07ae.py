"""Improved program for Sutton-Chen cluster global minima search using Shell-Based Initialization."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time


def _generate_shell_structure(n, rng, num_shells=None):
    """
    Generate an initial structure based on concentric shells.
    This mimics the typical packing of transition metal clusters.
    """
    if num_shells is None:
        # Heuristic for number of shells
        # Roughly n ~ 4/3 pi r^3, so r ~ (3n/4pi)^(1/3)
        # Shell radii roughly proportional to integer indices
        num_shells = max(1, int((n / 12) ** (1/3.0)) + 1)
    
    # Assign atoms to shells
    # First shell: 1 atom (center) or small number
    # Subsequent shells: more atoms
    # We'll distribute atoms roughly proportional to shell surface area
    
    # Simpler approach: Just place atoms in a few concentric spherical shells
    # with random angular distribution.
    
    # Determine radii
    # Assume average density.
    # Let's use a fixed set of relative radii for simplicity, scaled by total size.
    # Or, distribute atoms into shells such that volume is filled.
    
    # Let's use a simple heuristic:
    # Shell 0: 1 atom at center (if n > 1)
    # Shell 1: next ~12 atoms
    # Shell 2: next ~42 atoms
    # ...
    # This is getting complex. Let's use a simpler "Random Spherical Shells" approach.
    
    # Divide n atoms into K shells.
    K = min(num_shells, n)
    if K == 1:
        # Just a random ball
        r_max = (n / (4/3 * np.pi)) ** (1/3.0)
        radii = [r_max]
        counts = [n]
    else:
        # Distribute counts roughly proportional to shell volume (r^3)
        # Let shell i have radius proportional to (i+1)
        # Volume proportional to (i+1)^3
        weights = [(i+1)**3 for i in range(K)]
        total_weight = sum(weights)
        counts = [0] * K
        remaining = n
        for i in range(K-1):
            c = int(remaining * weights[i] / total_weight)
            c = max(1, c) # At least 1 atom per shell if possible
            counts[i] = c
            remaining -= c
        counts[K-1] = remaining
        
        radii = []
        # Scale radii so that the total volume matches n
        # Sum of shell volumes ~ n
        # Let's just pick radii linearly spaced and scale
        r_max = (n / (4/3 * np.pi)) ** (1/3.0) * 1.2
        radii = [r_max * (i+1) / K for i in range(K)]
        
    points = []
    
    for i in range(K):
        r = radii[i]
        count = counts[i]
        if count == 0:
            continue
        # Generate 'count' points on a sphere of radius r with some noise
        for j in range(count):
            # Random direction
            u = rng.random(3)
            # u[0] is for radial noise? No, let's put them on the shell with slight radial jitter
            # Direction
            theta = 2 * np.pi * u[1]
            phi = np.arccos(2 * u[2] - 1)
            # Radial jitter: +/- 10% of r
            r_jitter = r * (0.9 + 0.2 * u[0])
            
            x = r_jitter * np.sin(phi) * np.cos(theta)
            y = r_jitter * np.sin(phi) * np.sin(theta)
            z = r_jitter * np.cos(phi)
            points.append([x, y, z])
            
    return np.array(points)


def _basin_hopping_polish(positions, potential, kind, n_iter=20, T0=0.5, Tf=0.01, kick_size=0.3, rng=None):
    """
    A short Basin Hopping run to polish a structure.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    try:
        E_curr, x_curr = relax(positions, potential, kind)
    except Exception:
        return np.inf, positions
        
    E_best = E_curr
    x_best = x_curr.copy()
    
    T = T0
    cooling_factor = (Tf / T0) ** (1.0 / max(1, n_iter - 1))
    
    for i in range(n_iter):
        x_prop = x_curr + rng.normal(0, kick_size, x_curr.shape)
        
        try:
            E_prop, x_prop_min = relax(x_prop, potential, kind)
        except Exception:
            E_prop = np.inf
            x_prop_min = x_prop
            
        if np.isfinite(E_prop):
            dE = E_prop - E_curr
            if dE < 0 or rng.random() < np.exp(-dE / T):
                x_curr = x_prop_min
                E_curr = E_prop
                if E_prop < E_best:
                    E_best = E_prop
                    x_best = x_prop_min.copy()
        T *= cooling_factor
        
    return E_best, x_best


def construct(instance):
    """
    Multi-start optimization using Shell-Based Initialization followed by BH Polish.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Time budget management
    # Total 300s for 6 instances. ~50s per instance.
    # We will do multiple starts.
    
    # Number of starts depends on size
    if size < 40:
        n_starts = 10
        bh_iters = 30
    elif size < 60:
        n_starts = 6
        bh_iters = 20
    else:
        n_starts = 4
        bh_iters = 10
        
    best_energy = np.inf
    best_positions = None
    
    start_time = time.time()
    time_per_start = 40.0 / n_starts # Leave 10s buffer
    
    for start in range(n_starts):
        if time.time() - start_time > 45.0:
            break
            
        # Generate initial structure
        # Vary the number of shells and randomness
        num_shells = rng.integers(2, 6)
        x0 = _generate_shell_structure(size, rng, num_shells=num_shells)
        
        # Initial Relaxation
        try:
            E0, x0_relaxed = relax(x0, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(E0):
            continue
            
        # Polish with BH
        E, x = _basin_hopping_polish(x0_relaxed, potential, kind, n_iter=bh_iters, T0=0.5, Tf=0.01, kick_size=0.3, rng=rng)
        
        if np.isfinite(E) and E < best_energy:
            best_energy = E
            best_positions = x
            
    # Check validity
    if best_positions is not None:
        min_d = np.inf
        for a in range(len(best_positions)):
            for b in range(a+1, len(best_positions)):
                d = np.linalg.norm(best_positions[a] - best_positions[b])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            best_positions = None
            
    # Fallback
    if best_positions is None:
        # Use a simple random ball and relax
        x0 = np.random.default_rng(1).random((size, 3)) * 2.0 - 1.0
        try:
            E, x = relax(x0, potential, kind)
            if np.isfinite(E):
                best_positions = x
        except Exception:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
