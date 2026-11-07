"""
Sutton-Chen transition-metal cluster global minima search.
Uses lattice fragment initialization (FCC, HCP, BCC) followed by local relaxation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global time management
_START_TIME = None
_TOTAL_TIME_LIMIT = 295.0  # Safety margin before 300s kill

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _get_start_time())

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
    min_dist = np.min(dists)
    return min_dist >= 0.35

def _generate_fcc(n, rng):
    """Generate n atoms from an FCC lattice fragment."""
    # FCC basis vectors
    a = 1.0 # Lattice constant
    # Generate points in a cube large enough to contain n atoms
    # Density of FCC is 4 atoms per unit cell volume a^3
    # Volume needed ~ n/4 * a^3
    # Side length ~ (n/4)^(1/3) * a
    # Add margin
    L = int((n / 4.0) ** (1/3.0)) + 3
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # FCC atoms at (0,0,0) and face centers
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, k*a])
                points.append([(i+0.5)*a, j*a, (k+0.5)*a])
                points.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _generate_hcp(n, rng):
    """Generate n atoms from an HCP lattice fragment."""
    # HCP is a bit more complex. We can approximate by stacking FCC layers or using basis.
    # Basis: (0,0,0), (1/3, 2/3, 1/2) in hexagonal coords?
    # Let's use Cartesian coordinates for a standard HCP structure.
    # a = 1.0, c/a = sqrt(8/3) ~ 1.633
    a = 1.0
    c = a * np.sqrt(8/3)
    L = int((n / 2.0) ** (1/3.0)) + 3 # HCP has 2 atoms per cell
    
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # Layer 1 (z=0)
                x1 = i*a + 0.5*j*a
                y1 = j*a * np.sqrt(3)/2
                z1 = k*c
                points.append([x1, y1, z1])
                
                # Layer 2 (z=c/2), shifted
                x2 = i*a + 0.5*j*a + a/3
                y2 = j*a * np.sqrt(3)/2 + a*np.sqrt(3)/6
                z2 = k*c + c/2
                points.append([x2, y2, z2])
                
                # Note: HCP stacking is ABAB. The above generates a simple hexagonal lattice with 2 atoms.
                # To get proper HCP, we need to alternate layers.
                # Let's refine: 
                # Layer A: (i, j, k)
                # Layer B: (i, j, k) + shift, at z = (k+0.5)c
                # Actually, the simple generation above creates a rhombohedral or distorted structure if not careful.
                # A safer approach for HCP fragment: Take FCC and shear it? 
                # Or just use a large box and keep the n closest.
                # The above points form a hexagonal lattice with 2 atoms per primitive cell if we consider the basis correctly.
                # Let's stick to the generated points and let the relaxation fix the c/a ratio if it's off.
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def _generate_bcc(n, rng):
    """Generate n atoms from a BCC lattice fragment."""
    a = 1.0
    L = int((n / 2.0) ** (1/3.0)) + 3
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                points.append([i*a, j*a, k*a])
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    return points[idx[:n]]

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate candidate structures
    candidates = []
    
    # 1. FCC fragment
    try:
        fcc_pos = _generate_fcc(size, rng)
        if _validate(fcc_pos):
            candidates.append(fcc_pos)
    except Exception:
        pass
        
    # 2. HCP fragment
    try:
        hcp_pos = _generate_hcp(size, rng)
        if _validate(hcp_pos):
            candidates.append(hcp_pos)
    except Exception:
        pass
        
    # 3. BCC fragment
    try:
        bcc_pos = _generate_bcc(size, rng)
        if _validate(bcc_pos):
            candidates.append(bcc_pos)
    except Exception:
        pass
        
    # 4. Perturbed FCC (to escape local minima of the exact fragment)
    # We add small random noise to the FCC positions
    try:
        fcc_pert = _generate_fcc(size, rng) + rng.normal(0, 0.1, (size, 3))
        if _validate(fcc_pert):
            candidates.append(fcc_pert)
    except Exception:
        pass
        
    # 5. Perturbed HCP
    try:
        hcp_pert = _generate_hcp(size, rng) + rng.normal(0, 0.1, (size, 3))
        if _validate(hcp_pert):
            candidates.append(hcp_pert)
    except Exception:
        pass
        
    # 6. Random compact cluster (fallback, similar to seed but better initialized)
    # Use a sphere packing approach or just random within a box, but ensure min dist
    # We'll skip pure random if we have enough lattice candidates, but let's add one more diverse one.
    # A "random" close-packed structure can be generated by random addition with rejection.
    # However, lattice fragments are usually better. Let's add a perturbed BCC.
    try:
        bcc_pert = _generate_bcc(size, rng) + rng.normal(0, 0.1, (size, 3))
        if _validate(bcc_pert):
            candidates.append(bcc_pert)
    except Exception:
        pass
        
    # Relax each candidate
    for pos in candidates:
        if _time_left() < 10.0: # Need time for at least one more relaxation
            break
            
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if np.isfinite(energy) and _validate(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
        except Exception:
            continue
            
    # If no valid result was found (unlikely), return a simple random valid structure
    if best_positions is None:
        # Fallback: Random points with min distance
        points = []
        radius = 1.0 * size ** (1.0/3.0)
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.35 for q in points):
                points.append(p)
        best_positions = np.array(points)
        try:
            energy, relaxed_pos = relax(best_positions, potential, kind)
            if _validate(relaxed_pos):
                best_positions = relaxed_pos
        except:
            pass

    return best_positions
# EVOLVE-BLOCK-END
