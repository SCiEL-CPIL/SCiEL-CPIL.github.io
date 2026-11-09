"""Sutton-Chen cluster global minima search using diverse lattice prototypes and multi-restart relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

# Deadline guard: aim to finish well before the 330s hard kill
_DEADLINE = time.time() + 280.0

def _remaining():
    return _DEADLINE - time.time()

def _valid(pos):
    """Check if positions are finite and no two atoms are too close."""
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    # Compute pairwise distances
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35

def _fcc_prototype(n):
    """Generate an FCC cluster of n atoms centered at origin."""
    pts = []
    # FCC lattice vectors: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # We can just iterate over a grid and filter, or use specific generation.
    # Simpler: generate points on FCC lattice, sort by r, take n.
    max_idx = int(np.cbrt(n)) + 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC points are at (i,j,k) where i,j,k are integers and i+j+k is even? No.
                # Standard FCC basis: (0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0).
                # Let's generate a supercell of simple cubic and keep FCC sites.
                pass
    
    # Better approach: Generate all FCC points in a cube, sort by distance.
    pts = []
    limit = int(np.cbrt(n)) + 3
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # FCC condition: i,j,k integers, and i+j+k even? 
                # Actually, FCC lattice can be described as integer coordinates where i+j+k is even.
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    # Center
    pts -= np.mean(pts, axis=0)
    # Sort by distance to origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _bcc_prototype(n):
    """Generate a BCC cluster of n atoms centered at origin."""
    pts = []
    limit = int(np.cbrt(n)) + 3
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # BCC points: (i,j,k) where i,j,k integers, and either all even or all odd?
                # BCC basis: (0,0,0) and (1/2,1/2,1/2).
                # So points are (i,j,k) and (i+0.5, j+0.5, k+0.5).
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _hcp_prototype(n):
    """Generate an HCP cluster of n atoms centered at origin."""
    pts = []
    limit = int(np.cbrt(n)) + 3
    a = 1.0 # Lattice constant
    c = np.sqrt(8.0/3.0) * a # Ideal c/a ratio
    
    # HCP layers at z = k * c/2
    # Layer 0: (0,0,0) type
    # Layer 1: (1/2, 1/2*sqrt(3)/3, c/2) type? 
    # Let's use a standard hexagonal grid generation.
    # Hexagonal lattice vectors: a1 = (a, 0, 0), a2 = (a/2, a*sqrt(3)/2, 0)
    # Basis: (0,0,0) and (a/2, a*sqrt(3)/6, c/2)
    
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                # Position in hexagonal coords
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3)/2 * j)
                z = c * k / 2.0
                pts.append([x, y, z])
                # Second atom in basis
                x2 = x + a * 0.5
                y2 = y + a * np.sqrt(3)/6
                z2 = z + c/2
                pts.append([x2, y2, z2])
                
    pts = np.array(pts, dtype=float)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _random_cluster(n, rng, spacing=0.8):
    """Generate a random compact cluster."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: fill remaining with random points near center
        while len(points) < n:
            points.append(rng.normal(0, 0.5, 3))
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate prototypes
    prototypes = []
    
    # 1. Ideal Lattices
    try:
        prototypes.append(_fcc_prototype(size))
    except:
        pass
    try:
        prototypes.append(_bcc_prototype(size))
    except:
        pass
    try:
        prototypes.append(_hcp_prototype(size))
    except:
        pass
        
    # 2. Perturbed Lattices (3 variations each)
    for base in list(prototypes):
        for sigma in [0.1, 0.2, 0.3]:
            perturbed = base + rng.normal(0, sigma, base.shape)
            prototypes.append(perturbed)
            
    # 3. Random Clusters (4 variations)
    for _ in range(4):
        prototypes.append(_random_cluster(size, rng))
        
    # Shuffle to ensure diversity in order of processing
    rng.shuffle(prototypes)
    
    # Relax each prototype
    for pos in prototypes:
        if _remaining() < 5.0:
            break
            
        if not _valid(pos):
            continue
            
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if _valid(relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue
            
    # If no valid result found, return a valid random cluster
    if best_positions is None:
        best_positions = _random_cluster(size, rng)
        
    return best_positions
# EVOLVE-BLOCK-END
