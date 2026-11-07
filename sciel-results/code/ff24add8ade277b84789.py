"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Efficient distance check
    # Using scipy.spatial.distance.pdist might be faster but numpy is sufficient for N~80
    # We need to check all pairs
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    pts = []
    # FCC lattice vectors
    # Center at origin
    # We will pick the n closest points in an FCC lattice
    # FCC points: (i,j,k) where i+j+k is even, scaled by a/sqrt(2)
    # Let's use a standard FCC basis
    # a = 1.0
    # points: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) in units of a/2?
    # Standard FCC: (0,0,0), (0, 1/2, 1/2), (1/2, 0, 1/2), (1/2, 1/2, 0)
    # Lattice constant a=1.
    
    max_radius = int(np.ceil(n ** (1/3))) + 5
    candidates = []
    for i in range(-max_radius, max_radius + 1):
        for j in range(-max_radius, max_radius + 1):
            for k in range(-max_radius, max_radius + 1):
                # FCC condition: i+j+k even? No, that's BCC or simple cubic variants.
                # FCC points are integer coordinates (i,j,k) with i+j+k even? 
                # Actually, FCC is often represented as:
                # (i, j, k) where i,j,k are integers and i+j+k is even? 
                # Let's check: (0,0,0) sum=0 (even). (1,1,0) sum=2 (even). (1,0,1) sum=2.
                # Distance (1,1,0) is sqrt(2). Nearest neighbor in FCC is a/sqrt(2).
                # If we use integer grid with step 1, nearest neighbor is 1.
                # FCC nearest neighbor distance is a/sqrt(2).
                # If we scale by 1/sqrt(2), distance is 1.
                # Let's just generate points on a grid and filter.
                
                # Simpler: Generate a grid of points, assign to FCC sites.
                # Site 1: (i, j, k)
                # Site 2: (i+0.5, j+0.5, k)
                # Site 3: (i+0.5, j, k+0.5)
                # Site 4: (i, j+0.5, k+0.5)
                # This is getting complex. Let's use a simpler heuristic:
                # Just take the n closest points to the origin in an FCC lattice.
                
                pass
    
    # Alternative: Use the known coordinates for small clusters? No, n varies.
    # Let's use a "shell" filling approach for FCC.
    # Layers in FCC: 1, 12, 42, 100...
    # This is tricky.
    
    # Let's try a different approach: 
    # Generate points on a cubic grid, filter to keep only those that form an FCC-like packing.
    # Actually, the most robust simple generator is:
    # 1. Create a list of all integer coordinates in a box.
    # 2. Keep only those where (i+j+k) % 2 == 0. This is a BCC lattice? No.
    #    BCC: (0,0,0) and (0.5, 0.5, 0.5).
    #    FCC: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0).
    
    # Let's generate FCC points properly.
    pts = []
    scale = 0.5 # Half lattice constant
    # We want nearest neighbor distance ~ 0.7-0.8
    # In FCC, NN distance = a / sqrt(2). If a=1, NN = 0.707.
    # Points: (i, j, k) * a where i,j,k integers and i+j+k even?
    # If i,j,k are integers, (1,1,0) is a point. Dist to (0,0,0) is sqrt(2).
    # If we scale by 1/sqrt(2), dist is 1.
    # Let's use the set: { (i, j, k) | i,j,k in Z, i+j+k even } * (1/sqrt(2))
    
    max_r = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    pts = []
    # BCC: (i, j, k) and (i+0.5, j+0.5, k+0.5)
    # NN distance in BCC is a*sqrt(3)/2.
    # If a=1, NN = 0.866.
    max_r = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_icosahedron(n):
    """Generate an icosahedral cluster."""
    # For n <= 13, it's the perfect icosahedron.
    # For larger n, add shells.
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = []
    for i in range(12):
        # Standard icosahedron coordinates
        # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
        pass
    
    # Simpler: Use known 12 vertices
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    # Normalize to unit radius
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)] # Center
    if n > 1:
        pts.extend(v[:n-1])
        
    if n > 13:
        # Add second shell (42 atoms)
        # This is complex. Let's just add random points on a sphere of radius ~1.8
        # Or better, just return the first n points from a larger generated set.
        # For now, let's stick to FCC/BCC which are easier to generate generally.
        # Icosahedral clusters are often local minima, but FCC/BCC are better global minima candidates for larger N in SC potential?
        # Actually, SC potential favors FCC-like structures for many sizes.
        
        # Let's just use FCC for n > 13 in this helper, or a mix.
        pass
    
    return np.array(pts[:n])

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    # kind is '12-6', '9-6', '10-8'
    # The relax function expects potential='sutton_chen' and kind=kind?
    # The signature is relax(positions, 'sutton_chen', kind)
    # Let's assume kind is passed directly.
    
    best_energy = np.inf
    best_pos = None
    
    # List of generator functions
    generators = [_generate_fcc, _generate_bcc]
    
    # If n is small, icosahedron might be good, but let's stick to FCC/BCC for robustness.
    
    # We will try multiple random perturbations of these base structures.
    # Time budget: ~50s per instance.
    
    rng = np.random.default_rng(42 + n)
    
    # Number of restarts
    # We don't know how fast relax is. Let's assume it's fast enough for ~100-200 restarts.
    # But we must guard time.
    
    max_restarts = 200
    
    for i in range(max_restarts):
        if _time_left() < 10.0:
            break
            
        # Choose a base structure
        gen = generators[i % len(generators)]
        base = gen(n)
        
        # Perturb
        # Small noise to break symmetry and escape local minima
        noise = rng.normal(0, 0.1, base.shape)
        pos = base + noise
        
        # Relax
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            
            if not np.isfinite(energy):
                continue
                
            if _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
            
    # If no valid position found, return a default valid one (e.g. FCC)
    if best_pos is None:
        best_pos = _generate_fcc(n)
        # Ensure it's valid? FCC should be valid.
        if not _is_valid(best_pos):
             # Fallback to random
             best_pos = rng.uniform(-1, 1, (n, 3))
             # This might be invalid, but better than crashing?
             # The task says "Non-finite coordinates or two atoms closer than 0.35 make the whole result invalid."
             # We must return a valid array.
             # Let's try to fix distances if invalid.
             # But random uniform in [-1,1] for N=80 might have collisions.
             # Let's use the FCC generator again, it's very likely valid.
             best_pos = _generate_fcc(n)
             
    return best_pos

# EVOLVE-BLOCK-END
