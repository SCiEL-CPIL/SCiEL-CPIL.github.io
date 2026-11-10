"""
Sutton-Chen cluster global minima search.
Uses lattice fragment seeding and simulated annealing refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_fcc(n, rng, jitter_scale=0.1):
    """Generate a random FCC fragment of n atoms."""
    # Generate a large FCC lattice
    # a = 1.0 (lattice constant)
    # Points: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by a
    # We generate points in a box [-L, L]^3
    L = int(np.ceil(n ** (1/3))) + 2
    pts = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # FCC basis
                pts.append([i, j, k])
                pts.append([i, j+0.5, k+0.5])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i+0.5, j+0.5, k])
    pts = np.array(pts)
    # Center at origin
    pts -= np.mean(pts, axis=0)
    # Sort by distance to origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    # Take n closest
    frag = pts[idx[:n]]
    # Random rotation
    # Use quaternion or rotation matrix
    # Simple rotation matrix from random angles
    t1 = rng.uniform(0, 2*np.pi)
    t2 = rng.uniform(0, 2*np.pi)
    t3 = rng.uniform(0, 2*np.pi)
    R1 = np.array([[1,0,0],[0,np.cos(t1),-np.sin(t1)],[0,np.sin(t1),np.cos(t1)]])
    R2 = np.array([[np.cos(t2),0,np.sin(t2)],[0,1,0],[-np.sin(t2),0,np.cos(t2)]])
    R3 = np.array([[np.cos(t3),-np.sin(t3),0],[np.sin(t3),np.cos(t3),0],[0,0,1]])
    R = R1 @ R2 @ R3
    frag = frag @ R.T
    
    # Jitter
    frag += rng.normal(0, jitter_scale, frag.shape)
    
    return frag

def _generate_hcp(n, rng, jitter_scale=0.1):
    """Generate a random HCP fragment of n atoms."""
    # HCP structure
    # a = 1.0, c/a = 1.633
    c = 1.633
    # Basis in hexagonal coords -> Cartesian
    # Layer 0: (0,0,0), (1,0,0), (0,1,0), (1,1,0), ... in hex grid
    # Let's generate hex grid points
    L = int(np.ceil(n ** (1/3))) + 2
    pts = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            # Hex to cart
            x = i + 0.5*j
            y = np.sqrt(3)/2 * j
            z = 0
            pts.append([x, y, z])
            # Layer 1 (shifted)
            if (i+j) % 2 == 0: # Simplified stacking, might not be perfect HCP but close
                 # Actually HCP is ABAB. 
                 # A layer: (0,0,0)
                 # B layer: (1/3, 2/3, 1/2) in fractional coords of hex cell
                 # Let's just generate two layers and stack them
                 pass
    
    # This hex generation is tricky to get right quickly. 
    # Alternative: Use FCC and rotate? No.
    # Let's stick to FCC which is very common for these clusters, and maybe BCC?
    # Actually, for TM clusters, FCC is the dominant motif.
    # I will implement a simpler HCP or just rely on FCC + Random.
    
    # Let's just use FCC for now, it's robust.
    # If we have time, add BCC.
    return _generate_fcc(n, rng, jitter_scale)

def _random_packed(n, rng, radius=None):
    if radius is None:
        radius = n ** (1/3) * 0.8
    pts = []
    min_dist = 0.7
    max_tries = 10000
    tries = 0
    while len(pts) < n and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            tries += 1
            continue
        # Check distance
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
        else:
            tries += 1
    if len(pts) < n:
        # Fill rest randomly
        while len(pts) < n:
            pts.append(rng.uniform(-radius, radius, 3))
    return np.array(pts)

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    start_time = time.time()
    deadline = 300.0 - 10.0 # Safety margin
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in dev, but maybe random is better?
    # Using fixed seed might be bad if it fails for a specific n. 
    # Let's use random seed.
    rng = np.random.default_rng()
    
    best_energy = float('inf')
    best_pos = None
    
    # 1. Generate diverse seeds
    seeds = []
    
    # FCC fragments with different jitters
    for j in [0.01, 0.05, 0.1, 0.2]:
        seeds.append(_generate_fcc(n, rng, j))
        
    # Random packed
    seeds.append(_random_packed(n, rng))
    seeds.append(_random_packed(n, rng))
    
    # 2. Relax seeds
    for seed in seeds:
        if time.time() > deadline:
            break
        try:
            e, pos = relax(seed, 'sutton_chen', kind)
            if e < best_energy and np.isfinite(e):
                # Check validity
                if np.all(np.isfinite(pos)):
                    # Check min distance
                    d = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
                    np.fill_diagonal(d, np.inf)
                    if np.min(d) > 0.35:
                        best_energy = e
                        best_pos = pos
        except Exception:
            pass
            
    # 3. Simulated Annealing / Kicks
    if best_pos is not None:
        sigma = 0.1
        while time.time() < deadline - 5.0:
            perturbed = best_pos + rng.normal(0, sigma, best_pos.shape)
            # Check validity before expensive relax
            d = np.linalg.norm(perturbed[:, np.newaxis] - perturbed[np.newaxis, :], axis=2)
            np.fill_diagonal(d, np.inf)
            if np.min(d) < 0.35:
                continue
            
            try:
                e, pos = relax(perturbed, 'sutton_chen', kind)
                if e < best_energy and np.isfinite(e):
                    if np.all(np.isfinite(pos)):
                        d = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
                        np.fill_diagonal(d, np.inf)
                        if np.min(d) > 0.35:
                            best_energy = e
                            best_pos = pos
                            # Decrease sigma slightly? Or keep it?
                            # sigma *= 0.95 
                            pass
                else:
                    # If we didn't improve, maybe increase sigma?
                    # Or just keep trying.
                    pass
            except Exception:
                pass

    if best_pos is None:
        # Fallback: return a valid random config
        return _random_packed(n, rng)
        
    return best_pos

# EVOLVE-BLOCK-END
