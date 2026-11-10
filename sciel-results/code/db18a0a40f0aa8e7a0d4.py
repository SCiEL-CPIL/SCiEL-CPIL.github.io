"""Dzugutov cluster global minima: divide-and-conquer polytetrahedral assembly."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 318.0  # safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_icosahedron(scale=1.0, center=None):
    """
    Generate a 13-atom icosahedron (1 center + 12 vertices).
    The Dzugutov potential favors icosahedral order.
    Returns (13, 3) array.
    """
    if center is None:
        center = np.zeros(3)
    
    phi = (1 + np.sqrt(5)) / 2
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norm = np.linalg.norm(verts[0])
    verts = verts / norm
    
    # Scale to appropriate bond length for Dzugutov
    # The minimum of the Dzugutov potential is around r=1.0-1.1 depending on parameters
    # Let's use a scale that puts vertices at ~1.0 distance from center
    # Actually, the potential has a minimum. Let's assume standard icosahedron geometry 
    # is a good starting point and let relaxation adjust.
    # A typical nearest neighbor distance in these clusters is ~1.0.
    
    points = np.vstack([center, verts * scale])
    return points

def _generate_motif(size, rng, ico_scale=1.0):
    """
    Generate a motif of 'size' atoms by combining icosahedra.
    If size is not a multiple of 13, we add atoms or trim.
    For now, we just create a chain or cluster of icosahedra.
    """
    points = []
    n_icos = size // 13
    remainder = size % 13
    
    # Place icosahedra in a line or grid to form a larger structure
    # A simple strategy: place them along an axis with some overlap
    # or in a 3D grid.
    
    if n_icos == 0:
        # Just a partial icosahedron or random
        ico = _generate_icosahedron(ico_scale)
        return ico[:size]
    
    # Place icosahedra centers
    centers = []
    for i in range(n_icos):
        # Arrange in a grid
        ix = i % 3
        iy = (i // 3) % 3
        iz = i // 9
        # Spacing needs to be such that they interpenetrate slightly or touch
        # Icosahedron radius is ~1.0. Distance between centers should be ~2.0
        c = np.array([ix * 2.0, iy * 2.0, iz * 2.0])
        centers.append(c)
        
    for c in centers:
        ico = _generate_icosahedron(ico_scale, center=c)
        points.append(ico)
        
    points = np.vstack(points)
    
    # If we have remainder, add them randomly near the surface or just append
    if remainder > 0:
        # Take the first 'remainder' atoms from a new icosahedron
        extra_ico = _generate_icosahedron(ico_scale, center=np.zeros(3))
        # Add only the vertices (skip center to avoid duplicate center if we had one)
        # Actually, just take the first 'remainder' points
        points = np.vstack([points, extra_ico[:remainder]])
        
    return points[:size]

def _assemble_cluster(n, rng, ico_scale=1.0):
    """
    Assemble a cluster of n atoms using divide-and-conquer.
    Strategy:
    1. If n <= 13, use a single icosahedron.
    2. If n > 13, combine icosahedra.
    
    We will try a few different assembly patterns.
    """
    if n <= 13:
        ico = _generate_icosahedron(ico_scale)
        return ico[:n]
    
    # Pattern 1: Linear chain of icosahedra
    # Pattern 2: 2D grid
    # Pattern 3: Random placement of icosahedra with repulsion
    
    # Let's try a random placement of icosahedra centers, then relax the whole thing.
    # This is effectively "seeded" basin hopping but with better initial local order.
    
    n_icos = n // 13
    remainder = n % 13
    
    if n_icos == 0:
        ico = _generate_icosahedron(ico_scale)
        return ico[:n]
    
    # Generate centers for the icosahedra
    # We want them to be close but not overlapping too much
    # Radius of icosahedron is ~1.0. So centers should be at least ~1.5 apart?
    # Actually, in polytetrahedral packing, they share vertices.
    # Distance between centers of touching icosahedra is ~2 * cos(31.7 deg) ~ 1.7?
    # Let's just place them randomly in a box and let relaxation fix it.
    
    box_size = 1.5 * n_icos ** (1/3)
    centers = rng.uniform(-box_size, box_size, (n_icos, 3))
    
    points = []
    for c in centers:
        ico = _generate_icosahedron(ico_scale, center=c)
        points.append(ico)
        
    points = np.vstack(points)
    
    # Add remainder atoms
    if remainder > 0:
        # Place them near the last icosahedron
        last_center = centers[-1]
        extra_ico = _generate_icosahedron(ico_scale, center=last_center)
        # Add vertices only to avoid center duplicate if we are careful, 
        # but for now just add points
        points = np.vstack([points, extra_ico[:remainder]])
        
    return points[:n]

def _basin_hopping(positions, potential, rng, max_iters=50, step_scale=0.4, time_check=None):
    """
    Basin hopping algorithm.
    """
    best_positions = positions.copy()
    best_energy, _ = relax(positions, potential)
    current_positions = positions.copy()
    current_energy = best_energy
    
    for it in range(max_iters):
        if time_check is not None and time_check() < 20:
            break
            
        # Random perturbation
        # Use a step size that allows jumping over barriers
        step = step_scale * rng.normal(0, 1, current_positions.shape)
        new_positions = current_positions + step
        
        # Relax
        energy, relaxed = relax(new_positions, potential)
        
        # Metropolis acceptance
        if energy < current_energy:
            current_positions = relaxed.copy()
            current_energy = energy
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        else:
            delta_e = energy - current_energy
            # Temperature T=1.0 is a reasonable guess for these systems
            if rng.random() < np.exp(-delta_e / 1.0):
                current_positions = relaxed.copy()
                current_energy = energy
                
    return best_positions, best_energy

def construct(instance):
    """
    Find low-energy Dzugutov clusters using divide-and-conquer assembly.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple diverse assemblies
    # We will try a few different scales and random seeds for the assembly
    n_seeds = 5
    seeds = []
    
    for i in range(n_seeds):
        # Vary the icosahedron scale slightly to explore different bond lengths
        scale = 1.0 + rng.uniform(-0.1, 0.1)
        seed = _assemble_cluster(n, rng, ico_scale=scale)
        seeds.append(seed)
        
    # Also try a single icosahedron based seed if n is close to 13, 26, etc.
    # For larger n, the assembly is key.
    
    # Relax all seeds
    for seed in seeds:
        # Enforce minimum distance to avoid singularities in potential
        # The relax function might handle this, but let's be safe
        # Simple repulsion
        for _ in range(5):
            moved = False
            for i in range(len(seed)):
                for j in range(i+1, len(seed)):
                    d = np.linalg.norm(seed[i] - seed[j])
                    if d < 0.5:
                        diff = seed[i] - seed[j]
                        if d < 1e-10:
                            diff = np.array([1.0, 0.0, 0.0])
                        else:
                            diff /= d
                        push = (0.5 - d) / 2 + 0.01
                        seed[i] += diff * push
                        seed[j] -= diff * push
                        moved = True
            if not moved:
                break
                
        energy, positions = relax(seed, potential)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
            
    # Basin hopping from the best seed
    # We can also run BH from the top 2 seeds if time permits
    time_remaining = _get_time_remaining()
    
    if best_positions is not None:
        # Run BH on the best
        if time_remaining > 30:
            n_iters = min(100, int((time_remaining - 30) / 1.5)) # ~1.5s per iteration?
            # Relax is fast, BH step is just one relax.
            # Let's estimate 0.5s per BH iteration for n=60.
            n_iters = min(150, int((time_remaining - 30) / 0.5))
            
            best_positions, best_energy = _basin_hopping(
                best_positions, potential, rng,
                max_iters=n_iters, step_scale=0.4,
                time_check=_get_time_remaining
            )
            
        # If we have time, try the second best seed
        if time_remaining > 30 and len(seeds) > 1:
            # Find the second best seed from the initial relaxation
            # We didn't store them, so let's just pick another random one from the list
            # Or re-relax a different seed
            # For simplicity, we just do more BH on the best one.
            pass
            
    # Final relaxation
    if best_positions is not None:
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
            
    if best_positions is None:
        # Fallback
        best_positions = np.array([[i * 1.1, 0.0, 0.0] for i in range(n)])
        
    return best_positions
# EVOLVE-BLOCK-END
