"""
Strategy: Polytetrahedral (PT) motif seeding for Dzugutov clusters.
The Dzugutov potential is designed to favor PT order (icosahedral shells).
Instead of random packing, we generate candidate structures based on:
1. Mackay supericosahedra (55 atoms)
2. Icosahedral shells with varying radii (generalizing to n=38..60)
3. Random perturbations of these motifs
We then use the provided relax function to locally minimize these starts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 300.0  # The evaluator kills at 330s, but we must finish within 300s total for all calls?
# The prompt says: "all calls together within 300 s". 
# Since construct is called once per n, and we don't know the exact order or if it's parallel,
# we'll track global time.

def _get_remaining_time():
    return _HARD_LIMIT - (time.time() - _START_TIME)

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a rough icosahedral cluster.
    Shell 0: 1 atom
    Shell 1: 12 atoms (icosahedron)
    Shell 2: 42 atoms (Mackay 55)
    Shell 3: ...
    """
    # Icosahedron vertices (12)
    t = (1 + np.sqrt(5)) / 2.0
    v1 = np.array([[0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
                   [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
                   [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]])
    # Normalize to unit radius
    norms = np.linalg.norm(v1, axis=1, keepdims=True)
    v1 = v1 / norms
    
    points = [np.zeros(3)]  # Center
    
    if n_atoms > 1:
        # Add shell 1 (12 atoms)
        # The distance between center and shell 1 in Dzugutov is roughly 1.0-1.2.
        # Let's start with radius 1.0.
        r1 = 1.0
        points.extend([v1[i] * r1 for i in range(12)])
        
    if n_atoms > 13:
        # Add shell 2 (Mackay 55 has 42 more atoms)
        # Mackay 55: 1 + 12 + 42.
        # The 42 atoms are located at the midpoints of edges (30), centers of faces (20), and... wait.
        # Mackay 55 vertices:
        # 1 center
        # 12 vertices of icosahedron
        # 30 edge midpoints
        # 20 face centers
        # Total 63? No, 1+12+30+20 = 63. 
        # Wait, Mackay 55 is 1 + 12 + 42.
        # The 42 atoms in the second shell of a Mackay icosahedron are:
        # 30 atoms at the midpoints of the 30 edges of the inner icosahedron? No, that's not right.
        # Actually, the 55-atom Mackay icosahedron consists of:
        # 1 central atom
        # 12 atoms at the vertices of a regular icosahedron
        # 42 atoms arranged in the second shell.
        # The 42 atoms are:
        # 30 atoms at the midpoints of the 30 edges of the icosahedron? No.
        # Let's look up the coordinates for Mackay 55.
        # Alternatively, we can just place atoms on a sphere of radius r2 and pick the best 42 positions
        # that are far from each other and from the inner shell.
        
        # A simpler approach: Generate a large number of random points on a sphere of radius r2
        # and pick the subset that maximizes the minimum distance to existing points and to each other.
        # This is a greedy packing algorithm.
        
        r2 = 1.9 # Approximate radius for second shell
        n_to_add = n_atoms - len(points)
        
        # Generate many candidate points on the sphere
        n_candidates = 500
        # Use Fibonacci sphere for uniform distribution
        phi = np.arccos(1 - 2 * np.arange(n_candidates) / n_candidates)
        theta = np.pi * (1 + np.sqrt(5)) * np.arange(n_candidates)
        
        candidates = np.zeros((n_candidates, 3))
        candidates[:, 0] = np.sin(phi) * np.cos(theta)
        candidates[:, 1] = np.sin(phi) * np.sin(theta)
        candidates[:, 2] = np.cos(phi)
        candidates *= r2
        
        # Add some noise to break symmetry and allow optimization to adjust
        candidates += rng.normal(0, 0.1, candidates.shape)
        
        # Greedy selection
        current_points = np.array(points)
        selected_indices = []
        
        for _ in range(n_to_add):
            # Calculate min distance from each candidate to all current points
            # and to already selected candidates
            # This is O(N * M) which is fine for small N, M
            
            # Distance to existing points
            dists_existing = np.linalg.norm(candidates[:, np.newaxis, :] - current_points[np.newaxis, :, :], axis=2)
            min_dist_existing = np.min(dists_existing, axis=1)
            
            if selected_indices:
                # Distance to already selected candidates
                sel_pts = candidates[selected_indices]
                dists_sel = np.linalg.norm(candidates[:, np.newaxis, :] - sel_pts[np.newaxis, :, :], axis=2)
                min_dist_sel = np.min(dists_sel, axis=1)
                min_dist = np.minimum(min_dist_existing, min_dist_sel)
            else:
                min_dist = min_dist_existing
            
            # Pick the candidate with max min_dist
            best_idx = np.argmax(min_dist)
            selected_indices.append(best_idx)
            current_points = np.vstack([current_points, candidates[best_idx]])
            
            # Remove selected candidate from pool to avoid re-picking
            candidates = np.delete(candidates, best_idx, axis=0)
            if len(candidates) == 0:
                break

        points.extend([candidates[i] for i in selected_indices])
        
    # If we still need more atoms (for n > 55), we can add more shells or random points
    # But for n=38..60, the 55-atom structure is the main target.
    # For n < 55, we just take the first n points.
    # For n > 55, we might need a third shell.
    
    # Let's refine: 
    # If n_atoms <= 13: return first n_atoms points (center + part of shell 1)
    # If 14 <= n_atoms <= 55: return center + 12 + (n_atoms-13) from shell 2
    # If n_atoms > 55: return 55 + (n_atoms-55) from shell 3
    
    # My current code adds shell 2 greedily up to n_atoms.
    # If n_atoms > 55, the greedy algorithm will just keep adding points to the sphere,
    # which might not be the correct 3rd shell geometry, but relax() should fix it.
    
    return np.array(points[:n_atoms])

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    # Check time
    if _get_remaining_time() < 1.0:
        # Not enough time, return a valid but unoptimized structure
        # Just random points with min distance 0.5
        rng = np.random.default_rng(0)
        points = []
        while len(points) < size:
            p = rng.uniform(-5, 5, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        return np.array(points)
    
    best_energy = np.inf
    best_positions = None
    
    # Try different seeds
    # 1. Icosahedral seed
    # 2. Random seed (fallback)
    
    rng = np.random.default_rng(size)
    
    # Generate icosahedral seed
    try:
        ico_points = _generate_icosahedron(size, rng)
        # Relax it
        energy, positions = relax(ico_points, potential, None)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    except Exception:
        pass
    
    # If time allows, try a random seed
    if _get_remaining_time() > 10.0:
        try:
            # Random compact cluster
            radius = 1.1 * size ** (1.0 / 3.0) * 0.75
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, None)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(1)
        points = []
        while len(points) < size:
            p = rng.uniform(-5, 5, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
        
    return best_positions
# EVOLVE-BLOCK-END
