# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 310.0  # Safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _min_distance(positions):
    """
    Vectorized computation of the minimum pairwise distance.
    """
    n = positions.shape[0]
    if n < 2:
        return float('inf')
    
    # Compute pairwise distances efficiently
    # diff[i, j] = positions[i] - positions[j]
    # We only need upper triangle
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    min_dist = np.min(dists[iu])
    return min_dist

def _generate_fcc_cluster(n, seed=42):
    """
    Generate an FCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n ** (1.0/3.0))
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
                        
    if not points:
        return _generate_random_cluster(n, seed)
        
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    cluster = points[sorted_indices[:n]]
    
    # Small jitter to break symmetry
    jitter = rng.uniform(-0.02, 0.02, (n, 3))
    cluster = cluster + jitter
    return cluster

def _generate_bcc_cluster(n, seed=43):
    """
    Generate a BCC-like cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n ** (1.0/3.0))
    points = []
    max_coord = int(math.ceil(radius)) + 2
    
    # BCC basis: (0,0,0), (0.5,0.5,0.5)
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
                        
    if not points:
        return _generate_random_cluster(n, seed)
        
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(distances)
    cluster = points[sorted_indices[:n]]
    
    jitter = rng.uniform(-0.02, 0.02, (n, 3))
    cluster = cluster + jitter
    return cluster

def _generate_icosahedral_cluster(n, seed=44):
    """
    Generate a Mackay icosahedral cluster.
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (normalized to distance 1 from center)
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay layers:
    # Layer 0: Center (1 atom)
    # Layer 1: 12 vertices of icosahedron (12 atoms)
    # Layer 2: 30 edges (30 atoms) - midpoints of edges
    # Layer 3: 20 faces (20 atoms) - centroids of faces
    # Total for full 3 layers: 1 + 12 + 30 + 20 = 63 atoms.
    # For n > 63, we can scale or add a 4th layer (if it exists in Mackay, usually just larger icosahedra).
    # For simplicity, we generate the 63-atom cluster and if n < 63, take the closest n.
    # If n > 63, we scale the 63-atom cluster up slightly or just use it as a seed (relaxation will adjust).
    # Actually, for n up to 80, a scaled icosahedron or a "super-icosahedron" might be needed.
    # Let's stick to the 63-atom core and if n > 63, we'll just pad with random points near the surface or scale.
    # Better: Generate the 63-atom cluster. If n <= 63, take closest n. If n > 63, scale the 63-atom cluster by (n/63)^(1/3) to increase density? No, that increases energy.
    # The 63-atom Mackay cluster is a known local minimum. For n=75, 80, it's a good starting point.
    
    points = [np.zeros(3)] # Center
    
    # Layer 1: Vertices
    for v in verts:
        points.append(v.copy())
        
    # Layer 2: Edges (midpoints)
    # Need to find edges of icosahedron.
    # An edge exists if distance between vertices is constant (sqrt(2) in this coord system? No, let's check).
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2).
    # phi^2 = phi+1. (phi-1)^2 = phi^2 - 2phi + 1 = phi+1 - 2phi + 1 = 2 - phi.
    # d^2 = 1 + 2 - phi + phi + 1 = 4. d = 2.
    # So edges connect vertices with distance 2.
    
    edge_points = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                # Normalize to radius 1? No, midpoints are at distance sqrt(1 - 1/4) = sqrt(3)/2?
                # Wait, the vertices are at distance 1. The midpoints of edges of a regular icosahedron inscribed in unit sphere are at distance sqrt(3)/2?
                # Let's just use the geometric midpoint.
                edge_points.append(mid)
                
    # Remove duplicates in edge_points (each edge is found once)
    # Actually, the loop i<j ensures each pair is unique.
    # But we need to ensure we don't add the same point twice if geometry is weird? No, icosahedron is unique.
    for p in edge_points:
        points.append(p)
        
    # Layer 3: Faces (centroids)
    # Faces are triangles of vertices.
    # A face is defined by 3 vertices that form an equilateral triangle of side 2.
    face_points = []
    for i in range(len(verts)):
        for j in range(i+1, len(verts)):
            for k in range(j+1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if abs(d1 - 2.0) < 1e-6 and abs(d2 - 2.0) < 1e-6 and abs(d3 - 2.0) < 1e-6:
                    centroid = (verts[i] + verts[j] + verts[k]) / 3.0
                    face_points.append(centroid)
                    
    for p in face_points:
        points.append(p)
        
    points = np.array(points)
    
    if len(points) < n:
        # If we need more atoms, we can't easily extend Mackay.
        # We will just scale the existing points to fill the volume?
        # Or just use the 63 points and let relaxation handle the rest?
        # The interface requires n atoms.
        # If n > 63, we will add random points near the surface.
        # This is a bit hacky but provides a structured core.
        rng2 = np.random.default_rng(seed + 100)
        while len(points) < n:
            # Pick a random existing point and add a small offset
            idx = rng2.integers(0, len(points))
            new_p = points[idx] + rng2.normal(0, 0.2, 3)
            points = np.vstack([points, new_p.reshape(1,3)])
    else:
        # Sort by distance to center and take closest n
        distances = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(distances)
        points = points[sorted_indices[:n]]
        
    # Jitter
    jitter = rng.uniform(-0.02, 0.02, (n, 3))
    points = points + jitter
    return points

def _generate_random_cluster(n, seed=45):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 5000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            # Vectorized check against existing points
            if points:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) < 0.6:
                    valid = False
            if valid:
                points.append(p)
        attempts += 1
        
    # If we still don't have enough, just fill with random (might be invalid, but relax will fix or fail)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Generate diverse high-quality templates
    templates = []
    
    try:
        templates.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
        
    try:
        templates.append(_generate_bcc_cluster(n, seed=43))
    except Exception:
        pass
        
    try:
        templates.append(_generate_icosahedral_cluster(n, seed=44))
    except Exception:
        pass
        
    try:
        templates.append(_generate_random_cluster(n, seed=45))
    except Exception:
        pass
        
    if not templates:
        templates.append(_generate_random_cluster(n, seed=0))
        
    # 2. Basin Hopping Strategy
    # For each template, run a short basin hopping sequence.
    # We allocate time proportionally or just run BH on all if time permits.
    # Given 300s total and 6 instances, we have ~50s per instance.
    # BH step: Perturb -> Relax -> Check.
    # Relax is the expensive part.
    
    rng = np.random.default_rng(12345)
    
    for t_idx, template in enumerate(templates):
        if _get_remaining_time() < 15.0:
            break
            
        # Check initial validity
        min_dist = _min_distance(template)
        if min_dist < 0.5:
            continue
            
        # Initialize current state
        try:
            current_energy, current_positions = relax(template, potential, rho)
        except Exception:
            continue
            
        # Check validity of relaxed structure
        min_dist = _min_distance(current_positions)
        if min_dist < 0.5:
            continue
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions.copy()
            
        # Basin Hopping loop
        # We want to do as many steps as possible.
        # Let's estimate time per step.
        # If we have 20s left for this template, and step takes 0.5s, we can do 40 steps.
        # We don't know step time exactly, so we just loop until time runs out.
        
        step_size = 0.3
        # Temperature for Metropolis acceptance (not strictly needed if we only keep best, but helps exploration)
        # We will use a simple "keep if better" or "Metropolis" scheme.
        # Metropolis is better for finding global min.
        temp = 1.0
        
        max_bh_steps = 100 # Safety cap, but time is the real limit
        
        for step in range(max_bh_steps):
            if _get_remaining_time() < 5.0:
                break
                
            # Perturb
            noise = rng.normal(0, step_size, current_positions.shape)
            perturbed = current_positions + noise
            
            # Relax
            try:
                new_energy, new_positions = relax(perturbed, potential, rho)
            except Exception:
                continue
                
            # Check validity
            min_dist = _min_distance(new_positions)
            if min_dist < 0.5:
                continue
                
            # Metropolis Acceptance
            if new_energy < current_energy:
                accept = True
            else:
                delta_e = new_energy - current_energy
                if temp > 1e-6:
                    accept_prob = math.exp(-delta_e / temp)
                    accept = rng.random() < accept_prob
                else:
                    accept = False
                    
            if accept:
                current_energy = new_energy
                current_positions = new_positions
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_positions.copy()
                    
                # Anneal
                temp *= 0.95
                step_size *= 0.98
            else:
                # Even if rejected, we might want to cool down
                temp *= 0.98
                step_size *= 0.99
                
    if best_positions is None:
        # Absolute fallback
        best_positions = templates[0]
        
    return best_positions
# EVOLVE-BLOCK-END
