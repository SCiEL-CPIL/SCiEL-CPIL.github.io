"""Improved program using FCC lattice packing for 100 equal spheres in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def construct():
    """
    Generate a face-centered cubic (FCC) lattice of points, filter those inside the cube,
    and select the 100 points that maximize the minimum distance to the boundary.
    """
    def generate_fcc_points(a, n_range):
        """Generate FCC lattice points with lattice constant a."""
        points = []
        for i in range(-n_range, n_range + 1):
            for j in range(-n_range, n_range + 1):
                for k in range(-n_range, n_range + 1):
                    if (i + j + k) % 2 == 0:
                        points.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))
        return np.array(points)
    
    best_points = None
    best_score = 0.0

    # We search for the optimal lattice constant 'a'.
    # The score is min(a/sqrt(2)/2, min_boundary_dist).
    # As 'a' increases, the first term increases, but the number of points decreases,
    # forcing us to pick points closer to the boundary, decreasing the second term.
    # We perform a fine-grained search.
    
    # Initial coarse search to find the region
    for a in np.linspace(0.25, 0.35, 21):
        n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
        points = generate_fcc_points(a, n_range)
        mask = np.all(np.abs(points) <= 0.5, axis=1)
        points = points[mask]
        
        if len(points) < N:
            continue
        
        dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
        order = np.argsort(dist_to_boundary)[::-1]
        selected = points[order[:N]]
        
        # For efficiency, we can estimate min_pair_dist as a/sqrt(2) 
        # but we verify it to be safe.
        tree = cKDTree(selected)
        # Query nearest neighbor for each point
        dists, _ = tree.query(selected, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        min_boundary_dist = np.min(dist_to_boundary[order[:N]])
        
        radius = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if radius > best_score:
            best_score = radius
            best_points = selected

    # Fine-tune around the best 'a' found in coarse search
    if best_points is not None:
        # Refine around the current best a. We need to track which a gave the best score.
        # Let's just re-run a finer grid around the likely optimal range [0.28, 0.32]
        for a in np.linspace(0.28, 0.32, 41):
            n_range = int(np.ceil(0.5 / (a / 2.0))) + 2
            points = generate_fcc_points(a, n_range)
            mask = np.all(np.abs(points) <= 0.5, axis=1)
            points = points[mask]
            
            if len(points) < N:
                continue
            
            dist_to_boundary = np.min(0.5 - np.abs(points), axis=1)
            order = np.argsort(dist_to_boundary)[::-1]
            selected = points[order[:N]]
            
            tree = cKDTree(selected)
            dists, _ = tree.query(selected, k=2)
            min_pair_dist = np.min(dists[:, 1])
            
            min_boundary_dist = np.min(dist_to_boundary[order[:N]])
            
            radius = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if radius > best_score:
                best_score = radius
                best_points = selected

    if best_points is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_points
# EVOLVE-BLOCK-END
