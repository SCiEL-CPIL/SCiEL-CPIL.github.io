"""Rewritten program using direct FCC lattice construction."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit to be safe
    
    # The Lennard-Jones minimum distance is 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    
    # We want to construct the Face-Centered Cubic (FCC) lattice fragment.
    # The FCC lattice can be generated from integer coordinates (i, j, k) 
    # where i + j + k is even.
    # The nearest neighbor distance in this lattice (before scaling) is sqrt(2).
    # We scale the coordinates so that the nearest neighbor distance becomes r_min.
    # Scaling factor s such that s * sqrt(2) = r_min  =>  s = r_min / sqrt(2)
    
    scale = r_min / np.sqrt(2.0)
    
    # Generate a sufficiently large box of candidate FCC points
    # For N=38, the cluster is small. A box of +/- 4 or 5 is more than enough.
    # Max radius of 38 atoms in FCC is roughly ~3-4 layers.
    max_coord = 5
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    # Convert to numpy array
    pts = np.array(points, dtype=float)
    pts *= scale
    
    # Sort by distance to the origin to form a compact cluster
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first N points
    cluster = pts[sorted_indices[:N]]
    
    # Try to relax using the provided science_chemistry.relax function
    # This should converge to the exact local minimum of the FCC fragment,
    # which is the global minimum for LJ38.
    try:
        import science_chemistry
        # Check time
        if time.time() < deadline:
            relaxed_cluster = science_chemistry.relax(cluster)
            # Ensure the result is valid (finite, etc.)
            if relaxed_cluster is not None and np.all(np.isfinite(relaxed_cluster)):
                cluster = relaxed_cluster
    except Exception as e:
        # If relaxation fails or times out, return the initial FCC fragment
        pass
        
    # Final time check
    if time.time() > deadline:
         # This shouldn't happen if relax is fast, but just in case
         pass

    return cluster
# EVOLVE-BLOCK-END
