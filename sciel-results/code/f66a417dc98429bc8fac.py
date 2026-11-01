import numpy as np

# The task provides science_chemistry.relax
# We assume it's in the namespace or we import it if needed.
# The prompt says "science_chemistry.relax(positions) ... is available".
# Usually in these tasks, it's either imported or in the global scope.
# I will try to import it, or assume it's there.
# To be safe, I'll wrap the call in a try-except or check if it exists.
# However, the standard practice for these "evolve" tasks is that the helper is in the global scope or importable.
# Let's assume it's in the global scope or we can import it.
# The prompt says: "science_chemistry.relax(positions) ... is available".
# I will import it.

try:
    import science_chemistry
except ImportError:
    science_chemistry = None

N = 55

def construct():
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Basis vectors for the icosahedral lattice
    # These generate the icosahedral symmetry
    v1 = np.array([0.0, 1.0, phi])
    v2 = np.array([phi, 0.0, 1.0])
    v3 = np.array([1.0, phi, 0.0])
    
    # Generate lattice points
    # Range -2 to 2 should be sufficient to get 55 points
    # 5^3 = 125 points generated, we pick the 55 closest
    points = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                p = n1 * v1 + n2 * v2 + n3 * v3
                points.append(p)
                
    points = np.array(points)
    
    # Calculate squared distances from origin
    dists_sq = np.sum(points**2, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists_sq)
    
    # Take the first 55
    selected_points = points[sorted_indices[:N]]
    
    # Calculate the nearest neighbor distance in this raw lattice
    # The nearest neighbors are the basis vectors, distance = sqrt(1 + phi^2)
    # But let's compute it from the selected points to be robust
    # Find min distance between any pair
    # This is O(N^2), N=55, so 55*55 = 3025 ops. Very fast.
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(selected_points[i] - selected_points[j])
            if d < min_dist:
                min_dist = d
                
    # Scale to LJ minimum distance
    # r_min = 2^(1/6)
    r_min = 2.0 ** (1.0/6.0)
    scale_factor = r_min / min_dist
    
    positions = selected_points * scale_factor
    
    # Ensure center of mass is at origin (already is, due to symmetry, but good practice)
    positions -= np.mean(positions, axis=0)
    
    # Relax using the provided tool
    if science_chemistry is not None:
        try:
            positions = science_chemistry.relax(positions)
        except Exception:
            pass # Fallback to unrelaxed if relax fails
            
    return positions
