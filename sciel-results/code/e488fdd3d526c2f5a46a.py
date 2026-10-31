import numpy as np
import time
import science_chemistry

N = 38

def _lj_energy(pos):
    """Calculate Lennard-Jones energy for a set of positions."""
    # pos shape (N, 3)
    # Compute all pair distances
    # Using broadcasting
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist_sq, 1.0)
    
    # LJ potential: 4 * (r^-12 - r^-6)
    # r^-6 = 1 / r^6 = 1 / (dist_sq^3)
    inv_r6 = 1.0 / (dist_sq**3)
    inv_r12 = inv_r6**2
    
    # Sum over all pairs, divide by 2 to avoid double counting
    # Diagonal is 0 energy contribution (we set dist_sq=1, so 4*(1-1)=0)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def _is_valid(pos):
    """Check if all pair distances are at least 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = np.min(dist_sq)
    return min_dist_sq >= 0.5**2

def _generate_fcc_38(variant=0):
    """Generate a 38-atom FCC-like cluster."""
    # FCC lattice constant a such that nearest neighbor d = 2^(1/6)
    # d = a / sqrt(2) => a = d * sqrt(2)
    d = 2.0 ** (1.0/6.0)
    a = d * np.sqrt(2.0)
    
    # Generate FCC points
    # Basis vectors:
    # a1 = (0, a/2, a/2)
    # a2 = (a/2, 0, a/2)
    # a3 = (a/2, a/2, 0)
    # Points: n1*a1 + n2*a2 + n3*a3
    # x = a/2 * (n2 + n3)
    # y = a/2 * (n1 + n3)
    # z = a/2 * (n1 + n2)
    
    points = []
    # Range of n to cover enough points
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                x = a/2.0 * (n2 + n3)
                y = a/2.0 * (n1 + n3)
                z = a/2.0 * (n1 + n2)
                points.append([x, y, z])
                
    points = np.array(points)
    
    # Calculate distances from origin
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists)
    points_sorted = points[sorted_indices]
    
    # We need 38 atoms.
    # The 27th atom is the last of the 6-atom shell (dist sqrt(2)*d).
    # The 28th to 51st are the 24-atom shell (dist sqrt(3)*d).
    # We take the first 27, and then 11 from the next 24.
    
    core = points_sorted[:27]
    outer = points_sorted[27:51]
    
    # Randomly select 11 from outer to break symmetry and create diversity
    # Use variant to seed the random selection
    np.random.seed(variant)
    outer_indices = np.random.choice(len(outer), size=11, replace=False)
    selected_outer = outer[outer_indices]
    
    cluster = np.vstack((core, selected_outer))
    
    # Add small jitter to help escape symmetry traps if needed
    # But for FCC, symmetry is good.
    # Let's return the exact FCC points for now.
    return cluster

def _generate_icosahedral_38():
    """Generate an icosahedral-like cluster."""
    # 1 center
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 12 vertices of icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    scale = 2.0 ** (1.0/6.0) # Approximate nearest neighbor distance
    
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize and scale
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * scale
    
    # 12 edge midpoints (approximate)
    # We need 12 more atoms. In a 55-atom icosahedron, the second shell is 12.
    # For 38, we might take 12 from the second shell and 12 from the third?
    # 1 + 12 + 12 + 12 + 1 = 38.
    # Let's generate a larger icosahedral cluster and take the closest 38.
    
    # Generate icosahedral lattice points?
    # Simpler: Just use the 13-atom icosahedron and add random points in a shell?
    # No, let's just use the 13-atom core and add 25 random points in a sphere of radius 2*d.
    
    # This is a weak guess. Let's skip complex icosahedral generation and rely on FCC and Random.
    # But icosahedral is a strong local minimum.
    # Let's just return the 13-atom icosahedron + 25 random points in a shell.
    
    shell_points = []
    np.random.seed(42)
    for _ in range(25):
        # Random point on sphere of radius 1.5*d to 2.0*d
        r = np.random.uniform(1.5 * scale, 2.0 * scale)
        u = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.sqrt(1 - u**2) * np.cos(theta)
        y = r * np.sqrt(1 - u**2) * np.sin(theta)
        z = r * u
        shell_points.append([x, y, z])
        
    shell_points = np.array(shell_points)
    return np.vstack((center, verts, shell_points))

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin for 150s limit
    
    best_energy = np.inf
    best_pos = None
    
    # List of initial configurations
    configs = []
    
    # 1. FCC 38 variants
    for v in range(5):
        configs.append(_generate_fcc_38(v))
        
    # 2. Icosahedral-like
    configs.append(_generate_icosahedral_38())
    
    # 3. Random configurations
    for _ in range(10):
        np.random.seed(np.random.randint(10000))
        # Random points in a sphere of radius 3.0
        # Uniform in volume
        radii = np.random.uniform(0, 3.0, size=(N, 1))
        u = np.random.uniform(-1, 1, size=(N, 1))
        theta = np.random.uniform(0, 2 * np.pi, size=(N, 1))
        x = radii * np.sqrt(1 - u**2) * np.cos(theta)
        y = radii * np.sqrt(1 - u**2) * np.sin(theta)
        z = radii * u
        configs.append(np.hstack((x, y, z)))
        
    # 4. Seed configuration (Cubic)
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    seed_pos = np.array(points[:N], dtype=float) * 1.12
    configs.append(seed_pos)
    
    # Relax each configuration
    for i, pos in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Check initial validity
        if not _is_valid(pos):
            # Perturb if invalid
            pos = pos + np.random.normal(0, 0.1, pos.shape)
            if not _is_valid(pos):
                continue
                
        try:
            # Use the provided relax function
            relaxed = science_chemistry.relax(pos)
            
            # Check validity after relaxation
            if _is_valid(relaxed):
                energy = _lj_energy(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            pass
            
    # If no valid result found, return the seed (or a valid random one)
    if best_pos is None:
        # Return a valid random configuration as fallback
        np.random.seed(0)
        pos = np.random.uniform(-2, 2, size=(N, 3))
        # Ensure validity
        while not _is_valid(pos):
             pos = np.random.uniform(-2, 2, size=(N, 3))
        best_pos = pos
        
    return best_pos
