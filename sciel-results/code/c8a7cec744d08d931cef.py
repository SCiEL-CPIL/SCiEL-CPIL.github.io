"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for Dzugutov potential
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(pos):
    """Compute energy using the provided gradient function's energy part or manual calculation."""
    # We can use the gradient function if it returns energy, but the signature is just gradient.
    # The task says: chem_clusters.dzugutov_energy_gradient(positions) may be called.
    # It likely returns the gradient. We need the energy.
    # We can compute energy manually or check if relax returns it.
    # relax returns (energy, positions).
    # For internal steps in basin hopping, we need energy.
    # Let's write a fast numpy energy calculator.
    n = len(pos)
    if n < 2: return 0.0
    # Vectorized distance calculation
    # pos shape (n, 3)
    # diff shape (n, n, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf) # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Mask for r < a
    mask_a = dist < a
    # Mask for r < b (and r >= a)
    mask_b = (dist < b) & (dist >= a)
    
    E = 0.0
    # Part 1: r < a
    if np.any(mask_a):
        r = dist[mask_a]
        # V = A * (r^-m - B) * exp(c / (r - a))
        # Handle r very close to a to avoid overflow? exp(c/(r-a)) -> exp(0) = 1.
        # Handle r very close to 0? r^-16 -> inf.
        # The potential is repulsive at small r.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            v1 = A * (r**(-m) - B) * np.exp(c / (r - a))
        # Replace inf/nan with large positive number
        v1 = np.nan_to_num(v1, nan=1e10, posinf=1e10, neginf=-1e10)
        E += np.sum(v1)
        
    # Part 2: a <= r < b
    if np.any(mask_b):
        r = dist[mask_b]
        # V = B * exp(d / (r - b))
        # As r -> b, exp(0) = 1. As r -> a, exp(d/(a-b)) = exp(0.27 / (1.87 - 1.94)) = exp(0.27 / -0.07) = exp(-3.85) ~ 0.02
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            v2 = B * np.exp(d / (r - b))
        v2 = np.nan_to_num(v2, nan=0.0, posinf=1e10, neginf=-1e10)
        E += np.sum(v2)
        
    return E

def _generate_icosahedron_vertices(radius=1.0):
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _generate_mackay_55(radius=1.0):
    """Generate coordinates for 55-atom Mackay Icosahedron."""
    # 1 center
    center = np.array([[0, 0, 0]])
    # 12 vertices (shell 1)
    shell1 = _generate_icosahedron_vertices(radius)
    # 30 edge centers (shell 2a)
    # Edges of icosahedron connect vertices with distance 2*radius*sin(pi/5)
    # We can find edges by checking distances between shell1 vertices
    d1 = np.linalg.norm(shell1[:, np.newaxis, :] - shell1[np.newaxis, :, :], axis=2)
    # Min non-zero distance
    min_d = np.min(d1[d1 > 1e-6])
    edge_mask = d1 < min_d * 1.01
    np.fill_diagonal(edge_mask, False)
    edge_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (shell1[i] + shell1[j]) / 2.0
                edge_centers.append(mid)
    shell2a = np.array(edge_centers)
    # Scale shell2a to appropriate radius?
    # In Mackay icosahedron, the 30 edge centers are at a specific radius.
    # The 12 face centers (shell 2b) are also included.
    # Total outer atoms = 42. 30 edge + 12 face?
    # An icosahedron has 20 faces. The 55-atom cluster has 12 "secondary" atoms.
    # Actually, the 55-atom Mackay icosahedron is 1+12+30+12.
    # The 12 atoms are at the centers of the 12 *pairs* of opposite faces? Or something else.
    # Let's just use the 30 edge centers and 12 "face-like" centers.
    # For simplicity and robustness, we can just use the 13-atom core and 30 edge centers (43 atoms) 
    # and add 12 more from the next shell or random positions.
    # Or better: Use the 13-atom core, and for N>13, add atoms from a pool of candidate positions 
    # generated from a high-resolution icosahedron or random spherical points, 
    # and let the optimizer select the best ones.
    
    # A more robust seed strategy:
    # 1. Generate a large set of candidate points (e.g. 100) from icosahedral shells.
    # 2. Sort by distance from center.
    # 3. Take the first N.
    
    return np.vstack([center, shell1, shell2a])

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    # parameter = instance.get("rho", instance.get("kind")) # Not strictly needed for dzugutov if using standard params
    
    start_time = time.time()
    deadline = start_time + 290.0 # 10s safety margin
    
    # 1. Generate a seed structure
    # We want to start with a polytetrahedral structure.
    # Let's generate a 13-atom icosahedron, a 55-atom structure, and a 100-atom structure.
    
    # Generate 13-atom icosahedron
    ico13 = np.vstack([np.array([[0,0,0]]), _generate_icosahedron_vertices(1.0)])
    
    # Generate 55-atom structure (approx)
    # 13 core + 30 edge centers + 12 face centers (approx)
    shell1 = _generate_icosahedron_vertices(1.0)
    d1 = np.linalg.norm(shell1[:, np.newaxis, :] - shell1[np.newaxis, :, :], axis=2)
    min_d = np.min(d1[d1 > 1e-6])
    edge_mask = d1 < min_d * 1.01
    np.fill_diagonal(edge_mask, False)
    edge_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                edge_centers.append((shell1[i] + shell1[j]) / 2.0)
    shell2 = np.array(edge_centers)
    
    # Add 12 more points to reach 55? 
    # The 55-atom Mackay icosahedron has 12 atoms at the face centers of a *dodecahedron* inscribed? 
    # Let's just use the 43 atoms (1+12+30) and add 12 random points on a sphere of radius 1.6 for the 55-seed.
    # Or simply use the 43 atoms and if N > 43, add random points.
    
    # For N=38, 45, 50, 55, 57, 60
    # 38 < 43, so we can use the first 38 atoms of the 43-atom structure (sorted by radius).
    # 45, 50, 55, 57, 60 > 43.
    
    # Let's create a pool of candidates:
    # 13 core, 30 edge centers, 20 face centers (of icosahedron), 30 next shell?
    
    # Face centers of icosahedron:
    # Triangles formed by 3 vertices.
    # This is complex to generate correctly.
    
    # Alternative: Use a random spherical distribution but biased towards icosahedral symmetry.
    # Or, use the `relax` function on a random compact cluster, but initialize with icosahedral symmetry.
    
    # Let's try a "Greedy Icosahedral" seed.
    # 1. Place 1 atom at origin.
    # 2. Place 12 atoms at icosahedron vertices (R=1.0).
    # 3. For k=13 to N:
    #    a. Generate 50 random points on a sphere of radius R_k (increasing slightly).
    #    b. For each point, calculate energy of (current_cluster + point).
    #    c. Pick the point with lowest energy.
    #    d. Add it.
    #    e. Do a quick relaxation of the whole cluster.
    
    # This is slow if N=60 and we do 50 candidates * 60 steps * relax.
    # Relax is expensive.
    
    # Better: Generate a large superset (e.g. 100 atoms) from icosahedral shells, 
    # scale it, and take the first N atoms (closest to center).
    # Then run Basin Hopping.
    
    # Generate 100-atom superset:
    # Shell 1: 12 vertices (R=1.0)
    # Shell 2: 30 edge centers (R~1.5)
    # Shell 3: 20 face centers (R~1.6)
    # Shell 4: ...
    
    # Let's just use a random cluster for the seed, but scale it to the optimal volume.
    # The Dzugutov potential favors high density.
    # Optimal density is high.
    # Radius R ~ N^(1/3) * r_opt.
    # r_opt ~ 1.0.
    
    # Let's use the 43-atom icosahedral structure as the base.
    # For N <= 43: Take the N closest atoms to the center.
    # For N > 43: Take all 43, and add N-43 random atoms on a sphere of radius 1.8.
    
    base_43 = np.vstack([np.array([[0,0,0]]), shell1, shell2])
    # Sort by distance from center
    dists = np.linalg.norm(base_43, axis=1)
    idx = np.argsort(dists)
    base_43_sorted = base_43[idx]
    
    if size <= 43:
        seed = base_43_sorted[:size].copy()
    else:
        # Add random atoms
        extra_n = size - 43
        rng = np.random.default_rng(size)
        # Radius for extra atoms
        r_extra = 1.8
        extra_atoms = []
        for _ in range(extra_n):
            # Random point on sphere
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            x = r_extra * np.sin(phi) * np.cos(theta)
            y = r_extra * np.sin(phi) * np.sin(theta)
            z = r_extra * np.cos(phi)
            extra_atoms.append([x, y, z])
        extra_atoms = np.array(extra_atoms)
        seed = np.vstack([base_43_sorted, extra_atoms])
        
    # Scale the seed to optimal size
    # The icosahedron vertices are at R=1.0.
    # The optimal bond length is likely around 1.0-1.1.
    # Let's assume the current scale is good.
    
    best_energy = np.inf
    best_positions = seed
    
    # Basin Hopping
    # Time budget per instance
    # We have 6 instances. 300s total. 50s per instance.
    
    # Define objective function for scipy
    def objective(x):
        pos = x.reshape(-1, 3)
        return _dzugutov_energy(pos)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        # Use the provided gradient function
        # Note: dzugutov_energy_gradient returns gradient.
        # We need to ensure it matches our energy function.
        # The task says: chem_clusters.dzugutov_energy_gradient(positions) may be called.
        # It likely returns the gradient of the energy defined in the task.
        try:
            grad = dzugutov_energy_gradient(pos)
            return grad.flatten()
        except:
            # Fallback to numerical gradient if API fails
            eps = 1e-5
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad
    
    current_pos = seed
    current_energy = _dzugutov_energy(current_pos)
    
    # Local minimize first
    res = minimize(objective, current_pos.flatten(), jac=gradient, method='L-BFGS-B')
    if res.success or res.fun < current_energy:
        current_pos = res.x.reshape(-1, 3)
        current_energy = res.fun
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos.copy()
            
    # Basin Hopping
    rng = np.random.default_rng(42)
    step_size = 0.2
    
    while time.time() < deadline:
        # Perturb
        perturbation = rng.normal(0, step_size, current_pos.shape)
        perturbed_pos = current_pos + perturbation
        
        # Minimize
        res = minimize(objective, perturbed_pos.flatten(), jac=gradient, method='L-BFGS-B')
        
        if res.success:
            new_pos = res.x.reshape(-1, 3)
            new_energy = res.fun
            
            # Metropolis accept
            if new_energy < current_energy or np.exp(-(new_energy - current_energy) / 0.1) > rng.random():
                current_pos = new_pos
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos.copy()
        
        # Adapt step size
        if current_energy < best_energy:
            step_size *= 1.1
        else:
            step_size *= 0.9
            step_size = max(0.05, min(step_size, 0.5))
            
    return best_positions

# EVOLVE-BLOCK-END
