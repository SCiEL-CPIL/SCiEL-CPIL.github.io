import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lj_energy_and_grad(positions):
    """
    Calculate the total Lennard-Jones energy and the gradient.
    positions: (N, 3) array
    Returns: energy, gradient
    """
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    
    # Mask diagonal (self-interaction)
    np.fill_diagonal(r2, np.inf)
    
    # Avoid division by zero for very close pairs (should not happen with valid input)
    # r2 is squared distance
    r6 = r2**-3
    r12 = r6**2
    
    # Energy: 4 * sum (r^-12 - r^-6)
    # We sum over all pairs i,j. The sum over i!=j counts each pair twice.
    # Standard LJ energy is sum over i<j.
    # So energy = 0.5 * 4 * sum_{i!=j} (r^-12 - r^-6) = 2 * sum_{i!=j} (r^-12 - r^-6)
    energy = 2.0 * np.sum(r12 - r6)
    
    # Gradient
    # dE/dr_i = sum_{j!=i} 24 * (2*r^-14 - r^-8) * (r_i - r_j)
    # Note: r^-14 = r^-12 * r^-2 = r12 / r2
    # r^-8 = r^-6 * r^-2 = r6 / r2
    # Term = 24 * (2*r12/r2 - r6/r2) * diff
    #      = 24/r2 * (2*r12 - r6) * diff
    
    # r2 is (N, N)
    # diff is (N, N, 3)
    
    # Factor for each pair (i,j)
    # We need to be careful with broadcasting.
    # grad_i = sum_j 24 * (2*r12_ij - r6_ij) / r2_ij * (r_i - r_j)
    
    # Create a matrix of coefficients for each pair (i,j)
    # coeff_ij = 24 * (2*r12_ij - r6_ij) / r2_ij
    coeff = 24.0 * (2.0 * r12 - r6) / r2
    
    # Set diagonal to 0 to avoid self-interaction in gradient
    np.fill_diagonal(coeff, 0.0)
    
    # grad = coeff @ diff
    # coeff is (N, N), diff is (N, N, 3)
    # We want sum_j coeff_ij * diff_ij for each i
    # This is equivalent to (coeff * diff).sum(axis=1) if we multiply element-wise
    # coeff is (N, N, 1) effectively when broadcasted to (N, N, 3)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # 1. Construct Mackay Icosahedron
    # Core
    core = np.array([[0.0, 0.0, 0.0]])
    
    # Inner Shell: 12 vertices of Icosahedron
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of a regular icosahedron centered at origin
    # Using the standard coordinates involving phi
    inner_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    inner_verts = inner_verts / np.linalg.norm(inner_verts[0])
    
    # Outer Shell: 42 points
    # 20 face centers
    # Faces are defined by triplets of vertices. 
    # Instead of explicitly listing faces, we can use the fact that face centers of an icosahedron 
    # form a dodecahedron.
    # The vertices of a dodecahedron can be generated from the icosahedron vertices.
    # Alternatively, we can just generate the 42 points by taking the 12 edge midpoints 
    # and 20 face centers of the icosahedron.
    
    # Let's identify the faces and edges of the icosahedron defined by inner_verts.
    # This is a bit tedious to do robustly on the fly. 
    # However, we can use a known property: 
    # The 42 outer atoms in the Mackay icosahedron are located at:
    # - 12 positions corresponding to the edge centers of the inner icosahedron.
    # - 30 positions? No, 42 = 12 (edge centers) + 30 (face centers? No, 20 faces).
    # Wait, the Mackay icosahedron shells are:
    # N=1 (center)
    # N=12 (icosahedron vertices)
    # N=42 (next shell). 
    # The 42 atoms are located at the centers of the 20 faces and the centers of the 12 edges? 
    # 20 + 12 = 32. That's not 42.
    
    # Let's re-verify the Mackay icosahedron structure for N=55.
    # Shell 1: 1 atom.
    # Shell 2: 12 atoms (Icosahedron).
    # Shell 3: 42 atoms.
    # The 42 atoms are arranged in a truncated icosahedron? No.
    # Actually, the 42 atoms form the vertices of a "Mackay" shell.
    # The positions are:
    # - 20 face centers of the inner icosahedron.
    # - 12 edge centers of the inner icosahedron.
    # - 10 additional points?
    
    # Let's look at the geometry more carefully.
    # A common way to generate the N=55 Mackay icosahedron is:
    # 1. Center
    # 2. 12 Icosahedron vertices
    # 3. 42 points which are the vertices of a larger, distorted structure.
    
    # Actually, it is easier to use the known coordinates or a generation method.
    # The 42 outer atoms are located at the centers of the 20 triangular faces and the 12 edges of the icosahedron?
    # 20 + 12 = 32.
    # Where are the other 10?
    # Ah, the "Mackay" icosahedron is a specific stacking.
    # The third shell in a Mackay icosahedron consists of 42 atoms.
    # These 42 atoms are located at:
    # - 20 face centers of the icosahedron.
    # - 12 edge centers of the icosahedron.
    # - 10 points corresponding to the 5-fold axes?
    
    # Let's use a different approach. 
    # We can generate the N=55 structure by taking the N=13 icosahedron and adding a shell.
    # Or, we can use the fact that the global minimum is very close to the ideal Mackay icosahedron.
    
    # Let's try to generate the 42 points properly.
    # The 42 points are the vertices of a "Truncated Icosahedron"? No, that has 60 vertices.
    
    # Let's use the coordinates from the Cambridge Cluster Database or a reliable source.
    # Since I cannot fetch external data, I will use a robust generation method.
    
    # Method:
    # 1. Generate 12 icosahedron vertices.
    # 2. Generate 20 face centers.
    # 3. Generate 12 edge centers.
    # 4. The remaining 10 points are located on the 5-fold symmetry axes, further out?
    
    # Actually, a simpler and more robust way to get a very good starting point is to use 
    # the "Wulff construction" or just relax a slightly perturbed icosahedron.
    
    # However, the most reliable "known" structure for N=55 is the Mackay icosahedron.
    # Let's use the fact that the outer shell atoms are close to the inner shell atoms.
    
    # Alternative: Use the N=13 icosahedron and add 42 atoms in a spherical shell around it, 
    # then relax. The relaxation will push them into the correct positions.
    
    # Let's try this:
    # 1. Core: 1 atom.
    # 2. Inner: 12 atoms at r=1.2.
    # 3. Outer: 42 atoms distributed on a sphere of radius r=1.8.
    
    # To distribute 42 atoms on a sphere, we can use a Fibonacci sphere or a grid.
    # But we want to respect the icosahedral symmetry to help the optimizer.
    
    # Let's use the 20 face centers and 12 edge centers as a subset, and add 10 more.
    # The 10 extra points are likely on the 5-fold axes.
    
    # Let's just use a uniform distribution on the sphere for the outer 42, 
    # but perturb it slightly to break symmetry if needed, or rely on the symmetry.
    
    # Better: Use the icosahedral symmetry group to generate the outer shell.
    # The 42 atoms form orbits under the icosahedral group.
    # Orbit sizes: 12 (vertices), 20 (faces), 30 (edges), 60 (general).
    # 42 = 12 + 30? Or 20 + 22?
    # Actually, the 42 atoms in the Mackay icosahedron are:
    # - 12 atoms at the edge centers of the inner icosahedron.
    # - 30 atoms at the face centers? No, there are only 20 faces.
    
    # Let's check the literature value for the structure.
    # The N=55 Mackay icosahedron has:
    # - 1 atom at center.
    # - 12 atoms at vertices of icosahedron (radius ~1.2).
    # - 42 atoms in the next shell.
    # The 42 atoms are located at:
    # - 20 face centers of the icosahedron.
    # - 12 edge centers of the icosahedron.
    # - 10 points on the 5-fold axes?
    
    # Let's assume the 42 points are:
    # 12 edge centers + 30 points?
    
    # Let's use a simpler heuristic:
    # Generate 55 points on a sphere with varying radii, then relax.
    # To be safe, I will use the icosahedron vertices and face centers, and add the rest randomly on the sphere, then relax.
    
    # Actually, the most robust method is to use the known "Mackay" coordinates if I can reconstruct them.
    # The 42 outer atoms are the vertices of a "Mackay" shell.
    # I will use the 20 face centers and 12 edge centers (32 points) and add 10 points on the 5-fold axes.
    
    # 5-fold axes pass through vertices and face centers.
    # There are 6 axes (12 vertices / 2).
    # Wait, 10 points on 5-fold axes?
    
    # Let's try a different strategy: 
    # Start with the N=13 icosahedron.
    # Add 42 atoms at random positions on a sphere of radius 1.8.
    # Relax.
    # This is stochastic but likely to converge to the global minimum if the basin is large.
    # To improve reliability, I will use multiple restarts if time permits.
    
    # However, a deterministic construction is preferred.
    # Let's use the icosahedral symmetry.
    # The 42 atoms can be generated by taking the 12 edge centers and 30 "face edge" midpoints? No.
    
    # Let's use the following known set of 42 points for the Mackay icosahedron:
    # They are the vertices of a "Truncated Icosahedron" minus 18 vertices? No.
    
    # I will use a robust random initialization on a spherical shell, guided by the inner shell.
    # 1. Core: 1
    # 2. Inner: 12 (Icosahedron vertices)
    # 3. Outer: 42 (Random on sphere r=1.8)
    
    # To make it deterministic and symmetric, I will use a Fibonacci sphere for the 42 outer points.
    
    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / (n - 1.)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    # Initialize
    inner_radius = 1.2
    outer_radius = 1.8
    
    positions = np.vstack([
        core,
        inner_verts * inner_radius,
        fibonacci_sphere(42) * outer_radius
    ])
    
    # Relax
    result = minimize(
        lambda x: lj_energy_and_grad(x.reshape(N, 3))[0],
        positions.flatten(),
        jac=lambda x: lj_energy_and_grad(x.reshape(N, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    best_pos = result.x.reshape(N, 3)
    best_energy = result.fun
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # Try a few more random restarts if time allows, to ensure we didn't get stuck in a bad local minimum
    # (Though Mackay is the global min, the random outer shell might collapse to a bad structure)
    
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
            
        # Perturb the outer shell
        new_positions = best_pos.copy()
        # Add noise to outer 42 atoms
        noise = np.random.normal(0, 0.5, (42, 3))
        new_positions[13:] += noise
        
        res = minimize(
            lambda x: lj_energy_and_grad(x.reshape(N, 3))[0],
            new_positions.flatten(),
            jac=lambda x: lj_energy_and_grad(x.reshape(N, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_pos = res.x.reshape(N, 3)
            
    return best_pos

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 55

def lj_energy_and_grad(positions):
    """
    Calculate the total Lennard-Jones energy and the gradient.
    positions: (N, 3) array
    Returns: energy, gradient
    """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    
    np.fill_diagonal(r2, np.inf)
    
    r6 = r2**-3
    r12 = r6**2
    
    energy = 2.0 * np.sum(r12 - r6)
    
    coeff = 24.0 * (2.0 * r12 - r6) / r2
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Construct Mackay Icosahedron
    core = np.array([[0.0, 0.0, 0.0]])
    
    phi = (1 + np.sqrt(5)) / 2
    inner_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    inner_verts = inner_verts / np.linalg.norm(inner_verts[0])
    
    def fibonacci_sphere(n):
        points = []
        phi_fib = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / (n - 1.)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi_fib * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    inner_radius = 1.2
    outer_radius = 1.8
    
    positions = np.vstack([
        core,
        inner_verts * inner_radius,
        fibonacci_sphere(42) * outer_radius
    ])
    
    result = minimize(
        lambda x: lj_energy_and_grad(x.reshape(N, 3))[0],
        positions.flatten(),
        jac=lambda x: lj_energy_and_grad(x.reshape(N, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    best_pos = result.x.reshape(N, 3)
    best_energy = result.fun
    
    if time.time() > deadline:
        return best_pos
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        new_positions = best_pos.copy()
        noise = np.random.normal(0, 0.5, (42, 3))
        new_positions[13:] += noise
        
        res = minimize(
            lambda x: lj_energy_and_grad(x.reshape(N, 3))[0],
            new_positions.flatten(),
            jac=lambda x: lj_energy_and_grad(x.reshape(N, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_pos = res.x.reshape(N, 3)
            
    return best_pos
# EVOLVE-BLOCK-END
