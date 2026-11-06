"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 290.0 # Safety margin for 300s limit

def _energy_and_grad(pos, rho):
    """Wrapper for scipy minimize."""
    # Ensure pos is flat
    pos_flat = pos.reshape(-1)
    # Compute energy and gradient using the provided helper
    # Note: The task description says chem_clusters.morse_energy_gradient(positions, rho) may be called
    # We assume it returns (energy, gradient) or we compute manually if not available/consistent.
    # Let's try calling the provided function first. If it fails, we fall back to manual computation.
    try:
        e, g = morse_energy_gradient(pos.reshape(-1, 3), rho)
        return e, g.reshape(-1)
    except Exception:
        # Fallback manual computation
        n = len(pos_flat) // 3
        p = pos_flat.reshape(n, 3)
        e = 0.0
        g = np.zeros_like(p)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = p[i] - p[j]
                r = np.linalg.norm(r_vec)
                if r < 1e-6:
                    continue
                exp_term = np.exp(rho * (1.0 - r))
                pair_e = exp_term * (exp_term - 2.0)
                e += pair_e
                # Gradient of pair_e wrt r: d/dr [exp(rho(1-r))(exp(rho(1-r))-2)]
                # Let u = rho(1-r), du/dr = -rho
                # f(u) = e^u(e^u-2) = e^(2u) - 2e^u
                # df/du = 2e^(2u) - 2e^u = 2e^u(e^u-1)
                # dr_i/dr = r_vec_i / r
                # g_i += df/du * du/dr * dr_i/dr = 2e^u(e^u-1) * (-rho) * (r_vec/r)
                # g_j += -g_i
                grad_r = 2.0 * exp_term * (exp_term - 1.0) * (-rho)
                grad_vec = grad_r * (r_vec / r)
                g[i] += grad_vec
                g[j] -= grad_vec
        return e, g.reshape(-1)

def _make_icosahedron(n):
    """Generate a Mackay icosahedron-like structure."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    points = []
    # Center atom
    if n > 0:
        points.append(np.array([0,0,0]))
    
    # Shell 1: 12 atoms
    if n > 12:
        for v in verts:
            if len(points) < n:
                points.append(v)
    
    # Shell 2: 30 atoms (faces of icosahedron)
    # This is a simplified approach: just add more layers radially
    # For a better structure, we'd need specific coordinates, but radial expansion works reasonably well for local relaxation
    if n > 12:
        # Generate points on sphere for shell 2
        # Use Fibonacci sphere for uniform distribution
        def fib_sphere(n_points, radius):
            points = []
            phi = np.pi * (3.0 - np.sqrt(5.0))
            for i in range(n_points):
                y = 1 - (i / (n_points - 1)) * 2
                r = np.sqrt(1 - y*y)
                th = phi * i
                x = np.cos(th) * r
                z = np.sin(th) * r
                points.append(np.array([x, y, z]) * radius)
            return points
        
        shell2 = fib_sphere(30, 2.0)
        for p in shell2:
            if len(points) < n:
                points.append(p)
                
    # If we still need more, add random points on a larger sphere
    while len(points) < n:
        # Random direction
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        points.append(v * 3.0)
        
    return np.array(points[:n])

def _make_fcc(n):
    """Generate an FCC-like cluster."""
    a = 1.0 # lattice constant, will be scaled
    points = []
    # Generate points in a cubic box around origin
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = [
        [0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]
    ]
    
    # Estimate size needed
    # Volume ~ n / density. FCC density is high.
    # Try expanding a cube
    max_coord = 5
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            for z in range(-max_coord, max_coord+1):
                for b in basis:
                    p = np.array([x, y, z]) + np.array(b)
                    # Keep only points within a sphere to make it compact
                    if np.linalg.norm(p) < max_coord * 0.8:
                        points.append(p)
    
    # Scale to ensure nearest neighbor distance is ~1.0
    # In FCC, nearest neighbor distance is a/sqrt(2).
    # Our basis uses a=1, so NN is 1/sqrt(2) ~ 0.707.
    # We want NN=1.0, so scale by sqrt(2).
    points = np.array(points) * np.sqrt(2)
    
    # Sort by distance to center to get the most compact core
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices]
    
    # If we have more than n, take the first n
    if len(points) > n:
        points = points[:n]
    else:
        # Pad with random points if not enough (unlikely for n<=80 with max_coord=5)
        while len(points) < n:
            p = np.random.randn(3) * 1.0
            points = np.vstack([points, p])
            
    return points

def _make_hcp(n):
    """Generate an HCP-like cluster (approximated as FCC with slight distortion or just use FCC if HCP is hard to generate simply)."""
    # For simplicity and robustness, we often use FCC as a proxy for close-packed structures.
    # HCP is a local minimum for some N, but FCC is often competitive.
    return _make_fcc(n)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # List of starting geometries
    starts = []
    
    # 1. Icosahedral
    try:
        starts.append(_make_icosahedron(size))
    except Exception:
        pass
        
    # 2. FCC
    try:
        starts.append(_make_fcc(size))
    except Exception:
        pass
        
    # 3. HCP (using FCC proxy for now, or distinct if easy)
    try:
        starts.append(_make_hcp(size))
    except Exception:
        pass
        
    # 4. Random compact (seed-like but better)
    rng = np.random.default_rng(size)
    radius = 1.0 * size ** (1.0 / 3.0) * 0.8
    while len(starts) < 4:
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        if len(points) == size:
            starts.append(np.array(points))
            
    best_energy = np.inf
    best_pos = starts[0]
    
    # Optimization Loop
    # We will try to optimize each start.
    # Given time constraints, we limit the number of restarts.
    max_restarts = 10
    current_restart = 0
    
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Add small noise to break symmetry
        noisy_start = start + rng.normal(0, 0.05, start.shape)
        
        # Run local optimization
        try:
            res = minimize(
                _energy_and_grad,
                noisy_start.reshape(-1),
                args=(parameter,),
                jac=True,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-6}
            )
            
            if res.success:
                e = res.fun
                if e < best_energy:
                    best_energy = e
                    best_pos = res.x.reshape(-1, 3)
        except Exception:
            pass

        # Try a few random perturbations of the best found so far
        if current_restart < max_restarts:
             # Check time
             if time.time() - START_TIME > TIME_LIMIT - 10:
                 break
             
             perturbed = best_pos + rng.normal(0, 0.1, best_pos.shape)
             try:
                 res = minimize(
                     _energy_and_grad,
                     perturbed.reshape(-1),
                     args=(parameter,),
                     jac=True,
                     method='BFGS',
                     options={'maxiter': 200, 'gtol': 1e-5}
                 )
                 if res.success and res.fun < best_energy:
                     best_energy = res.fun
                     best_pos = res.x.reshape(-1, 3)
             except Exception:
                 pass
             current_restart += 1

    return best_pos
# EVOLVE-BLOCK-END
