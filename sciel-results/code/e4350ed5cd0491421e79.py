import numpy as np
import itertools
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # 10 minutes
    
    # 1. Generate Dodecahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    for a in [-1/phi, 1/phi]:
        for b in [-phi, phi]:
            verts.append([0, a, b])
            verts.append([a, b, 0])
            verts.append([b, 0, a])
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # 2. Find best subset of 14
    best_score = 0
    best_subset_idx = None
    n_verts = 20
    
    for combo in itertools.combinations(range(n_verts), 14):
        pts = verts[list(combo)]
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        score = (d_min / d_max) ** 2
        if score > best_score:
            best_score = score
            best_subset_idx = combo
            
    initial_points = verts[list(best_subset_idx)].copy()
    
    # 3. Particle Swarm Optimization on the sphere
    # We optimize the minimum pairwise distance directly
    # Since points are on the unit sphere, d_max <= 2, so maximizing d_min
    # also helps maximize the ratio
    
    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms
    
    def compute_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    # PSO parameters
    n_particles = 30
    n_iterations = 200
    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive coefficient
    c2 = 1.5  # social coefficient
    
    # Initialize particles
    np.random.seed(42)
    particles = np.zeros((n_particles, 14, 3))
    velocities = np.zeros((n_particles, 14, 3))
    
    # Initialize from dodecahedron subset with some randomness
    for i in range(n_particles):
        if i == 0:
            particles[i] = initial_points
        else:
            # Random perturbation of the initial configuration
            noise = np.random.randn(14, 3) * 0.3
            particles[i] = project_to_sphere(initial_points + noise)
    
    # Compute initial scores
    scores = np.array([compute_score(project_to_sphere(particles[i])) for i in range(n_particles)])
    pbest = particles.copy()
    pbest_scores = scores.copy()
    gbest_idx = np.argmax(scores)
    gbest = particles[gbest_idx].copy()
    gbest_score = scores[gbest_idx]
    
    best_final_points = initial_points
    best_final_score = best_score
    
    # PSO loop
    for it in range(n_iterations):
        if time.time() > deadline - 30:
            break
        
        # Update velocities and positions
        for i in range(n_particles):
            r1 = np.random.rand(14, 3)
            r2 = np.random.rand(14, 3)
            
            # Velocity update
            vel = (w * velocities[i] + 
                   c1 * r1 * (pbest[i] - particles[i]) + 
                   c2 * r2 * (gbest - particles[i]))
            
            # Position update
            new_pos = particles[i] + vel
            
            # Project to sphere
            new_pos = project_to_sphere(new_pos)
            
            # Compute score
            score = compute_score(new_pos)
            
            # Update personal best
            if score > pbest_scores[i]:
                pbest[i] = new_pos
                pbest_scores[i] = score
                
            # Update global best
            if score > gbest_score:
                gbest = new_pos
                gbest_score = score
            
            # Store
            particles[i] = new_pos
            velocities[i] = vel
    
    # Use the best found by PSO
    if gbest_score > best_final_score:
        best_final_points = gbest
        best_final_score = gbest_score
    
    # 4. Local refinement with Nelder-Mead from the best PSO result
    def objective_ratio(x):
        pts = x.reshape(14, 3)
        pts = project_to_sphere(pts)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0
        return -((d_min / d_max) ** 2)
    
    # Try local optimization from the best PSO result
    try:
        from scipy.optimize import minimize
        x0 = best_final_points.flatten()
        res = minimize(objective_ratio, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < objective_ratio(x0):
            opt_pts = project_to_sphere(res.x.reshape(14, 3))
            score = compute_score(opt_pts)
            if score > best_final_score:
                best_final_score = score
                best_final_points = opt_pts
                
    except Exception:
        pass
    
    return best_final_points
