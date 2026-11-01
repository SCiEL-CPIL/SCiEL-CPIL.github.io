"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_covering_radius(points):
    """
    Compute the exact covering radius in radians.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Fallback for degenerate cases

    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        denom = 2.0 * np.dot(cross, cross)
        
        if denom < 1e-12:
            continue
            
        a11 = np.dot(v1, v1)
        a12 = np.dot(v1, v2)
        a22 = np.dot(v2, v2)
        b1 = 0.5 * a11
        b2 = 0.5 * a22
        
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            continue
            
        t1 = (b1 * a22 - b2 * a12) / det
        t2 = (a11 * b2 - a12 * b1) / det
        
        O = p1 + t1 * v1 + t2 * v2
        norm_O = np.linalg.norm(O)
        if norm_O < 1e-10:
            continue
        C = O / norm_O
        
        cos_theta = np.clip(np.dot(C, p1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_radius:
            max_radius = theta
            
    return max_radius

def _pso_optimize(time_limit=110.0):
    """
    Particle Swarm Optimization to minimize covering radius.
    """
    rng = np.random.RandomState(42)
    
    # Swarm parameters
    P = 15 # Number of particles
    w = 0.9 # Inertia weight
    c1 = 1.5 # Cognitive coefficient
    c2 = 1.5 # Social coefficient
    
    # Initialize particles
    # Each particle is an array of shape (N, 3)
    particles = []
    velocities = [] # Shape (N, 3)
    
    # Initial configurations: mix of Fibonacci and Random
    for i in range(P):
        if i == 0:
            # Fibonacci sphere
            idx = np.arange(N) + 0.5
            z = 1.0 - 2.0 * idx / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
            r = np.sqrt(1.0 - z * z)
            p = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random points
            p = rng.randn(N, 3)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        # Add some noise to diversify initial swarm slightly if needed
        # But random is already diverse.
        
        particles.append(p)
        v = np.zeros((N, 3))
        velocities.append(v)
        
    # Initialize bests
    personal_bests = []
    personal_best_fitness = []
    global_best = None
    global_best_fitness = np.pi
    
    for i in range(P):
        fitness = _compute_covering_radius(particles[i])
        personal_bests.append(particles[i].copy())
        personal_best_fitness.append(fitness)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best = particles[i].copy()
            
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Update particles
        for i in range(P):
            # Random factors
            r1 = rng.rand(N, 1)
            r2 = rng.rand(N, 1)
            
            # Cognitive term: move towards personal best
            cognitive = c1 * r1 * (personal_bests[i] - particles[i])
            # Social term: move towards global best
            social = c2 * r2 * (global_best - particles[i])
            
            # Update velocity
            velocities[i] = w * velocities[i] + cognitive + social
            
            # Update position
            new_positions = particles[i] + velocities[i]
            
            # Normalize to sphere
            norms = np.linalg.norm(new_positions, axis=1, keepdims=True)
            # Handle zero norms (unlikely but safe)
            norms[norms == 0] = 1.0
            new_positions = new_positions / norms
            
            # Evaluate new position
            fitness = _compute_covering_radius(new_positions)
            
            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_bests[i] = new_positions.copy()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = new_positions.copy()
                    
            particles[i] = new_positions
            
        # Optional: Decay velocity or parameters to refine search
        # w can be decreased slightly over time
        # w = max(0.5, w * 0.999)
        
    return global_best, global_best_fitness

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points, best_radius = _pso_optimize(time_limit)
    
    # Final sanity check and normalization
    if best_points is None:
        # Fallback to Fibonacci if PSO failed completely
        idx = np.arange(N) + 0.5
        z = 1.0 - 2.0 * idx / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * idx
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
