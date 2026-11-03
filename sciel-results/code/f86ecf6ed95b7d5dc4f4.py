"""
Strategy: Global optimization using Differential Evolution on the true covering radius objective.
The current program uses local heuristics (SA, gradient-like forces) which often stagnate.
DE explores the space more robustly. We also optimize the covering radius calculation
to be as fast as possible to allow more DE iterations within the time limit.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_fast(pts):
    """
    Compute covering radius in radians.
    Uses convex hull facets. For each triangular facet, the circumcenter
    on the sphere is the point furthest from the 3 vertices of that facet.
    The global covering radius is the max over all such circumcenters.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: if hull fails, return a large value to discourage invalid configs
        # Or estimate via sampling if we want robustness, but DE should keep points valid.
        # Let's return a penalty.
        return 2.0

    max_radius = 0.0
    # Vectorized-ish approach for circumcenters
    # For a triangle a, b, c on unit sphere:
    # Normal n = cross(a-b, a-c)
    # Circumcenter direction u = n / ||n||
    # Distance to a is arccos(dot(u, a))
    # Note: if the triangle is obtuse, the circumcenter might be 'outside' the cap?
    # No, for a convex hull on a sphere, the facets are the boundaries of the empty caps.
    # The circumcenter of the facet is the center of the empty cap.
    # The angular distance from the circumcenter to the vertices is the radius of that cap.
    
    # Extract vertices
    idx = hull.simplices
    a = pts[idx[:, 0]]
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    
    # Handle degenerate cases (collinear points)
    valid = norms[:, 0] > 1e-12
    if not np.all(valid):
        # Filter out invalid facets
        a = a[valid]
        b = b[valid]
        c = c[valid]
        norms = norms[valid]
        if len(a) == 0:
            return 2.0
            
    u = cross / norms
    
    # Distance from u to a (one of the vertices)
    dots = np.sum(u * a, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    radii = np.arccos(dots)
    
    # The covering radius is the maximum of these facet radii.
    # Is it possible that the maximum distance is to a vertex of the hull rather than a facet center?
    # No, the farthest point from the set of points on the sphere is always the circumcenter of some facet
    # of the convex hull of the points.
    
    return np.max(radii)

def _objective(pts):
    """Objective function for optimization."""
    return _covering_radius_fast(pts)

def _random_initialization():
    """Generate a random set of points on the sphere."""
    pts = np.random.randn(N, 3)
    return _to_sphere(pts)

def _fibonacci_initialization():
    """Generate a Fibonacci spiral set of points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _to_sphere(pts)

def _local_polish(pts, max_time=5.0):
    """
    Quick local polish using small random perturbations and acceptance.
    Helps fine-tune the DE result.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.05
    temperature = 0.01
    
    while time.time() - start < max_time:
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        new_obj = _objective(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or (temperature > 1e-6 and np.random.random() < np.exp(-delta / temperature)):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            # Slightly increase step if improving
            step_size = min(step_size * 1.02, 0.1)
        else:
            step_size *= 0.98
            temperature *= 0.95
            
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses Differential Evolution for global search, followed by local polishing.
    """
    # Initialize DE population
    # DE parameters
    pop_size = 40
    F = 0.5  # Differential weight
    CR = 0.9  # Crossover rate
    
    # Initial population: mix of random and Fibonacci to ensure diversity
    pop = []
    for i in range(pop_size):
        if i == 0:
            pop.append(_fibonacci_initialization())
        elif i == 1:
            # Antipodal Fibonacci? Or just random
            pop.append(_random_initialization())
        else:
            # Perturbed Fibonacci
            base = _fibonacci_initialization()
            noise = np.random.randn(N, 3) * 0.1
            pop.append(_to_sphere(base + noise))
    
    pop = np.array(pop)
    
    # Evaluate initial population
    objs = np.array([_objective(p) for p in pop])
    best_idx = np.argmin(objs)
    best_pts = pop[best_idx].copy()
    best_obj = objs[best_idx]
    
    start_de = time.time()
    max_de_time = _DEADLINE - 20.0  # Reserve 20s for polishing and safety
    
    generation = 0
    no_improve_count = 0
    
    while time.time() - start_de < max_de_time:
        generation += 1
        
        # Check for stagnation
        if no_improve_count > 10:
            # Restart with new random population to escape local minima
            # Keep the best found so far
            pop = []
            for i in range(pop_size):
                if i == 0:
                    pop.append(best_pts + np.random.randn(N, 3) * 0.1)
                    pop[-1] = _to_sphere(pop[-1])
                else:
                    pop.append(_random_initialization())
            pop = np.array(pop)
            objs = np.array([_objective(p) for p in pop])
            best_idx = np.argmin(objs)
            if objs[best_idx] < best_obj:
                best_obj = objs[best_idx]
                best_pts = pop[best_idx].copy()
            no_improve_count = 0
            continue

        # Differential Evolution step
        new_pop = pop.copy()
        new_objs = objs.copy()
        
        for i in range(pop_size):
            # Select 3 distinct random indices
            j, k, l = np.random.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[j] + F * (pop[k] - pop[l])
            mutant = _to_sphere(mutant)
            
            # Crossover
            trial = pop[i].copy()
            for d in range(3):
                if np.random.random() < CR or (np.random.randint(N) == i): # Ensure at least one gene from mutant
                    trial[:, d] = mutant[:, d]
            trial = _to_sphere(trial)
            
            # Evaluate
            trial_obj = _objective(trial)
            
            # Selection
            if trial_obj <= new_objs[i]:
                new_pop[i] = trial
                new_objs[i] = trial_obj
                
        pop = new_pop
        objs = new_objs
        
        # Update global best
        gen_best_idx = np.argmin(objs)
        if objs[gen_best_idx] < best_obj:
            best_obj = objs[gen_best_idx]
            best_pts = pop[gen_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Final Polish
    remaining = _DEADLINE - time.time()
    if remaining > 2:
        polished_pts, polished_obj = _local_polish(best_pts, max_time=min(remaining - 2, 15))
        if polished_obj < best_obj:
            best_pts = polished_pts
            best_obj = polished_obj
            
    # Ensure valid output
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points (should be rare with DE + sphere projection)
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.9999:
        # Nudge coincident points
        for i in range(N):
            for j in range(i+1, N):
                if abs(dots[i,j]) > 0.9999:
                    best_pts[j] += np.random.randn(3) * 0.01
        best_pts = _to_sphere(best_pts)
        
    return best_pts
