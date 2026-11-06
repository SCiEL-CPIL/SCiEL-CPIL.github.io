"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed for n circles."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius, kind="stable")[:n]
            return points[idx]
        spacing *= 0.98

def _compute_score(points):
    """Compute the packing score: min of pairwise dist/2 and boundary dist."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Boundary distances
    radii = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair, min_boundary)

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin
    
    # 1. Generate initial seed
    centers = _hexagonal_seed(N).copy()
    best_centers = centers.copy()
    best_r = _compute_score(centers)
    
    # Precompute for fast evaluation
    def _evaluate_fast(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        radii = np.sqrt(np.sum(c ** 2, axis=1))
        min_boundary = np.min(1.0 - radii)
        return min(min_pair, min_boundary)

    # 2. Local Optimization Phase (Gradient Descent / Nelder-Mead hybrid)
    # We maximize r, so we minimize -r.
    # However, simple gradient descent on min distance is non-smooth.
    # We use a "push" method: if circles overlap, push them apart. If they hit boundary, push in.
    
    def _push_step(c, step=0.1):
        c = c.copy()
        for _ in range(10): # Inner iterations for stability
            # Check overlaps
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Find minimum distance pair
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = min_idx
            d = dists[i, j]
            
            if d < 2 * _compute_score(c) * 0.99: # If overlapping or very close
                # Push apart
                vec = c[j] - c[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec = vec / norm
                    c[i] -= vec * step * 0.5
                    c[j] += vec * step * 0.5
            
            # Check boundary
            radii = np.sqrt(np.sum(c ** 2, axis=1))
            # If any circle is outside or too close to boundary, pull in
            for k in range(N):
                r_k = radii[k]
                if r_k > 1.0 - _compute_score(c) * 0.99:
                    vec = -c[k] / (r_k + 1e-10)
                    c[k] += vec * step * 0.5
            
            # Ensure inside container
            radii = np.sqrt(np.sum(c ** 2, axis=1))
            mask = radii > 1.0
            if np.any(mask):
                c[mask] = c[mask] / radii[mask, np.newaxis] * 0.99
        
        return c

    # 3. Simulated Annealing / Random Perturbation Loop
    current_c = centers.copy()
    current_r = _compute_score(current_c)
    
    # Temperature for annealing
    T = 0.05
    T_min = 0.0001
    cooling_rate = 0.9995
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < timeout:
        # Check time budget
        if time.time() - start_time > timeout - 5.0:
            break
            
        # Pick a random circle to move
        i = np.random.randint(N)
        
        # Generate a random perturbation
        # Step size scales with temperature
        delta = np.random.normal(0, T, size=2)
        new_c = current_c.copy()
        new_c[i] += delta
        
        # Ensure the moved circle is inside the container
        r_new = np.linalg.norm(new_c[i])
        if r_new > 1.0:
            new_c[i] = new_c[i] / r_new * 0.99
        
        new_r = _evaluate_fast(new_c)
        
        # Acceptance criterion (Metropolis)
        if new_r > current_r:
            current_c = new_c
            current_r = new_r
            no_improve_count = 0
        else:
            # Accept worse moves with probability exp((new_r - current_r) / T)
            delta_r = new_r - current_r
            if random.random() < np.exp(delta_r / T):
                current_c = new_c
                current_r = new_r
            no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # If stagnating, do a local push to escape local minima
        if no_improve_count > 100:
            current_c = _push_step(current_c, step=0.05)
            current_r = _evaluate_fast(current_c)
            no_improve_count = 0
            T = max(T * 1.1, 0.01) # Slight reheat
            
        # Update best
        if current_r > best_r:
            best_r = current_r
            best_centers = current_c.copy()
            
    # 4. Final Refinement: Use scipy minimize on the best configuration
    # We try to maximize the minimum distance.
    # This is a non-smooth problem, but we can use Nelder-Mead on a smoothed version
    # or simply perform a fine-tuning search around the best point.
    
    # Let's do a fine-tuning random search around the best point
    final_c = best_centers.copy()
    final_r = best_r
    fine_T = 0.005
    fine_steps = 5000
    
    steps_done = 0
    while time.time() - start_time < timeout - 1.0 and steps_done < fine_steps:
        i = np.random.randint(N)
        delta = np.random.normal(0, fine_T, size=2)
        new_c = final_c.copy()
        new_c[i] += delta
        
        r_new = np.linalg.norm(new_c[i])
        if r_new > 1.0:
            new_c[i] = new_c[i] / r_new * 0.99
            
        new_r = _evaluate_fast(new_c)
        
        if new_r > final_r:
            final_c = new_c
            final_r = new_r
            # Shrink step size slightly on success
            fine_T *= 0.99
        else:
            # Expand step size slightly on failure? Or keep?
            # Keep simple: constant fine_T or very slow decay
            pass
        steps_done += 1
        
    if final_r > best_r:
        best_centers = final_c
        best_r = final_r
        
    return best_centers

# EVOLVE-BLOCK-END
