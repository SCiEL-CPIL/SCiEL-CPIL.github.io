"""Improved packing of 50 equal spheres in a unit cube using Repulsive Dynamics + L-BFGS-B."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score using KDTree for speed."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_score(centers, beta=50.0):
    """Differentiable soft-minimum score for gradient-based optimization."""
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    a = min_pair / 2.0
    b = min_wall
    # Soft min: -1/beta * log(exp(-beta*a) + exp(-beta*b))
    m = min(a, b)
    return m - math.log(math.exp(-beta * (a - m)) + math.exp(-beta * (b - m))) / beta

def _potential_energy(x, power=4.0):
    """Sum of repulsive potentials for spreading spheres."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    
    energy = 0.0
    # Pairwise repulsion
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 1e-6:
                d = 1e-6
            energy += 1.0 / (d ** power)
            
    # Wall repulsion
    for i in range(N):
        for axis in range(3):
            d_pos = 0.5 - centers[i, axis]
            d_neg = 0.5 + centers[i, axis]
            if d_pos < 1e-6: d_pos = 1e-6
            if d_neg < 1e-6: d_neg = 1e-6
            energy += 1.0 / (d_pos ** power) + 1.0 / (d_neg ** power)
            
    return energy

def _potential_grad(x, power=4.0):
    """Gradient of the potential energy."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    grad = np.zeros_like(x)
    
    # Pairwise repulsion
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < 1e-6:
                continue
            # d/dx_i (1/d^p) = -p/d^p * (x_i - x_j)/d = -p/d^(p+1) * (x_i - x_j)
            # Wait, gradient of 1/|x| is -x/|x|^3.
            # Gradient of 1/|x|^p is -p * x / |x|^(p+2)
            factor = -power / (d ** (power + 2))
            grad[i * 3 : (i + 1) * 3] += factor * diff
            grad[j * 3 : (j + 1) * 3] -= factor * diff
            
    # Wall repulsion
    for i in range(N):
        for axis in range(3):
            idx = i * 3 + axis
            d_pos = 0.5 - centers[i, axis]
            d_neg = 0.5 + centers[i, axis]
            
            if d_pos < 1e-6: d_pos = 1e-6
            if d_neg < 1e-6: d_neg = 1e-6
            
            # d/dx (1/(0.5-x)^p) = -p * (0.5-x)^(-p-1) * (-1) = p / (0.5-x)^(p+1)
            # d/dx (1/(0.5+x)^p) = -p * (0.5+x)^(-p-1) * (1) = -p / (0.5+x)^(p+1)
            
            grad[idx] += power / (d_pos ** (power + 1))
            grad[idx] -= power / (d_neg ** (power + 1))
            
    return grad

def _initial_grid_444():
    """4x4x4 grid, first 50 points."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_potential(centers, time_limit=10.0):
    """Minimize potential energy to spread spheres."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    result = minimize(
        _potential_energy, x0,
        jac=_potential_grad,
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > time_limit:
         return centers
         
    return np.clip(result.x.reshape(N, 3), -0.5, 0.5)

def _objective_soft_min(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    return -_soft_min_score(centers, beta=beta)

def _optimize_soft_min(centers, time_limit=10.0, beta=50.0):
    """Refine using L-BFGS-B with soft-minimum objective."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Coarse pass
    result1 = minimize(
        _objective_soft_min, x0, args=(beta,),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    x0 = result1.x
    
    # Fine pass if time allows
    if time.time() - start_time < time_limit * 0.5:
         result2 = minimize(
             _objective_soft_min, x0, args=(beta * 2,),
             method='L-BFGS-B', bounds=bounds,
             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
         )
         x0 = result2.x
         
    return np.clip(x0.reshape(N, 3), -0.5, 0.5)

def _local_search(centers, time_limit=5.0):
    """
    Local search: iteratively move each sphere to maximize its local clearance.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            # Get neighbors and wall distances
            # We want to move c_i to maximize min(dist to others, dist to walls)
            # Simple strategy: try moving in the direction away from the closest constraint
            
            # Find closest other sphere
            tree = cKDTree(current)
            dists, idxs = tree.query(current[i], k=2)
            # dists[0] is 0, dists[1] is closest neighbor
            min_d = dists[1]
            neighbor_idx = idxs[1]
            
            # Find closest wall
            wall_dists = []
            wall_dirs = []
            for axis in range(3):
                d_pos = 0.5 - current[i, axis]
                d_neg = 0.5 + current[i, axis]
                wall_dists.append(d_pos)
                wall_dirs.append(-1) # Push negative
                wall_dists.append(d_neg)
                wall_dirs.append(1)  # Push positive
                
            min_wall = np.min(wall_dists)
            
            # Current local radius is min(min_d/2, min_wall)
            # If min_d/2 < min_wall, move away from neighbor
            # Else move away from wall
            
            # To be more robust, we can do a small random search or gradient step
            # Let's try a small step in the direction of the dominant constraint
            
            move = np.zeros(3)
            if min_d / 2.0 < min_wall:
                # Move away from neighbor
                diff = current[i] - current[neighbor_idx]
                d = np.linalg.norm(diff)
                if d > 1e-6:
                    move = (diff / d) * 0.01
            else:
                # Move away from closest wall
                # Find which wall
                min_wall_idx = np.argmin(wall_dists)
                axis = min_wall_idx // 2
                sign = wall_dirs[min_wall_idx]
                move[axis] = -sign * 0.01 # Move inwards
            
            # Try the move
            new_pos = current[i] + move
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            # Check if it improves the global score? No, that's expensive.
            # Check if it improves local clearance?
            # Simple check: does it increase min distance to neighbors?
            new_dists, _ = tree.query(new_pos, k=2)
            new_min_d = new_dists[1]
            
            # Also check walls
            new_wall_dists = []
            for axis in range(3):
                new_wall_dists.append(0.5 - new_pos[axis])
                new_wall_dists.append(0.5 + new_pos[axis])
            new_min_wall = np.min(new_wall_dists)
            
            if new_min_d / 2.0 > min_d / 2.0 or new_min_wall > min_wall:
                current[i] = new_pos
                improved = True
                # Update score
                # It's expensive to update global score every step, so do it periodically or at end of loop
                if i % 10 == 0:
                    score = _exact_score(current)
                    if score > best_score:
                        best_score = score
                        best = current.copy()
        
        if not improved:
            # Try random perturbations to escape local minima
            for _ in range(10):
                idx = np.random.randint(N)
                current[idx] += np.random.uniform(-0.01, 0.01, 3)
                current = np.clip(current, -0.5, 0.5)
                score = _exact_score(current)
                if score > best_score:
                    best_score = score
                    best = current.copy()
                    break # If improved, restart the loop
            
        # Final score check
        score = _exact_score(current)
        if score > best_score:
            best_score = score
            best = current.copy()
            
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initialize with 4x4x4 grid
    centers = _initial_grid_444()
    best_centers = centers.copy()
    best_score = _exact_score(best_centers)
    
    # 2. Optimize Potential Energy to spread out
    if time.time() < deadline - 30.0:
        spread_centers = _optimize_potential(centers, time_limit=15.0)
        spread_score = _exact_score(spread_centers)
        if spread_score > best_score:
            best_score = spread_score
            best_centers = spread_centers
    
    # 3. Optimize Soft-Min to maximize minimum distance
    if time.time() < deadline - 20.0:
        refined = _optimize_soft_min(best_centers, time_limit=15.0, beta=50.0)
        ref_score = _exact_score(refined)
        if ref_score > best_score:
            best_score = ref_score
            best_centers = refined
    
    # 4. Local Search
    if time.time() < deadline - 10.0:
        ls_centers, ls_score = _local_search(best_centers, time_limit=15.0)
        if ls_score > best_score:
            best_score = ls_score
            best_centers = ls_centers
    
    # 5. Final Polish with L-BFGS-B
    if time.time() < deadline - 5.0:
        final = _optimize_soft_min(best_centers, time_limit=5.0, beta=100.0)
        final_score = _exact_score(final)
        if final_score > best_score:
            best_score = final_score
            best_centers = final
    
    return best_centers
# EVOLVE-BLOCK-END
