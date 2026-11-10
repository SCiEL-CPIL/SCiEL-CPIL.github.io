"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial import cKDTree

def _radius_of_placement(centers):
    """Compute the largest common radius for a given set of centers."""
    # Distance from each center to the nearest wall
    dist_to_wall = np.min([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=0)
    min_wall = np.min(dist_to_wall)
    
    # Minimum pairwise distance / 2 using KDTree for efficiency
    if len(centers) <= 1:
        min_pair = np.inf
    else:
        tree = cKDTree(centers)
        # Query nearest neighbors (excluding self)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1]) / 2.0
    
    return min(min_wall, min_pair)


def _hex_lattice_init():
    """Create hexagonal lattice initialization for better starting packing."""
    # For N=200, use a hexagonal pattern
    # Hex lattice: rows offset by half spacing
    # Try different densities
    best_centers = None
    best_r = 0
    
    for rows in range(10, 25):
        cols = int(np.ceil(N / rows))
        # Hexagonal spacing
        dx = 1.0 / (cols + 0.5)  # horizontal spacing
        dy = dx * np.sqrt(3) / 2  # vertical spacing for hex
        
        centers = []
        for row in range(rows):
            for col in range(cols):
                if len(centers) >= N:
                    break
                x = (col + 0.5) * dx - 0.5 + (0.5 * dx if row % 2 == 1 else 0)
                y = (row + 0.5) * dy - 0.5
                centers.append([x, y])
        
        if len(centers) < N:
            continue
        
        centers = np.array(centers[:N])
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers


def _repulsive_optimization(centers, max_iters=2000, time_limit=30):
    """
    Optimize using repulsive forces between circles.
    Each circle is pushed away from its nearest neighbors.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    step_size = 0.002
    no_improve = 0
    
    for it in range(max_iters):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Find nearest neighbor for each point
        tree = cKDTree(current)
        dists, neighbors = tree.query(current, k=2)
        nearest_dists = dists[:, 1]
        nearest_neighbors = neighbors[:, 1]
        
        # Compute repulsive forces
        forces = np.zeros_like(current)
        for i in range(len(current)):
            j = nearest_neighbors[i]
            if i == j:
                continue
            # Vector from i to j
            diff = current[j] - current[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Random push if coincident
                forces[i] += np.random.randn(2) * 0.01
            else:
                # Repulsive force (stronger when closer)
                force_mag = step_size * (1.0 / max(dist, 0.001))
                forces[i] += (diff / dist) * force_mag
        
        # Also add wall repulsion
        for i in range(len(current)):
            for coord in range(2):
                val = current[i, coord]
                if val < -0.4:
                    forces[i, coord] += 0.01 * (-0.4 - val)
                elif val > 0.4:
                    forces[i, coord] += 0.01 * (0.4 - val)
        
        # Update positions
        candidate = current + forces
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve % 50 == 0:
                step_size *= 0.9
                if step_size < 1e-6:
                    break
    
    return best


def _simulated_annealing(centers, max_iters=5000, time_limit=40):
    """
    Simulated annealing to escape local minima.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    temp = 0.01
    cooling = 0.995
    no_improve = 0
    
    for it in range(max_iters):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        idx = np.random.randint(len(current))
        perturbation = np.random.randn(2) * temp
        candidate = current.copy()
        candidate[idx] += perturbation
        candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        delta = candidate_r - current_r
        
        # Accept if better, or with probability exp(delta/temp) if worse
        if delta > 0 or np.random.random() < np.exp(delta / max(temp, 1e-10)):
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        temp *= cooling
        if temp < 1e-6:
            break
    
    return best


def _refine_with_kdtree(centers, max_iters=3000, time_limit=25):
    """
    Fine-tuning using KDTree to identify and fix close pairs.
    """
    current = centers.copy()
    current_r = _radius_of_placement(current)
    best = current.copy()
    best_r = current_r
    
    start_time = time.time()
    step_size = 0.001
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Find all pairs closer than 2*r
        tree = cKDTree(current)
        pairs = tree.query_pairs(2 * current_r * 1.05)
        
        if len(pairs) == 0:
            # No close pairs, try to increase radius by moving points apart
            # Find the closest pair
            dists, neighbors = tree.query(current, k=2)
            min_dist = np.min(dists[:, 1])
            if min_dist > 2 * current_r * 1.1:
                # Good separation, try small random moves
                idx = np.random.randint(len(current))
                perturbation = np.random.randn(2) * step_size
                candidate = current.copy()
                candidate[idx] += perturbation
                candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
                candidate_r = _radius_of_placement(candidate)
                if candidate_r > current_r:
                    current = candidate
                    current_r = candidate_r
                    if current_r > best_r:
                        best_r = current_r
                        best = current.copy()
            continue
        
        # Fix the closest pair
        min_pair_dist = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(current[i] - current[j])
            if d < min_pair_dist:
                min_pair_dist = d
                min_pair = (i, j)
        
        if min_pair is None:
            continue
        
        i, j = min_pair
        # Move both points apart
        diff = current[i] - current[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
        else:
            direction = diff / dist
        
        move = step_size * 0.5
        candidate = current.copy()
        candidate[i] += direction * move
        candidate[j] -= direction * move
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_r = _radius_of_placement(candidate)
        
        if candidate_r > current_r:
            current = candidate
            current_r = candidate_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
        else:
            # Try smaller step
            step_size *= 0.95
            if step_size < 1e-7:
                break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    # Start with hexagonal lattice
    centers = _hex_lattice_init()
    best_centers = centers.copy()
    best_r = _radius_of_placement(centers)
    
    # Phase 1: Repulsive optimization
    if time.time() < deadline - 60:
        centers = _repulsive_optimization(centers, max_iters=2000, time_limit=25)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 35:
        centers = _simulated_annealing(centers, max_iters=5000, time_limit=20)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 3: KDTree refinement
    if time.time() < deadline - 15:
        centers = _refine_with_kdtree(centers, max_iters=3000, time_limit=15)
        r = _radius_of_placement(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 4: More restarts if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        n_restarts = min(int(remaining_time / 10), 5)
        for restart in range(n_restarts):
            if time.time() >= deadline - 5:
                break
            
            # New hex lattice with random offset
            centers = _hex_lattice_init()
            # Add small random perturbation
            centers += np.random.randn(N, 2) * 0.005
            centers = np.clip(centers, -0.5, 0.5)
            
            # Quick optimization
            centers = _repulsive_optimization(centers, max_iters=500, time_limit=5)
            r = _radius_of_placement(centers)
            
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
    
    # Final refinement
    if time.time() < deadline - 3:
        best_centers = _refine_with_kdtree(best_centers, max_iters=1000, time_limit=3)
        final_r = _radius_of_placement(best_centers)
        if final_r > best_r:
            best_r = final_r
    
    return best_centers
# EVOLVE-BLOCK-END
