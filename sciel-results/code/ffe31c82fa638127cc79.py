"""
Pack 50 equal circles in a unit square using hard-constrained local search.
Strategy:
1. Generate diverse initial configurations (grid, hex, random).
2. Use Simulated Annealing / Hill Climbing with random perturbations.
   - Objective: max(min_pair_dist/2, min_wall_dist)
   - Move: pick random pair, displace along vector between them (repulsive) or random direction.
   - Acceptance: Accept if radius increases. In SA phase, accept worse with prob exp(-delta/T).
3. Final polish: Identify active constraints (pairs at min dist, walls at min dist)
   and solve a small QP/Newton system to maximize the margin locally.
"""
import numpy as np
import time
import random
from scipy.optimize import linprog

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _radius_fast(centers):
    """Faster radius calculation for tight loops, avoids full matrix if possible? 
    For N=50, full matrix is 2500 ops, which is fine. 
    But we can optimize the loop structure."""
    wall_r = np.min(0.5 - np.abs(centers))
    min_pair = np.inf
    for i in range(N):
        ci = centers[i]
        # Distance to all j > i
        diffs = centers[i+1:] - ci
        d2 = np.sum(diffs**2, axis=1)
        min_d2 = np.min(d2)
        min_d = np.sqrt(min_d2)
        if min_d < min_pair:
            min_pair = min_d
    return min(wall_r, min_pair / 2.0)

def _generate_grid(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _generate_hex(n_x, n_y):
    """Create a hexagonal grid."""
    # Estimate spacing
    # For 50 points, roughly 7x8 or 8x7. 
    # Hex packing is denser.
    # Let's just use a grid and jitter it slightly towards hex? 
    # Or generate a proper hex lattice and crop.
    points = []
    # Approx spacing for 50 points in unit square
    # Area per point = 1/50 = 0.02. Hex area = s^2 * sqrt(3)/2.
    # s^2 * 0.866 = 0.02 -> s^2 = 0.023 -> s = 0.15
    s = 0.15
    dy = s * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * s
            y = j * dy
            if j % 2 != 0:
                x += s / 2
            points.append((x, y))
    points = np.array(points)
    # Center and scale to fit
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points *= (0.45 / max_ext)
    return points[:N]

def _polish_active_set(centers, r, time_limit=5.0):
    """
    Identify active constraints and try to move them to increase r.
    This is a simplified Newton step on the active set.
    """
    start_time = time.time()
    centers = centers.copy()
    
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Find active constraints
        # 1. Walls
        active_walls = []
        for i in range(N):
            for k in range(2):
                d = 0.5 - np.abs(centers[i, k])
                if d < r + 1e-6:
                    # Gradient pushes away from wall
                    # If x > 0, push -x. If x < 0, push +x.
                    sign = 1 if centers[i, k] > 0 else -1
                    grad = np.zeros(2)
                    grad[k] = -sign
                    active_walls.append((i, k, grad))
        
        # 2. Pairs
        active_pairs = []
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        for i in range(N):
            for j in range(i+1, N):
                if dist[i, j] < 2*r + 1e-6:
                    # Gradient for i: push away from j
                    # Gradient for j: push away from i
                    vec = centers[i] - centers[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        grad_i = vec / norm
                        grad_j = -vec / norm
                    else:
                        grad_i = np.array([1.0, 0.0])
                        grad_j = -grad_i
                    active_pairs.append((i, j, grad_i, grad_j))
        
        if not active_walls and not active_pairs:
            # No active constraints? Increase r slightly?
            # Actually if no constraints are active, we can move freely.
            # But usually there are.
            break
            
        # Build a small linear system to find a direction that increases all active distances
        # We want to find delta such that:
        # For walls: delta_i . grad_wall > 0
        # For pairs: (delta_i - delta_j) . vec_ij > 0
        
        # This is a feasibility problem. 
        # Simple heuristic: Average the gradients.
        # d = sum of gradients for each point
        d = np.zeros((N, 2))
        
        for i, k, grad in active_walls:
            d[i] += grad * 0.5 # Weight walls less? Or more?
            
        for i, j, gi, gj in active_pairs:
            d[i] += gi
            d[j] += gj
            
        # Normalize d
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            break
        d /= norm_d
        
        # Try a small step
        step = 0.01
        new_centers = centers + d * step
        new_r = _radius_fast(new_centers)
        
        if new_r > r:
            centers = new_centers
            r = new_r
        else:
            # Try smaller step
            step = 0.001
            new_centers = centers + d * step
            new_r = _radius_fast(new_centers)
            if new_r > r:
                centers = new_centers
                r = new_r
            else:
                break
                
    return centers, r

def _local_search(centers, time_limit=20.0):
    """
    Simulated Annealing / Hill Climbing local search.
    """
    start_time = time.time()
    r = _radius_fast(centers)
    centers = centers.copy()
    
    # Parameters
    T_start = 0.01
    T_end = 0.0001
    cooling = 0.995
    
    # Random seed for reproducibility within run? 
    # Using global random state.
    
    while time.time() - start_time < time_limit:
        # Pick a random pair to perturb
        i, j = random.sample(range(N), 2)
        
        # Propose move
        # Option 1: Push apart
        # Option 2: Random kick
        if random.random() < 0.5:
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                # Push apart
                step_size = random.uniform(0.001, 0.01)
                centers[i] += vec / norm * step_size
                centers[j] -= vec / norm * step_size
            else:
                # Coincident, random kick
                centers[i] += random.uniform(-0.01, 0.01)
                centers[j] += random.uniform(-0.01, 0.01)
        else:
            # Random kick for i
            centers[i] += random.uniform(-0.005, 0.005, size=2)
            
        # Clamp to box
        centers = np.clip(centers, -0.49, 0.49)
        
        new_r = _radius_fast(centers)
        
        if new_r >= r:
            r = new_r
            # Reset temperature? Or just continue.
            # If we find a big improvement, we might want to cool down faster
            T = max(T_end, T * cooling)
        else:
            # Accept with prob exp((new_r - r) / T)
            delta = new_r - r
            if random.random() < np.exp(delta / T):
                r = new_r
            else:
                # Reject
                # Undo move
                # We didn't save old state, so we need to revert
                # This is inefficient. Better to save old state.
                pass # This logic is flawed without state saving.
        
        # Correct implementation with state saving
        # Let's restart the loop logic properly.
        pass

    return centers, r

def _local_search_efficient(centers, time_limit=20.0):
    """
    Efficient Hill Climbing with state saving.
    """
    start_time = time.time()
    r = _radius_fast(centers)
    centers = centers.copy()
    
    T = 0.005
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        i, j = random.sample(range(N), 2)
        
        old_i = centers[i].copy()
        old_j = centers[j].copy()
        
        # Propose move
        if random.random() < 0.7:
            # Push apart
            vec = old_i - old_j
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                step_size = random.uniform(0.001, 0.005)
                centers[i] = old_i + (vec / norm) * step_size
                centers[j] = old_j - (vec / norm) * step_size
            else:
                centers[i] = old_i + random.uniform(-0.005, 0.005, size=2)
                centers[j] = old_j + random.uniform(-0.005, 0.005, size=2)
        else:
            # Random kick
            centers[i] = old_i + random.uniform(-0.003, 0.003, size=2)
            
        # Clamp
        centers[i] = np.clip(centers[i], -0.49, 0.49)
        centers[j] = np.clip(centers[j], -0.49, 0.49)
        
        new_r = _radius_fast(centers)
        
        if new_r >= r:
            r = new_r
            T = max(0.0001, T * cooling)
        else:
            # Accept worse move?
            # In packing, usually we want to be greedy or very cautious with SA.
            # Let's try a simple greedy with occasional random restarts of temperature.
            if random.random() < np.exp((new_r - r) / T):
                r = new_r
            else:
                # Reject
                centers[i] = old_i
                centers[j] = old_j
            T = max(0.0001, T * cooling)
            
        # Occasionally cool down faster if stuck?
        # For now, simple cooling.
        
    return centers, r

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Initial configurations
    inits = []
    
    # 1. Grid 7x8
    inits.append(_generate_grid(7, 8))
    
    # 2. Grid 8x7
    inits.append(_generate_grid(8, 7))
    
    # 3. Hex
    inits.append(_generate_hex(8, 8))
    
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 5. Jittered Grid
    np.random.seed(43)
    inits.append(_generate_grid(7, 8) + np.random.uniform(-0.01, 0.01, size=(N, 2)))
    
    # Allocate time to each init
    time_per_init = (DEADLINE - 30) / len(inits) # Reserve 30s for final polish and checks
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_left = DEADLINE - 20 - (time.time() - start_time)
        alloc_time = min(time_per_init, time_left / (len(inits) - i))
        
        # Run local search
        refined, r = _local_search_efficient(init, time_limit=alloc_time * 0.8)
        
        # Quick polish
        if time.time() - start_time < DEADLINE - 10:
            refined, r_polish = _polish_active_set(refined, r, time_limit=alloc_time * 0.2)
            if r_polish > r:
                r = r_polish
                refined = refined.copy()
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # Final global polish on the best
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        # Run a longer local search on the best
        refined, r = _local_search_efficient(best_centers, time_limit=time_left * 0.5)
        refined, r_polish = _polish_active_set(refined, r, time_limit=time_left * 0.5)
        
        if r_polish > best_r:
            best_r = r_polish
            best_centers = refined.copy()
            
    # Ensure valid return
    if best_centers is None:
        # Fallback
        best_centers = _generate_grid(7, 8)
        
    return best_centers

if __name__ == "__main__":
    # Quick test
    c = construct()
    print(f"Radius: {_radius(c)}")
