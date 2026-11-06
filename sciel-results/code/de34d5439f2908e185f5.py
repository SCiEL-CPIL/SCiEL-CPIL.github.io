"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_icosahedron(n, scale=0.9):
    """
    Generates an icosahedral cluster.
    For n <= 13, returns a truncated icosahedron (center + 12 vertices).
    For n > 13, adds atoms on a second shell (approximating Mackay).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale such that nearest neighbor distance is approx 1.8
    # In unit icosahedron (radius 1), min dist is 2.
    # We want min dist ~ 1.8, so scale radius by 1.8/2 = 0.9
    verts = verts * 0.9 * scale
    
    points = [np.zeros(3)]
    if n > 1:
        points.extend([v.copy() for v in verts[:min(n-1, 12)]])
        
    if n > 13:
        # Add atoms on second shell
        # Mackay 55 has 42 in second shell.
        # Approximate radius for second shell in Mackay is ~ 1.5 * R1
        R2 = 0.9 * scale * 1.5
        num_to_add = n - len(points)
        
        # Use a Fibonacci sphere for better distribution
        # Or just random with rejection
        for _ in range(num_to_add):
            # Random point on sphere
            z = np.random.uniform(-1, 1)
            theta = np.random.uniform(0, 2 * np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * R2
            
            # Check min distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                # Try a few more times with slightly perturbed position
                for _attempt in range(5):
                    p_pert = p + np.random.normal(0, 0.1, 3)
                    valid = True
                    for q in points:
                        if np.linalg.norm(p_pert - q) < 0.5:
                            valid = False
                            break
                    if valid:
                        points.append(p_pert)
                        break
                else:
                    # Just add it, relax will fix overlaps if any (though relax might fail if too close)
                    points.append(p)
                
    return np.array(points)

def _generate_bcc(n):
    """
    Generates a BCC-like cluster.
    Center + 8 cube vertices + 6 face centers + ...
    """
    points = [np.zeros(3)]
    # Cube vertices
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append(np.array([i, j, k]) * 0.9)
    # Face centers
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                # Skip corners, add face centers
                pass 
    # Simpler: just add random points in a cube
    while len(points) < n:
        p = np.random.uniform(-1.5, 1.5, 3)
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            points.append(p)
        else:
            # Try again
            continue
    return np.array(points)

def _generate_random_cluster(n):
    """Generate a random cluster with minimum distance 0.5."""
    points = []
    for _ in range(n):
        for _attempt in range(100):
            p = np.random.uniform(-2, 2, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
        else:
            # Fallback: place it far away
            points.append(np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(-5, 5)]))
    return np.array(points)

def _basin_hopping_worker(start_pos, potential, parameter, time_limit, best_energy_holder, lock):
    """
    Worker for Basin Hopping.
    """
    pos = start_pos.copy()
    try:
        e, pos_relaxed = relax(pos, potential, parameter)
    except Exception:
        return 1e10
        
    current_energy = e
    T = 5.0
    cooling_rate = 0.98
    step_size = 0.5
    
    end_time = time.time() + time_limit
    
    while time.time() < end_time and _check_time():
        # Perturb
        # Move all atoms slightly, or a subset?
        # Moving all atoms with small Gaussian noise is standard BH.
        new_pos = pos + np.random.normal(0, step_size, (len(pos), 3))
        
        # Relax
        try:
            new_e, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
            
        delta_e = new_e - current_energy
        # Metropolis criterion
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            pos = new_pos_relaxed
            current_energy = new_e
            
            # Update global best
            with lock:
                if current_energy < best_energy_holder[0]:
                    best_energy_holder[0] = current_energy
                    best_energy_holder[1] = pos.copy()
                    
            # Adaptive step size: reduce if improving
            step_size *= 0.995
            if step_size < 0.1:
                step_size = 0.1
                
        T *= cooling_rate
        if T < 0.5:
            T = 5.0 # Reheat

    return current_energy

def construct(instance):
    """
    Strategy:
    1. Generate diverse initial guesses (Icosahedral, BCC, Random).
    2. Relax them.
    3. Run parallel Basin Hopping from these starting points.
    4. Return the best structure found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # 1. Generate Initial Guesses
    init_pos_ico = _generate_icosahedron(size, 1.0)
    init_pos_bcc = _generate_bcc(size)
    init_pos_rand = _generate_random_cluster(size)
    
    # Relax each
    start_points = []
    best_energy = 1e10
    best_positions = init_pos_ico.copy()
    
    for init in [init_pos_ico, init_pos_bcc, init_pos_rand]:
        try:
            e, pos_relaxed = relax(init, potential, parameter)
            start_points.append(pos_relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = pos_relaxed.copy()
        except Exception:
            continue
            
    # If all relaxations failed, fallback
    if not start_points:
        start_points = [init_pos_ico]
        best_energy = 1e10
        best_positions = init_pos_ico.copy()
        
    # 2. Parallel Basin Hopping
    # We have ~290s.
    # Use 4 threads.
    num_threads = 4
    time_per_thread = 270.0 / num_threads  # Leave buffer
    
    # Create multiple starting points for the threads
    # We have 3 good starts. Let's create 4 by perturbing the best one.
    while len(start_points) < num_threads:
        perturbed = best_positions + np.random.normal(0, 0.3, (size, 3))
        try:
            e, pos_relaxed = relax(perturbed, potential, parameter)
            start_points.append(pos_relaxed)
        except Exception:
            start_points.append(perturbed)
            
    best_energy_holder = [best_energy, best_positions]
    lock = threading.Lock()
    
    # Launch workers
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for sp in start_points:
            if not _check_time():
                break
            futures.append(executor.submit(_basin_hopping_worker, sp, potential, parameter, time_per_thread, best_energy_holder, lock))
            
        # Wait for completion or timeout
        try:
            for future in as_completed(futures, timeout=280):
                future.result()
        except Exception:
            pass
            
    # 3. Final Relaxation of the best found
    try:
        e_final, pos_final = relax(best_energy_holder[1], potential, parameter)
        return pos_final
    except Exception:
        return best_energy_holder[1]

# EVOLVE-BLOCK-END
