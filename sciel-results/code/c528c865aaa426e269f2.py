"""New program using Basin Hopping with structured initialization for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _init_fcc_sphere(n):
    """Initialize atoms in a truncated fcc sphere."""
    # Generate fcc points in a box
    a = 1.0
    points = []
    # Search radius to ensure enough points
    R = 5.0 
    for i in range(-int(R), int(R)+1):
        for j in range(-int(R), int(R)+1):
            for k in range(-int(R), int(R)+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i*a + dx*a, j*a + dy*a, k*a + dz*a])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return np.random.default_rng(0).uniform(-1, 1, (n, 3))
    
    # Sort by distance to center and take n closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    pts = points[idx].copy()
    
    # Center of mass to origin
    pts -= np.mean(pts, axis=0)
    return pts

def _init_random_compact(n, seed):
    """Initialize atoms in a random compact sphere."""
    rng = np.random.default_rng(seed)
    # Estimate radius for close packing
    # Volume ~ N * (4/3 pi r_eff^3). r_eff ~ 0.5 (half min dist)
    # R ~ (3N/4pi)^(1/3) * 1.2 (packing factor)
    R = (3*n / (4*np.pi))**(1/3) * 1.5
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 10000:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if len(pts) > 0:
                dists = np.linalg.norm(np.array(pts) - p, axis=1)
                if np.min(dists) >= 0.7: # Slightly loose to allow relaxation
                    pts.append(p)
            else:
                pts.append(p)
        attempts += 1
        
    # If not enough, fill with random points (relax will fix overlaps)
    while len(pts) < n:
        p = rng.uniform(-R, R, 3)
        pts.append(p)
        
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    return pts

def _basin_hopping(initial_pos, potential, rho, n, time_limit=40.0):
    """
    Perform Basin Hopping on the given initial configuration.
    Returns the best positions and energy found.
    """
    start_time = time.time()
    curr_pos = initial_pos.copy()
    curr_energy, _ = relax(curr_pos, potential, rho)
    best_pos = curr_pos.copy()
    best_energy = curr_energy
    
    # Temperature schedule
    # Start high enough to explore, end low to converge
    # Typical energies for these clusters are O(N). 
    # Let's estimate scale from initial energy.
    T = max(1.0, abs(curr_energy) / 50.0)
    T_min = 0.01
    cooling = 0.95
    
    step_size = 0.5
    
    while time.time() - start_time < time_limit:
        # Perturb
        noise = np.random.default_rng().normal(0, step_size, (n, 3))
        new_pos = curr_pos + noise
        
        # Relax
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, rho)
        except:
            continue
            
        if np.isnan(new_energy):
            continue

        # Metropolis Accept
        if new_energy < curr_energy:
            accept = True
        else:
            delta_e = new_energy - curr_energy
            if T > T_min:
                prob = np.exp(-delta_e / T)
                accept = np.random.default_rng().random() < prob
            else:
                accept = False

        if accept:
            curr_pos = new_pos_relaxed
            curr_energy = new_energy
            if curr_energy < best_energy:
                best_energy = curr_energy
                best_pos = curr_pos.copy()
        else:
            # If rejected, we still stay at curr_pos, but we might want to 
            # occasionally restart from best_pos if stuck? 
            # Standard BH stays.
            pass
            
        # Cool down
        T *= cooling
        # Adapt step size slightly? Keep constant for simplicity, 
        # or decrease it as T decreases.
        step_size *= 0.99

    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration using Basin Hopping."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget per instance: 300s / 6 instances = 50s.
    # Reserve 5s for overhead/initialization. 45s for search.
    # We will run 3 independent basin hopping runs with different inits.
    # Each gets ~15s.
    
    time_per_run = 14.0
    
    candidates = []
    
    # 1. FCC Sphere Init
    try:
        init1 = _init_fcc_sphere(n)
        pos1, e1 = _basin_hopping(init1, potential, rho, n, time_limit=time_per_run)
        candidates.append((e1, pos1))
    except Exception as e:
        pass
        
    # 2. Random Compact Init
    try:
        init2 = _init_random_compact(n, seed=1234)
        pos2, e2 = _basin_hopping(init2, potential, rho, n, time_limit=time_per_run)
        candidates.append((e2, pos2))
    except Exception as e:
        pass
        
    # 3. Another Random Compact Init (different seed)
    try:
        init3 = _init_random_compact(n, seed=5678)
        pos3, e3 = _basin_hopping(init3, potential, rho, n, time_limit=time_per_run)
        candidates.append((e3, pos3))
    except Exception as e:
        pass
        
    if not candidates:
        # Fallback: Return a simple valid config if all failed
        rng = np.random.default_rng(0)
        R = (3*n / (4*np.pi))**(1/3) * 1.5
        pts = []
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        # Try one last relax
        try:
            _, pts = relax(pts, potential, rho)
        except:
            pass
        return pts

    # Select the best candidate
    best_energy, best_pos = min(candidates, key=lambda x: x[0])
    
    return best_pos
# EVOLVE-BLOCK-END
