"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for time management
TIME_LIMIT = 300.0
SAFETY_MARGIN = 10.0
NUM_THREADS = 4

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _compute_energy(pos):
    """Compute the Dzugutov potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    # pos shape (n, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Mask for pairs within cutoff b
    # We only care about r < b
    # The potential is defined piecewise.
    # V(r) = A(r^-m - B)exp(c/(r - a)) for r < a
    # V(r) = B exp(d/(r - b)) for a <= r < b
    # V(r) = 0 for r >= b
    
    # Create masks
    mask_a = dist < a
    mask_b = (dist >= a) & (dist < b)
    
    energy = 0.0
    
    # Calculate for r < a
    if np.any(mask_a):
        # Get the unique pairs to avoid double counting
        # Since dist is symmetric, we can just sum upper triangle
        # But numpy boolean indexing on a 2D array gives all True indices
        # We need to be careful not to double count.
        # Let's just iterate or use a smarter vectorized approach.
        # For n up to 60, n^2 is 3600, which is small.
        # But let's stick to vectorized where possible.
        
        # Get indices
        rows, cols = np.where(mask_a)
        # Only take upper triangle (i < j)
        mask_upper = rows < cols
        rows_u = rows[mask_upper]
        cols_u = cols[mask_upper]
        
        r_vals = dist[rows_u, cols_u]
        
        # Potential term
        # A * (r^-m - B) * exp(c / (r - a))
        # Handle r-a being close to 0? The potential is singular at r=a? 
        # Actually, the Dzugutov potential is continuous. 
        # At r=a, term 1: A(a^-m - B)exp(1/0) -> Infinity? 
        # Wait, the standard Dzugutov potential is:
        # V(r) = A(r^-m - B)exp(c/(r-a)) for r < a
        # The term exp(c/(r-a)) goes to infinity as r->a from below.
        # However, usually there is a cutoff or the form is slightly different.
        # Let's check the provided formula: V(r) = A(r^-m - B)exp(c/(r - a)) [r < a]
        # If r approaches a from below, r-a is a small negative number. c/(r-a) is large negative. exp(large negative) is 0.
        # So V(a) = 0.
        # As r -> 0, r^-m -> inf, exp(c/(r-a)) -> exp(-c/a) (finite). So V(r) -> inf.
        # So it's repulsive at short range, attractive at intermediate, zero at a and b.
        
        # Calculation
        term1 = r_vals ** (-m) - B
        term2 = np.exp(c / (r_vals - a))
        e_a = A * term1 * term2
        energy += np.sum(e_a)
        
    # Calculate for a <= r < b
    if np.any(mask_b):
        rows, cols = np.where(mask_b)
        mask_upper = rows < cols
        rows_u = rows[mask_upper]
        cols_u = cols[mask_upper]
        
        r_vals = dist[rows_u, cols_u]
        
        # Potential term: B * exp(d / (r - b))
        # As r -> b from below, r-b is small negative. d/(r-b) is large negative. exp -> 0.
        # As r -> a, r-b is negative. d/(r-b) is negative. exp is finite.
        # Continuity check at r=a:
        # Term 1 limit: A(a^-m - B) * 0 = 0.
        # Term 2 at a: B * exp(d / (a - b)).
        # Are they continuous? 
        # A(a^-m - B) is not necessarily 0. 
        # The Dzugutov potential is usually constructed to be continuous.
        # Let's assume the provided constants ensure continuity or the evaluator handles it.
        # Actually, looking at literature, the Dzugutov potential is:
        # V(r) = A(r^-m - B) exp(c/(r-a)) for r < a
        # V(r) = B exp(d/(r-b)) for a <= r < b
        # It is designed such that V(a) = 0 from both sides?
        # Left side: exp(c/(a-a)) is undefined/infinite? No, limit r->a- is exp(-inf) = 0.
        # Right side: B exp(d/(a-b)). This is non-zero unless d=0 or a=b.
        # There is a discontinuity at r=a in this specific formula if interpreted naively.
        # However, the task says "Energies are recomputed exactly". I must implement exactly what is given.
        # V(r) = ... [r < a] + ... [r < b]. The notation [condition] usually means 1 if true, 0 if false.
        # So for r < a, only first term. For a <= r < b, only second term (since first term's condition r<a is false).
        # Wait, the notation `V(r) = Term1 [r < a] + Term2 [r < b]` implies:
        # If r < a: V = Term1 + Term2 (if r < b is also true, which it is since a < b).
        # If a <= r < b: V = 0 + Term2.
        # If r >= b: V = 0.
        
        # Let's re-read carefully: `V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]`
        # This means:
        # For r < a: V = A(...)exp(...) + B(...)exp(...)
        # For a <= r < b: V = 0 + B(...)exp(...)
        
        # My previous implementation assumed mutual exclusivity. Let's correct that.
        
        pass # The logic above needs to be updated in the final function.

def _dzugutov_energy(pos):
    """
    Compute Dzugutov energy exactly as specified.
    V(r) = A(r^-m - B)exp(c/(r - a)) * (r < a) + B exp(d/(r - b)) * (r < b)
    """
    n = len(pos)
    if n < 2:
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Masks
    mask_a = dist < a
    mask_b = dist < b
    
    # We sum over i < j
    # To avoid double counting, we can sum the whole matrix and divide by 2, 
    # but we must ensure diagonal is 0 (it is inf, so masked out).
    # However, the conditions are per-pair.
    
    energy = 0.0
    
    # Term 1: A(r^-m - B)exp(c/(r - a)) for r < a
    if np.any(mask_a):
        # Get upper triangle indices where mask is true
        rows, cols = np.where(mask_a)
        mask_upper = rows < cols
        r_vals = dist[rows[mask_upper], cols[mask_upper]]
        
        # Avoid division by zero or invalid ops if r is very close to a?
        # r < a, so r - a is negative.
        term1 = A * (r_vals ** (-m) - B) * np.exp(c / (r_vals - a))
        energy += np.sum(term1)
        
    # Term 2: B exp(d/(r - b)) for r < b
    if np.any(mask_b):
        rows, cols = np.where(mask_b)
        mask_upper = rows < cols
        r_vals = dist[rows[mask_upper], cols[mask_upper]]
        
        # r < b, so r - b is negative.
        term2 = B * np.exp(d / (r_vals - b))
        energy += np.sum(term2)
        
    return energy

def _local_minimize(pos):
    """
    Use scipy minimize with L-BFGS-B or similar to relax the cluster.
    We use the provided `relax` function if it's fast, but we can also use scipy.
    The task says `chem_clusters.relax` may be called. It's likely optimized.
    However, `relax` might have its own convergence criteria.
    Let's use the provided `relax` for the heavy lifting of local minimization,
    as it is part of the environment and likely robust.
    """
    try:
        energy, relaxed_pos = relax(pos, 'dzugutov', None)
        return relaxed_pos, energy
    except Exception:
        # Fallback to scipy if relax fails
        try:
            # We need a gradient for scipy to be efficient.
            # dzugutov_energy_gradient is available.
            def fun(x):
                return _dzugutov_energy(x.reshape(-1, 3))
            
            def jac(x):
                # The provided gradient function returns (energy, grad) or just grad?
                # Task says: `chem_clusters.dzugutov_energy_gradient(positions)`
                # Usually these return just the gradient or (E, G).
                # Let's assume it returns the gradient vector.
                # If it returns (E, G), we need to unpack.
                # Let's check typical signatures. Often it's just grad.
                # To be safe, let's try calling it.
                # If it fails, we use finite difference.
                try:
                    res = dzugutov_energy_gradient(x.reshape(-1, 3))
                    if isinstance(res, tuple):
                        return res[1].flatten()
                    return res.flatten()
                except:
                    # Finite difference fallback
                    eps = 1e-5
                    grad = np.zeros_like(x)
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += eps
                        x_minus[i] -= eps
                        grad[i] = (fun(x_plus.reshape(-1, 3)) - fun(x_minus.reshape(-1, 3))) / (2 * eps)
                    return grad
            
            res = minimize(fun, pos.flatten(), jac=jac, method='L-BFGS-B', options={'maxiter': 500})
            return res.x.reshape(-1, 3), res.fun
        except Exception:
            return pos, _dzugutov_energy(pos)

def _basin_hopping_worker(args):
    """
    Worker function for parallel basin hopping.
    args: (n, seed, time_limit, start_pos)
    """
    n, seed, time_limit, start_pos = args
    rng = np.random.default_rng(seed)
    best_energy = float('inf')
    best_pos = None
    
    # Initial local minimization
    try:
        pos, energy = _local_minimize(start_pos)
        best_energy = energy
        best_pos = pos
    except Exception:
        return None
    
    current_pos = best_pos
    current_energy = best_energy
    
    # Basin Hopping Loop
    # Step size: random displacement
    # For n atoms, typical step is ~0.1-0.5
    step_size = 0.25
    
    while time.time() < time_limit:
        # Generate perturbation
        perturbation = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Local minimization
        try:
            min_pos, min_energy = _local_minimize(new_pos)
        except Exception:
            continue
            
        # Metropolis acceptance
        delta_e = min_energy - current_energy
        if delta_e < 0:
            # Always accept downhill
            current_pos = min_pos
            current_energy = min_energy
            if min_energy < best_energy:
                best_energy = min_energy
                best_pos = min_pos
        else:
            # Accept uphill with probability exp(-delta_e / T)
            # T should be small, e.g., 0.01 or 0.1
            T = 0.05
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                current_pos = min_pos
                current_energy = min_energy
        
        # Check time
        if time.time() > time_limit:
            break
            
    return best_pos, best_energy

def _generate_initial_guess(n, rng):
    """
    Generate a good initial guess for basin hopping.
    Use an icosahedral packing as a starting point, then fill randomly.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts * (1.0 / norms[:, np.newaxis]) # Unit icosahedron
    
    # Scale to fit n atoms
    # Approximate volume of n atoms with packing fraction ~0.74
    # V ~ n * r_min^3 / 0.74
    # r_min ~ 0.5 (constraint)
    # Let's just scale the icosahedron to have a radius that fits n atoms roughly.
    # Radius R ~ (n)^(1/3) * spacing
    spacing = 1.0
    R = spacing * (n / 13.0) ** (1.0/3.0)
    
    pos_list = []
    
    # Add center
    if n > 1:
        pos_list.append(np.array([0.0, 0.0, 0.0]))
        
    # Add icosahedron shell
    if n > 13:
        pos_list.extend(ico_verts * R)
        
    # Add more shells or random points
    current_n = len(pos_list)
    if current_n > n:
        # Truncate
        pos_list = pos_list[:n]
    else:
        # Add random points in a sphere of radius R * 1.5
        while len(pos_list) < n:
            p = rng.uniform(-R*1.5, R*1.5, 3)
            # Check distance to existing
            if all(np.linalg.norm(p - q) > 0.5 for q in pos_list):
                pos_list.append(p)
            else:
                # If too close, try again or push out
                # For simplicity, just add and let relaxation fix it, 
                # but this might cause high initial energy.
                # Better to reject and resample a few times.
                pass
        
        # If we still don't have enough, just add random
        while len(pos_list) < n:
            p = rng.uniform(-R*2.0, R*2.0, 3)
            pos_list.append(p)
            
    return np.array(pos_list[:n])

def construct(instance):
    """
    Construct the lowest-energy structure using parallel Basin Hopping.
    """
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Prepare tasks for parallel workers
    # We divide the time budget among the workers.
    # Each worker runs for (deadline - start_time) / NUM_THREADS? 
    # No, they run concurrently. So each worker has the full time budget.
    # But we need to manage the pool.
    
    # Generate diverse starting points
    rng_master = np.random.default_rng(size)
    
    tasks = []
    for i in range(NUM_THREADS):
        seed = size * 1000 + i
        start_pos = _generate_initial_guess(size, np.random.default_rng(seed))
        # Slightly perturb the start to ensure they don't converge to the same basin immediately
        start_pos += np.random.default_rng(seed + 1).normal(0, 0.1, start_pos.shape)
        tasks.append((size, seed, deadline, start_pos))
        
    # Use multiprocessing pool
    # Note: On some systems, creating a pool has overhead.
    # For 4 processes and 300s, it's negligible.
    
    try:
        with mp.Pool(processes=NUM_THREADS) as pool:
            # Use async to allow checking time
            async_results = [pool.apply_async(_basin_hopping_worker, args=(task,)) for task in tasks]
            
            best_energy = float('inf')
            best_pos = None
            
            # Wait for results
            while True:
                current_time = time.time()
                if current_time > deadline:
                    break
                
                for res in async_results:
                    if res.ready():
                        try:
                            pos, energy = res.get()
                            if pos is not None and energy < best_energy:
                                best_energy = energy
                                best_pos = pos
                        except Exception:
                            pass
                            
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
    except Exception as e:
        # Fallback to single-threaded if multiprocessing fails
        print(f"Multiprocessing failed: {e}")
        # Run single worker
        task = tasks[0]
        pos, energy = _basin_hopping_worker(task)
        if pos is not None:
            best_pos = pos
            best_energy = energy
            
    # If no result, fallback to a simple relaxation of a random guess
    if best_pos is None:
        start_pos = _generate_initial_guess(size, np.random.default_rng(12345))
        best_pos, best_energy = _local_minimize(start_pos)
        
    return best_pos
# EVOLVE-BLOCK-END
