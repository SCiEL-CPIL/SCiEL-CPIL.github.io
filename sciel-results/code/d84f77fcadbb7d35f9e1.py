import numpy as np
import time
import concurrent.futures
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import scipy.linalg

# We will try to use the provided relax if it's fast, but primarily rely on our own vectorized L-BFGS
# The task says chem_clusters.relax and morse_energy_gradient may be called.
# However, to be "materially different" and robust, we implement a high-performance local optimizer
# that takes advantage of the symmetry of the seeds.

def _morse_energy_and_grad(pos, rho):
    """
    Compute Morse energy and gradient vectorized.
    E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)
    dE/dr = -2*rho*exp(rho(1-r))*(exp(rho(1-r)) - 1)
    grad_pos_i = sum_{j!=i} dE/dr * (r_i - r_j) / r_ij
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Compute Morse terms
    # x = rho * (1 - r)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    
    # Energy term per pair: ex * (ex - 2)
    pair_energy = ex * (ex - 2.0)
    total_energy = 0.5 * np.sum(pair_energy) # 0.5 to avoid double counting
    
    # Gradient term per pair: dE/dr = -2 * rho * ex * (ex - 1)
    # Note: d/dr [exp(rho(1-r))(exp(rho(1-r))-2)] 
    # Let u = rho(1-r), du/dr = -rho
    # f(u) = e^u(e^u - 2) = e^(2u) - 2e^u
    # f'(u) = 2e^(2u) - 2e^u = 2e^u(e^u - 1)
    # dE/dr = f'(u) * du/dr = 2e^u(e^u - 1) * (-rho) = -2*rho*e^u(e^u-1)
    dE_dr = -2.0 * rho * ex * (ex - 1.0)
    
    # Gradient vector: grad_i = sum_{j!=i} dE_dr_ij * (pos_i - pos_j) / dist_ij
    # dE_dr is the derivative with respect to distance.
    # Chain rule: dE/dpos_i = sum_j (dE/dr_ij) * (d r_ij / d pos_i)
    # r_ij = ||pos_i - pos_j||
    # d r_ij / d pos_i = (pos_i - pos_j) / r_ij
    
    # We have dE_dr matrix (n x n) and diff matrix (n x n x 3).
    # grad_i = sum_j dE_dr[i,j] * diff[i,j] / dists[i,j]
    
    # Avoid division by zero (diagonal is inf, so dE_dr is 0 there effectively? No, exp(inf) is inf. 
    # We must mask diagonal.)
    mask = np.eye(n, dtype=bool)
    dists_safe = np.where(mask, 1.0, dists)
    
    # Weight matrix for gradient: W[i,j] = dE_dr[i,j] / dists[i,j]
    W = dE_dr / dists_safe
    W[mask] = 0.0
    
    # grad = W @ diff? No.
    # grad_i = sum_j W[i,j] * (pos_i - pos_j)
    # Let's compute: 
    # sum_j W[i,j] * pos_i - sum_j W[i,j] * pos_j
    # = pos_i * sum_j W[i,j] - (W @ pos.T).T
    
    row_sums = np.sum(W, axis=1)
    grad = pos * row_sums[:, np.newaxis] - (W @ pos.T).T
    
    return total_energy, grad.flatten()

def _constraint_min_dist(pos, min_d=0.5):
    """
    Constraint for L-BFGS-B: all pairwise distances >= min_d.
    L-BFGS-B only supports bounds, not general constraints.
    However, we can use SLSQP or handle violations by projecting/penalizing.
    Given the time limit, L-BFGS-B with bounds is much faster.
    But atoms must be at least 0.5 apart.
    If we use L-BFGS-B, we might violate the constraint.
    Strategy: Use L-BFGS-B to minimize energy, then check constraint.
    If violated, apply a small repulsive force or use SLSQP for final polish.
    Actually, for these clusters, the global minima usually satisfy min_dist > 0.5 comfortably 
    (equilibrium is 1.0). Violations happen during optimization if the landscape is rugged.
    We will use a penalty function approach within a custom objective for L-BFGS-B if necessary, 
    or just trust that starting from good symmetric seeds keeps us away from singularities.
    
    Better: Use SLSQP for the final step? It's slow.
    Let's stick to L-BFGS-B and check validity at the end. If invalid, we reject or repair.
    Repair: If min dist < 0.5, push atoms apart.
    """
    pass

def _generate_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron.
    Magic numbers: 13, 55, 147, 309...
    For n not magic, we take the largest icosahedron <= n and add random atoms? 
    Or truncate?
    Doye-Wales minima for n=38, 50, 75, 80 are often based on icosahedral or FCC cores.
    n=38: 13 (center) + 12 (1st shell) + 13 (partial 2nd shell)? 
    Actually, 38 is close to 55 (full 2nd shell) but missing 17.
    Let's generate the 13-atom icosahedron and the 55-atom one.
    """
    # 13 atom icosahedron
    if n <= 13:
        # Center + 12 vertices
        phi = (1 + np.sqrt(5)) / 2
        # Vertices of icosahedron
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        # Normalize to distance 1 from center?
        # The distance from origin to vertex is sqrt(1 + phi^2) = sqrt(1 + 1.618^2) ~ 1.902
        # We want nearest neighbor distance 1.
        # In icosahedron, edge length is 2 if vertices are unit sphere? 
        # Standard icosahedron edge length is 2 for radius sqrt(1+phi^2)/phi? 
        # Let's just scale so that min distance is 1.
        d = np.linalg.norm(verts[0] - verts[1])
        scale = 1.0 / d
        verts *= scale
        pos = np.vstack([np.zeros(3), verts])
        if n < 13:
            pos = pos[:n]
        return pos + rng.normal(0, 0.01, size=(n,3))
    
    # 55 atom icosahedron (13 + 12 + 30)
    # 1st shell: 12 vertices (dist 1 from center)
    # 2nd shell: 30 vertices (dist 2 from center? No, Mackay icosahedron shells are at distances 1, 2, 3... in units of edge length? 
    # Actually, in Mackay icosahedra, the distance from center to shell k is k * (edge_length/2 * something)?
    # Let's use the known coordinates or generate via symmetry.
    # Simpler: Use the 13-atom one and add atoms at positions corresponding to the 2nd shell.
    # The 2nd shell of a Mackay icosahedron consists of 30 atoms.
    # Total 13 + 12 + 30 = 55.
    
    # For n=38, we take 13 + 12 + 13 (partial 2nd shell).
    # For n=50, 13 + 12 + 25.
    # For n=75, 13 + 12 + 30 + 20 (partial 3rd shell, total 3rd shell is 42? 13+12+30+42=97).
    # For n=80, 13 + 12 + 30 + 25.
    
    # Let's generate the full 55-atom icosahedron.
    # Coordinates for 55-atom icosahedron:
    # Center: (0,0,0)
    # Shell 1 (12): The 12 vertices of icosahedron.
    # Shell 2 (30): Midpoints of edges of the icosahedron? No, Mackay icosahedron atoms are at lattice points.
    # It is easier to generate an FCC cluster and an Icosahedron cluster and let the optimizer decide.
    # The icosahedron generator for arbitrary n is complex. 
    # Let's stick to FCC for n > 55 or use a "deformed icosahedron" approach.
    
    # Actually, for n=38, 50, 75, 80, FCC is often competitive or the global min is a distorted FCC.
    # The 13 and 55 are the main icosahedral magic numbers.
    # Let's just generate FCC and a "Spherical" random start, and a "13+12+..." specific start for n=38, 50.
    
    # For now, return FCC for n > 55 to keep it simple and robust, as icosahedral shells get complex to truncate.
    return _generate_fcc_cluster(n, rng)

def _generate_fcc_cluster(n, rng):
    """Generate an FCC cluster of n atoms."""
    # Generate a large FCC lattice
    side = int(np.ceil(n ** (1.0/3.0))) + 3
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i, j, k]) + np.array(b)
                    pts.append(p)
    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    # Add very small noise to break symmetry if needed, but L-BFGS can handle symmetric starts
    # if we perturb slightly.
    pts += rng.normal(0, 0.01, size=pts.shape)
    return pts

def _optimize_single(start_pos, rho, deadline):
    """
    Optimize a single starting configuration using L-BFGS-B.
    """
    n = len(start_pos)
    x0 = start_pos.flatten()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g
    
    # Use L-BFGS-B
    # We don't have easy bounds for pairwise distances in L-BFGS-B.
    # We rely on the fact that the energy goes to infinity as r->0.
    # The optimizer should avoid collisions.
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 10000})
        if res.success or res.status == 1: # status 1 is max iterations
            final_pos = res.x.reshape(-1, 3)
            final_e, _ = _morse_energy_and_grad(final_pos, rho)
            
            # Check min distance
            diff = final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d >= 0.5:
                return final_e, final_pos
            else:
                # If invalid, try to repair by pushing apart
                # Simple repair: scale up? No, that increases energy.
                # Just return invalid to be safe, or return the position and let caller check.
                return np.inf, final_pos
        else:
            return np.inf, start_pos
    except Exception:
        return np.inf, start_pos

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance.get("potential", 'morse')
    
    start_time = time.time()
    deadline = start_time + 45.0 # Safe margin for 6 calls in 300s
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse starts
    starts = []
    
    # 1. FCC
    starts.append(_generate_fcc_cluster(n, rng))
    
    # 2. FCC with different noise
    starts.append(_generate_fcc_cluster(n, rng))
    
    # 3. Random Packed (Sphere)
    radius = n ** (1.0/3.0) * 1.1
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
        attempts += 1
    if len(pts) < n:
        # Fill with random
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p)
    starts.append(np.array(pts))
    
    # 4. BCC-like (Cubic)
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    starts.append(pts[indices[:n]] + rng.normal(0, 0.05, size=(n,3)))
    
    # Parallelize optimization
    best_energy = np.inf
    best_positions = starts[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for s in starts:
            if time.time() > deadline - 5.0:
                break
            futures.append(executor.submit(_optimize_single, s, rho, deadline))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
    
    # If we still have time, do a quick basin hopping from the best
    if time.time() < deadline - 10.0 and best_energy < np.inf:
        current_energy = best_energy
        current_positions = best_positions
        kick_sigma = 0.1
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            new_pos = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
            e, pos = _optimize_single(new_pos, rho, deadline)
            if e < current_energy:
                current_energy = e
                current_positions = pos
            else:
                # Metropolis
                T = 0.1
                if np.random.rand() < np.exp(-(e - current_energy) / T):
                    current_energy = e
                    current_positions = pos
    
    return best_positions
