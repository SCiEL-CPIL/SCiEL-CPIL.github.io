"""New program: Multistart Basin Hopping with Lattice-based Initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

def _get_kind_params(kind):
    """Map kind string to (n_exp, m_exp, c)."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41

def _compute_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy efficiently."""
    n = len(positions)
    if n == 0:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_r_n = np.power(dists, -n_exp)
    inv_r_m = np.power(dists, -m_exp)
    
    sum_inv_r_n = np.sum(inv_r_n, axis=1)
    sum_inv_r_m = np.sum(inv_r_m, axis=1)
    
    e_rep = 0.5 * np.sum(sum_inv_r_n)
    
    sum_inv_r_m_no_self = sum_inv_r_m - 1.0
    sum_inv_r_m_no_self = np.maximum(sum_inv_r_m_no_self, 0.0)
    e_att = -c * np.sum(np.sqrt(sum_inv_r_m_no_self))
    
    return e_rep + e_att

def _local_minimize(positions, potential, kind):
    """Use the provided relax helper for local minimization."""
    from autoresearch_suite.chem_clusters import relax
    try:
        e, p = relax(positions, potential, kind)
        if np.isfinite(e) and np.all(np.isfinite(p)):
            return e, p
    except Exception:
        pass
    return np.inf, positions

def _generate_lattice_seed(n, lattice_type='bcc', spacing=1.0):
    """Generate a seed structure from a lattice."""
    # Estimate radius
    r_est = spacing * (n / 4.0) ** (1.0/3.0) # Rough estimate
    max_coord = int(np.ceil(r_est * 1.5)) + 2
    
    points = []
    if lattice_type == 'bcc':
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                for z in range(-max_coord, max_coord + 1):
                    # Corners
                    points.append(np.array([x, y, z]))
                    # Body center
                    points.append(np.array([x + 0.5, y + 0.5, z + 0.5]))
    elif lattice_type == 'fcc':
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                for z in range(-max_coord, max_coord + 1):
                    # Corners
                    points.append(np.array([x, y, z]))
                    # Face centers
                    points.append(np.array([x + 0.5, y + 0.5, z]))
                    points.append(np.array([x + 0.5, y, z + 0.5]))
                    points.append(np.array([x, y + 0.5, z + 0.5]))
    elif lattice_type == 'hcp':
        # Approximate HCP by stacking FCC layers? Or just use BCC/FCC for simplicity.
        # For small clusters, BCC and FCC are the main competitors.
        # Let's just return BCC for HCP request to keep it simple, or generate a specific HCP fragment.
        # HCP structure:
        # Layer 1: Hexagonal grid
        # Layer 2: Offset
        a = 1.0
        c = 1.633 * a
        for layer in range(-max_coord//2, max_coord//2 + 1):
            z = layer * c
            for i in range(-max_coord, max_coord + 1):
                for j in range(-max_coord, max_coord + 1):
                    if layer % 2 == 0:
                        x = i * a + 0.5 * j * a
                        y = j * a * np.sqrt(3)/2
                    else:
                        x = i * a
                        y = j * a * np.sqrt(3)/2 + a * np.sqrt(3)/2
                    points.append(np.array([x, y, z]))
    
    if not points:
        return np.zeros((n, 3))
        
    points = np.array(points)
    # Center
    points -= np.mean(points, axis=0)
    # Scale to fit n atoms roughly
    # Sort by radius
    radii = np.linalg.norm(points, axis=1)
    idx = np.argsort(radii)
    points = points[idx]
    
    # We need exactly n points.
    if len(points) < n:
        # Pad with random points near the surface
        # Find max radius
        r_max = np.max(radii) if len(radii) > 0 else 1.0
        while len(points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            points = np.vstack([points, p])
            radii = np.append(radii, np.linalg.norm(p))
            idx = np.argsort(radii)
            points = points[idx]
            radii = radii[idx]
            
    return points[:n] * spacing

def _generate_icosahedral_seed(n, spacing=1.0):
    """Generate an icosahedral seed."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    # Add center
    points.append(np.zeros(3))
    # Add first shell
    points.extend(verts)
    # Add second shell (octahedral vertices scaled)
    oct_verts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    scale2 = 0.8 # Approximate
    points.extend(oct_verts * scale2)
    # Add more by rotating icosahedron?
    # For simplicity, just repeat and scale if n is larger
    # A better way: generate points on icosahedral shells
    shells = [
        np.array([[0, 0, 0]]),
        verts,
        verts * 0.7 + np.array([0.3, 0.3, 0.3]), # Rough
        oct_verts * 1.2
    ]
    
    all_points = []
    for shell in shells:
        all_points.extend(shell)
        
    all_points = np.array(all_points)
    all_points -= np.mean(all_points, axis=0)
    radii = np.linalg.norm(all_points, axis=1)
    idx = np.argsort(radii)
    all_points = all_points[idx]
    
    if len(all_points) < n:
        # Pad with random
        r_max = np.max(radii) if len(radii) > 0 else 1.0
        while len(all_points) < n:
            p = np.random.uniform(-r_max, r_max, 3)
            all_points = np.vstack([all_points, p])
            radii = np.append(radii, np.linalg.norm(p))
            idx = np.argsort(radii)
            all_points = all_points[idx]
            radii = radii[idx]
            
    return all_points[:n] * spacing

def _basin_hopping(positions, potential, kind, n_exp, m_exp, c, 
                   T0=1.0, Tend=0.01, steps=50, seed=42, kick=0.2):
    """
    Perform Basin Hopping.
    """
    n = len(positions)
    rng = np.random.default_rng(seed)
    
    # Initial minimize
    E, P = _local_minimize(positions, potential, kind)
    if not np.isfinite(E):
        return np.inf, positions
    
    best_E = E
    best_P = P.copy()
    
    for i in range(steps):
        T = T0 * (0.95 ** i)
        if T < Tend:
            break
            
        # Perturb
        # Randomly displace some atoms
        n_kick = max(1, n // 4)
        idxs = rng.choice(n, size=n_kick, replace=False)
        kicks = rng.normal(0, kick, size=(n_kick, 3))
        P_new = P.copy()
        P_new[idxs] += kicks
        
        # Minimize perturbed
        E_new, P_new = _local_minimize(P_new, potential, kind)
        
        if not np.isfinite(E_new):
            continue
            
        delta_E = E_new - E
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            E = E_new
            P = P_new
            if E < best_E:
                best_E = E
                best_P = P.copy()
                
    return best_E, best_P

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    n_exp, m_exp, c = _get_kind_params(kind)
    
    start_time = time.time()
    time_limit = 280.0 # Leave buffer
    
    # Generate diverse high-quality seeds
    seeds = []
    
    # 1. BCC
    try:
        bcc = _generate_lattice_seed(size, 'bcc', spacing=1.0)
        seeds.append(bcc)
    except Exception:
        pass
        
    # 2. FCC
    try:
        fcc = _generate_lattice_seed(size, 'fcc', spacing=1.0)
        seeds.append(fcc)
    except Exception:
        pass
        
    # 3. HCP
    try:
        hcp = _generate_lattice_seed(size, 'hcp', spacing=1.0)
        seeds.append(hcp)
    except Exception:
        pass
        
    # 4. Icosahedral
    try:
        ico = _generate_icosahedral_seed(size, spacing=1.0)
        seeds.append(ico)
    except Exception:
        pass
        
    # 5. Random Compact (a couple of seeds)
    for s in [42, 123]:
        try:
            # Simple random compact generation
            rng = np.random.default_rng(s)
            radius = 1.0 * size ** (1/3)
            pts = []
            min_dist = 0.5
            attempts = 0
            while len(pts) < size and attempts < size * 500:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                        pts.append(p)
                attempts += 1
            while len(pts) < size:
                pts.append(rng.uniform(-radius, radius, 3))
            seeds.append(np.array(pts))
        except Exception:
            pass
            
    best_E = np.inf
    best_P = None
    
    # Run Basin Hopping on each seed
    # Allocate time per seed
    time_per_seed = time_limit / len(seeds)
    
    for i, seed_pos in enumerate(seeds):
        if time.time() - start_time > time_limit:
            break
            
        # Determine number of BH steps based on time
        # Estimate: local min takes ~0.1s? BH step takes ~0.1s?
        # Let's try 20-50 steps per seed.
        max_steps = 30
        
        try:
            bh_E, bh_P = _basin_hopping(seed_pos, potential, kind, n_exp, m_exp, c, 
                                         T0=0.5, Tend=0.01, steps=max_steps, seed=100+i)
            
            if np.isfinite(bh_E) and bh_E < best_E:
                best_E = bh_E
                best_P = bh_P
        except Exception:
            continue
            
    # If we have time left, run more BH on the best solution found
    if best_P is not None and time.time() - start_time < time_limit - 10:
        remaining_time = time_limit - (time.time() - start_time)
        # Run a few more long BH chains from best_P
        for j in range(3):
            if time.time() - start_time > time_limit:
                break
            # Perturb best slightly and run longer BH
            perturbed = best_P + np.random.normal(0, 0.1, best_P.shape)
            try:
                bh_E, bh_P = _basin_hopping(perturbed, potential, kind, n_exp, m_exp, c, 
                                             T0=0.5, Tend=0.01, steps=50, seed=500+j)
                if np.isfinite(bh_E) and bh_E < best_E:
                    best_E = bh_E
                    best_P = bh_P
            except Exception:
                pass

    # Fallback
    if best_P is None:
        best_P = _generate_lattice_seed(size, 'bcc', spacing=1.0)
        best_E, best_P = _local_minimize(best_P, potential, kind)
        
    # Final validation
    if not np.all(np.isfinite(best_P)):
        best_P = _generate_lattice_seed(size, 'bcc', spacing=1.0)
        best_E, best_P = _local_minimize(best_P, potential, kind)
        
    # Check min distance
    min_d = np.inf
    for i, j in combinations(range(len(best_P)), 2):
        d = np.linalg.norm(best_P[i] - best_P[j])
        if d < min_d:
            min_d = d
            
    if min_d < 0.35:
        # Attempt to fix
        for _ in range(5):
            perturbed = best_P + np.random.normal(0, 0.2, best_P.shape)
            try:
                e, p = _local_minimize(perturbed, potential, kind)
                md = np.inf
                for i, j in combinations(range(len(p)), 2):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < md:
                        md = d
                if md >= 0.35 and np.isfinite(e):
                    best_E, best_P = e, p
                    break
            except Exception:
                continue
                
    return best_P
# EVOLVE-BLOCK-END
