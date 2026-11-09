"""Sutton-Chen cluster global minima via adaptive Basin Hopping with error repair."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    # Vectorized distance check for min distance
    # For n up to ~80, O(n^2) is fine. 
    # We can do a quick check.
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True

def _repair_clashes(pos, min_dist=0.36, max_iter=100):
    """
    Error repair: If any pair is closer than min_dist, push them apart.
    This is a simple iterative relaxation to remove invalid configurations.
    """
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(max_iter):
        min_d = np.inf
        min_pair = None
        # Find closest pair
        for i in range(n - 1):
            d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
            idx = np.argmin(d)
            if d[idx] < min_d:
                min_d = d[idx]
                min_pair = (i, i + 1 + idx)
        
        if min_d >= min_dist:
            break
        
        i, j = min_pair
        vec = pos[j] - pos[i]
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            # Randomly nudge
            pos[j] += np.random.normal(0, 0.01, 3)
        else:
            # Push apart to min_dist
            target_vec = vec / norm * min_dist
            # Move both half the distance needed
            diff = target_vec - vec
            pos[i] -= 0.5 * diff
            pos[j] += 0.5 * diff
            
    return pos

def _generate_initial_seeds(n):
    """Generate a few diverse starting points for Basin Hopping."""
    seeds = []
    
    # 1. FCC-like
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    pts = []
    lattice = 0.72
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    fcc_seed = pts[order[:n]]
    seeds.append(fcc_seed)

    # 2. HCP-like
    a = 0.72
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    hcp_seed = pts[order[:n]]
    seeds.append(hcp_seed)

    # 3. Random compact
    rng = np.random.default_rng(42)
    max_radius = 0.9 * n ** (1.0 / 3.0)
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 200:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            if pts:
                arr = np.array(pts)
                if np.min(np.linalg.norm(arr - p, axis=1)) < 0.7:
                    ok = False
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        pts.append(p)
    seeds.append(np.array(pts))
        
    return seeds

def construct(instance):
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Generate initial seeds and relax them to get a good starting point for BH
    seeds = _generate_initial_seeds(n)
    
    best_e = np.inf
    best_pos = None
    best_grad_norm = 0.0
    
    # Phase 1: Relax initial seeds to find a good starting valley
    for s in seeds:
        if time.time() > deadline:
            break
        s = _repair_clashes(s)
        if not _validate_positions(s):
            continue
        try:
            e, p = relax(s, potential, kind)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_e:
                    best_e = e
                    best_pos = p.copy()
                    # Get gradient to estimate sharpness
                    _, g = sutton_chen_energy_gradient(p, kind)
                    best_grad_norm = np.linalg.norm(g)
        except Exception:
            pass

    if best_pos is None:
        # Fallback: just return a valid random cluster
        pos = _repair_clashes(np.random.default_rng(0).normal(0, 1, (n, 3)))
        return pos

    # Phase 2: Basin Hopping
    # Adaptive perturbation radius
    # Start with a radius that allows exploration, shrink over time
    # We track 'steps' to schedule the radius
    current_radius = 0.5 
    min_radius = 0.1
    max_radius = 1.5
    
    # We will run BH until time runs out
    # We use a simple acceptance: Metropolis-like or just downhill?
    # For global minima, downhill (accept if E_new < E_old) is standard for finding the best local min,
    # but sometimes accepting uphill moves helps escape. 
    # Given the time constraint and the "global minimum" goal, strict downhill is safer to guarantee improvement,
    # but we can add a small probability of uphill acceptance if we are stuck.
    
    # Let's stick to a robust downhill BH with adaptive radius.
    # If we get stuck (no improvement for X steps), we increase radius slightly (reheat).
    
    steps_since_improvement = 0
    max_steps_no_improve = 20
    
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline - 5.0:
        # Adaptive radius based on gradient norm and stagnation
        # If gradient is large, we are far from min, can take bigger steps? 
        # Actually, standard BH uses a fixed radius or annealing radius.
        # Let's use a radius that shrinks slowly but can expand if stuck.
        
        # Reheat if stuck
        if steps_since_improvement > max_steps_no_improve:
            current_radius = min(max_radius, current_radius * 1.5)
            steps_since_improvement = 0
            max_steps_no_improve = int(max_steps_no_improve * 1.2) # Make it harder to reheat again
            
        # Scale radius by gradient? 
        # If gradient is small (sharp min), we might need to jump further to escape.
        # If gradient is large (flat), smaller steps might be better to stay in basin.
        # However, the minimizer will handle the local part. The radius is for the *perturbation*.
        # A common heuristic: radius ~ 1/sqrt(gradient_norm) or similar, but capped.
        
        # Let's just use the scheduled radius.
        
        # Generate perturbation
        # We perturb the best_pos
        perturbation = rng.normal(0, current_radius, size=(n, 3))
        new_pos = best_pos + perturbation
        
        # Error Repair: Fix any clashes before minimization
        new_pos = _repair_clashes(new_pos)
        
        if not _validate_positions(new_pos):
            # If repair failed, try a smaller perturbation
            continue
            
        # Minimize
        try:
            e_new, p_new = relax(new_pos, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(e_new) or not _validate_positions(p_new):
            continue
            
        # Acceptance
        if e_new < best_e:
            best_e = e_new
            best_pos = p_new.copy()
            steps_since_improvement = 0
            # Update gradient norm for potential future use (not strictly needed for simple BH)
            try:
                _, g = sutton_chen_energy_gradient(p_new, kind)
                best_grad_norm = np.linalg.norm(g)
            except:
                pass
            # Shrink radius slowly on success (annealing)
            current_radius = max(min_radius, current_radius * 0.95)
        else:
            steps_since_improvement += 1

    # Final polish: One more relaxation of the best found
    if time.time() < deadline - 2.0:
        try:
            e_final, p_final = relax(best_pos, potential, kind)
            if np.isfinite(e_final) and _validate_positions(p_final):
                best_e = e_final
                best_pos = p_final
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
