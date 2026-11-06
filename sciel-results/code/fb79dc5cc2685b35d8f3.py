import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _build_mackay_55():
    """
    Build the 55-atom Mackay icosahedron: 1 + 12 + 42.
    Returns positions array of shape (55, 3) and metadata.
    """
    ico_verts = _get_icosahedron_vertices()  # 12 unit vectors
    phi = (1 + np.sqrt(5)) / 2
    
    # The 55-atom Mackay icosahedron has shells:
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at icosahedron vertices, radius r1
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers), radius r2
    
    # For Dzugutov potential, optimal nearest-neighbor distance is around 1.0
    # Let's set r1 = 1.0 (first shell radius)
    r1 = 1.0
    # r2 for the second shell in a Mackay icosahedron
    # The edge midpoint of a unit icosahedron is at distance sqrt(1 + 1/phi^2) from center... 
    # Actually for Mackay icosahedron, the second shell radius is r1 * sqrt(1 + 1/phi^2)
    r2 = r1 * np.sqrt(1 + 1.0 / (phi ** 2))
    
    center = np.zeros(3)
    shell1 = ico_verts * r1
    
    # Find the 30 edges of the icosahedron (pairs at minimum distance)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    
    # 30 edge midpoints
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 0:
                    edge_mids.append(mid / mid_norm * r2)
    
    # 20 face centers (triangles formed by 3 mutually adjacent vertices)
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                    fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                    fc_norm = np.linalg.norm(fc)
                    if fc_norm > 0:
                        face_centers.append(fc / fc_norm * r2)
    
    # We need exactly 42 atoms in shell 2: 30 edge midpoints + 12 face centers
    # But there are 20 face centers. We need to select 12.
    # In the Mackay icosahedron, all 20 face centers are present, giving 30+20=50 in shell 2, total 1+12+50=63.
    # Wait, the standard Mackay icosahedron M(2) = 1+12+42 = 55.
    # The 42 are: 30 edge midpoints + 12 face centers? No, that's not right.
    # Actually M(2) has 1 + 12 + 42 = 55. The 42 are arranged as 30 edge midpoints + 12 face centers?
    # Let me reconsider. The second shell of a Mackay icosahedron of order 2 has 42 atoms.
    # These are located at the midpoints of the 30 edges and at the centers of 12 of the 20 faces.
    # Actually, I think the correct decomposition is different. Let me just use all 30 edge midpoints
    # and select 12 face centers to get 42.
    
    # For now, use 30 edge midpoints + first 12 face centers
    shell2 = np.vstack([np.array(edge_mids[:30]), np.array(face_centers[:12])])
    
    positions = np.vstack([center[None], shell1, shell2])
    
    return positions, {'r1': r1, 'r2': r2, 'shell1_idx': (1, 13), 'shell2_idx': (13, 55)}

def _generate_defect_seeds(size, rng, n_seeds=8):
    """
    Generate diverse defect-seeded structures from the 55-atom Mackay icosahedron.
    For size < 55: remove atoms from the outer shell.
    For size > 55: add atoms on a new outer shell.
    For size == 55: use the perfect structure with various perturbations.
    """
    mackay55, meta = _build_mackay_55()
    seeds = []
    
    if size == 55:
        # Use the perfect Mackay icosahedron with various perturbations
        for i in range(n_seeds):
            if i == 0:
                seeds.append(mackay55.copy())
            else:
                perturbed = mackay55.copy()
                # Random perturbation
                noise_scale = 0.05 + 0.05 * i
                perturbed += rng.normal(0, noise_scale, (55, 3))
                seeds.append(perturbed)
    elif size < 55:
        # Remove atoms from the outer shell (shell 2, indices 13-54)
        # Keep center + shell1 (13 atoms) and add (size-13) from shell2
        n_keep_shell2 = size - 13
        if n_keep_shell2 < 0:
            # Need to remove from shell1 too
            # This is rare for our sizes (38-54)
            n_keep_shell2 = 0
            n_keep_shell1 = size - 1
            # Keep center + n_keep_shell1 from shell1
            shell1_atoms = mackay55[1:13]
            # Select n_keep_shell1 atoms from shell1
            idx = rng.choice(12, size=min(n_keep_shell1, 12), replace=False)
            kept = np.vstack([mackay55[0:1], shell1_atoms[idx]])
            seeds.append(kept[:size])
        else:
            shell2_atoms = mackay55[13:55]
            for i in range(n_seeds):
                if i == 0:
                    # Keep first n_keep_shell2 atoms from shell2
                    idx = np.arange(n_keep_shell2)
                else:
                    # Random selection
                    idx = rng.choice(42, size=n_keep_shell2, replace=False)
                kept = np.vstack([mackay55[0:13], shell2_atoms[idx]])
                # Add small perturbation for diversity
                if i > 0:
                    kept += rng.normal(0, 0.03, (size, 3))
                seeds.append(kept)
    else:
        # size > 55: add atoms on a new outer shell
        extra = size - 55
        r3 = meta['r2'] * 1.2
        for i in range(n_seeds):
            base = mackay55.copy()
            # Add extra atoms on a sphere of radius r3 using Fibonacci spiral
            extra_atoms = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for j in range(extra):
                y = 1 - 2 * (j + 0.5) / extra
                radius_at_y = np.sqrt(1 - y ** 2)
                theta = golden_angle * j
                x = np.cos(theta) * radius_at_y
                z = np.sin(theta) * radius_at_y
                extra_atoms.append([x, y, z] * r3)
            extra_atoms = np.array(extra_atoms)
            new_pos = np.vstack([base, extra_atoms])
            if i > 0:
                new_pos += rng.normal(0, 0.03, (size, 3))
            seeds.append(new_pos)
    
    return seeds

def _basin_hopping(positions, rng, time_limit, initial_temp=0.1, initial_step=0.15):
    """
    Adaptive basin hopping with energy-based acceptance.
    """
    current_pos = positions.copy()
    try:
        current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    except:
        return positions, float('inf')
    
    best_energy = current_energy
    best_pos = current_relaxed.copy()
    
    temperature = initial_temp
    step_size = initial_step
    consecutive_rejects = 0
    improvements = 0
    
    while time.time() < time_limit:
        # Adaptive step size
        if improvements > 0:
            step_size *= 0.95
            improvements = 0
        elif consecutive_rejects > 15:
            step_size *= 1.2
            consecutive_rejects = 0
        
        step_size = np.clip(step_size, 0.03, 0.35)
        
        perturbation = rng.normal(0, step_size, size=(len(positions), 3))
        new_pos = current_pos + perturbation
        
        try:
            new_energy, new_relaxed = relax(new_pos, POTENTIAL)
        except:
            continue
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
            current_pos = new_relaxed.copy()
            current_energy = new_energy
            consecutive_rejects = 0
            improvements += 1
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0:
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                consecutive_rejects = 0
            else:
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_pos = new_relaxed.copy()
                    current_energy = new_energy
                consecutive_rejects += 1
        
        # Adaptive temperature
        if consecutive_rejects > 30:
            temperature *= 1.5
            consecutive_rejects = 0
        else:
            temperature *= 0.95
        
        temperature = max(temperature, 0.003)
    
    return best_pos, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 285.0  # Safety margin
    
    size = instance["n"]
    start_time = time.time()
    
    # Time budget: distribute among 6 instances
    # Total 285s, ~47s per instance
    instance_deadline = min(start_time + 45.0, _GLOBAL_DEADLINE - 5.0)
    
    rng = np.random.default_rng(42 + size * 7)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse defect-seeded structures
    n_seeds = 6
    seeds = _generate_defect_seeds(size, rng, n_seeds)
    
    # Relax each seed and keep the best
    for seed in seeds:
        if time.time() > instance_deadline - 10:
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            continue
    
    if best_positions is None:
        # Fallback: random compact structure
        radius = 0.8 * size ** (1 / 3) + 1.0
        best_positions = rng.uniform(-radius, radius, (size, 3))
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass
    
    # Run multiple basin hopping trajectories
    remaining_time = instance_deadline - time.time()
    if remaining_time > 5:
        # Allocate time to multiple BH trajectories
        n_bh = min(4, max(1, int(remaining_time / 10)))
        time_per_bh = (remaining_time - 3) / n_bh
        
        for i in range(n_bh):
            if time.time() > instance_deadline - 3:
                break
            
            # Start BH from best position or a perturbed version
            if i == 0:
                start_pos = best_positions.copy()
            elif i == 1:
                start_pos = best_positions + rng.normal(0, 0.1, (size, 3))
            elif i == 2:
                # Start from a different seed
                if i < len(seeds):
                    start_pos = seeds[i].copy()
                else:
                    start_pos = best_positions + rng.normal(0, 0.2, (size, 3))
            else:
                start_pos = best_positions + rng.normal(0, 0.15, (size, 3))
            
            bh_deadline = time.time() + time_per_bh
            try:
                bh_pos, bh_energy = _basin_hopping(start_pos, rng, bh_deadline)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except:
                continue
    
    # Final relaxation
    if time.time() < instance_deadline - 2:
        try:
            e, pos = relax(best_positions, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            pass
    
    return best_positions
