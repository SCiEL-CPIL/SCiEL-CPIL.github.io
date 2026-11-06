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


def _get_icosahedron_edges():
    """Return the 30 edges of the icosahedron as index pairs."""
    verts = _get_icosahedron_vertices()
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
            dists[j, i] = dists[i, j]
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    return edges


def _generate_13_icosahedron(scale=1.0):
    """Generate a 13-atom Mackay icosahedron (1 center + 12 shell)."""
    verts = _get_icosahedron_vertices() * scale
    center = np.zeros(3)
    return np.vstack([center, verts])


def _generate_55_mackay(scale=1.0):
    """
    Generate a 55-atom Mackay icosahedron.
    Shell 1: 12 atoms at icosahedron vertices (radius r1)
    Shell 2: 30 atoms at edge midpoints (radius r2) + 12 atoms at face centers (radius r3)
    Actually: 1 + 12 + 42 = 55
    The 42 atoms: 30 at edge midpoints, 12 at face centers (dodecahedron vertices)
    """
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = _get_icosahedron_vertices() * scale
    edges = _get_icosahedron_edges()
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers - use dodecahedron vertices scaled appropriately
    # The 12 face centers of an icosahedron correspond to dodecahedron vertices
    # Compute them as centroids of the 20 triangular faces
    # But we need only 12 unique face normals... actually an icosahedron has 20 faces.
    # Wait, the 55-atom Mackay has 42 atoms in the outer shell.
    # 30 edge midpoints + 12 face centers? But icosahedron has 20 faces.
    # Let me reconsider: the 42 atoms are at positions corresponding to a truncated icosahedron?
    # No. Let me just use the geometric construction:
    # The 55-atom cluster has atoms at:
    # 1 center
    # 12 at icosahedron vertices (radius r1)
    # 42 at: 30 edge midpoints + 12 "face center" positions
    # The 12 face center positions are actually the vertices of the dual dodecahedron
    # scaled to the appropriate radius.
    
    # Dodecahedron vertices (dual of icosahedron)
    dod_verts_raw = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ]
    dod_verts = np.array(dod_verts_raw, dtype=float)
    dod_norms = np.linalg.norm(dod_verts, axis=1, keepdims=True)
    dod_verts = dod_verts / dod_norms
    
    # The edge midpoints are at radius:
    r1 = scale
    d_edge = np.linalg.norm(ico_verts[edges[0][0]] - ico_verts[edges[0][1]])
    r_edge = np.sqrt(r1**2 - (d_edge/2)**2)
    
    # Scale dodecahedron vertices to same radius as edge midpoints
    dod_scaled = dod_verts * r_edge
    
    center = np.zeros(3)
    return np.vstack([center, ico_verts, edge_mids, dod_scaled])


def _generate_decahedron(size, scale=1.0):
    """
    Generate a decahedral cluster seed.
    For 13 atoms: same as icosahedron (decahedron is not well-defined for 13)
    For larger sizes: build from tetrahedral building blocks
    """
    if size == 13:
        return _generate_13_icosahedron(scale)
    
    # For decahedral clusters, use a different packing
    # Start with 13-atom icosahedron and add atoms in tetrahedral positions
    base = _generate_13_icosahedron(scale)
    if size <= 13:
        return base[:size]
    
    current = base.copy()
    verts = _get_icosahedron_vertices() * scale
    rng = np.random.default_rng(42)
    
    for _ in range(size - 13):
        # Add atom near a random vertex, in a tetrahedral direction
        v_idx = rng.integers(0, 12)
        v = verts[v_idx]
        # Random direction offset
        offset = rng.normal(0, 0.3 * scale, 3)
        new_pos = v + offset
        current = np.vstack([current, new_pos])
    
    return current


def _fragment_merger_seeds(n, num_fragments=3, rng=None):
    """
    Generate seeds by merging pre-relaxed icosahedral fragments.
    This preserves local polytetrahedral order while allowing global diversity.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    seeds = []
    
    # Determine fragment sizes
    if n <= 13:
        fragments = [n]
    elif n <= 26:
        fragments = [13, n - 13]
    elif n <= 39:
        fragments = [13, 13, n - 26]
    elif n <= 52:
        fragments = [13, 13, 13, n - 39]
    else:
        # For larger clusters, use a mix
        if n <= 55:
            fragments = [13, 13, 13, 13, n - 52] if n > 52 else [13, 13, 13, 13]
        else:
            # Use 55-atom base and add
            fragments = [55] + [n - 55] if n > 55 else [55]
    
    # Filter out zero/negative fragments
    fragments = [f for f in fragments if f > 0]
    
    # Generate fragment positions
    for trial in range(num_fragments):
        fragment_positions = []
        for fsize in fragments:
            if fsize == 13:
                frag = _generate_13_icosahedron(scale=1.0)
            elif fsize == 55:
                frag = _generate_55_mackay(scale=1.0)
            elif fsize <= 13:
                frag = _generate_13_icosahedron(scale=1.0)[:fsize]
            else:
                # For intermediate sizes, use 13-atom base + extras
                frag = _generate_13_icosahedron(scale=1.0)
                if fsize > 13:
                    extras = rng.normal(0, 0.5, (fsize - 13, 3))
                    frag = np.vstack([frag, extras])
            
            # Relax the fragment
            try:
                e, frag_relaxed = relax(frag, POTENTIAL)
                fragment_positions.append(frag_relaxed)
            except:
                fragment_positions.append(frag)
        
        # Merge fragments with random offsets and rotations
        merged = []
        for i, frag in enumerate(fragment_positions):
            # Random rotation
            angle = rng.uniform(0, 2 * np.pi)
            axis = rng.normal(0, 1, 3)
            axis = axis / np.linalg.norm(axis)
            # Simple rotation around axis
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([
                [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
                [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
                [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]
            ])
            frag_rot = frag @ R.T
            
            # Random offset
            if i == 0:
                offset = np.zeros(3)
            else:
                offset = rng.uniform(-1.5, 1.5, 3)
            
            merged.append(frag_rot + offset)
        
        merged = np.vstack(merged)
        
        # Trim or pad to exact size
        if len(merged) > n:
            merged = merged[:n]
        elif len(merged) < n:
            extras = rng.normal(0, 1.0, (n - len(merged), 3))
            merged = np.vstack([merged, extras])
        
        seeds.append(merged)
    
    return seeds


def _generate_random_compact(size, rng):
    """Generate a random compact seed."""
    radius = 0.8 * size ** (1/3) + 1.0
    points = []
    for _ in range(size):
        found = False
        for _ in range(2000):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
                found = True
                break
        if not found:
            p = rng.uniform(-radius*1.2, radius*1.2, 3)
            points.append(p)
    return np.array(points)


def _basin_hopping(positions, rng, time_limit, initial_temp=0.12, initial_step=0.20):
    """Adaptive basin hopping with energy-based acceptance."""
    current_pos = positions.copy()
    current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    best_energy = current_energy
    best_pos = current_relaxed.copy()
    
    temperature = initial_temp
    step_size = initial_step
    consecutive_rejects = 0
    no_improve_count = 0
    
    while time.time() < time_limit:
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
            consecutive_rejects = 0
            no_improve_count = 0
            step_size *= 0.97
        else:
            delta_e = new_energy - current_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                consecutive_rejects += 1
                no_improve_count += 1
            else:
                consecutive_rejects += 1
                no_improve_count += 1
        
        if consecutive_rejects > 15:
            temperature *= 1.3
            step_size *= 1.15
            consecutive_rejects = 0
        else:
            temperature *= 0.96
            step_size *= 0.995
        
        if no_improve_count > 30:
            # Try a larger perturbation
            big_pert = rng.normal(0, step_size * 2, size=(len(positions), 3))
            big_pos = current_pos + big_pert
            try:
                big_energy, big_relaxed = relax(big_pos, POTENTIAL)
                if big_energy < best_energy:
                    best_energy = big_energy
                    best_pos = big_relaxed.copy()
                    current_pos = big_relaxed.copy()
                    no_improve_count = 0
                    step_size *= 0.95
                else:
                    no_improve_count = 0
            except:
                pass
        
        temperature = max(temperature, 0.005)
        step_size = min(step_size, 0.6)
    
    return best_pos, best_energy


def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 280.0

    size = instance["n"]
    start_time = time.time()
    
    # Allocate time per instance with safety margin
    instance_deadline = min(start_time + 42.0, _GLOBAL_DEADLINE - 5.0)
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    all_seeds = []
    
    # 1. Mackay icosahedron seeds
    try:
        if size == 13:
            mackay_seed = _generate_13_icosahedron(1.0)
        elif size == 55:
            mackay_seed = _generate_55_mackay(1.0)
        elif size <= 13:
            mackay_seed = _generate_13_icosahedron(1.0)[:size]
        elif size <= 55:
            # Grow from 13-atom base
            mackay_seed = _generate_13_icosahedron(1.0)
            extra = rng.normal(0, 0.5, (size - 13, 3))
            mackay_seed = np.vstack([mackay_seed, extra])
        else:
            # Grow from 55-atom base
            mackay_seed = _generate_55_mackay(1.0)
            extra = rng.normal(0, 0.5, (size - 55, 3))
            mackay_seed = np.vstack([mackay_seed, extra])
        
        if mackay_seed.shape[0] == size:
            all_seeds.append(mackay_seed)
    except:
        pass
    
    # 2. Fragment merger seeds (key innovation)
    try:
        frag_seeds = _fragment_merger_seeds(size, num_fragments=4, rng=rng)
        all_seeds.extend(frag_seeds)
    except:
        pass
    
    # 3. Random compact seeds
    for i in range(2):
        rng_sub = np.random.default_rng(42 + size + i * 777)
        all_seeds.append(_generate_random_compact(size, rng_sub))
    
    # 4. Decahedral seed
    try:
        deca_seed = _generate_decahedron(size, scale=1.0)
        if deca_seed.shape[0] == size:
            all_seeds.append(deca_seed)
    except:
        pass
    
    # Phase 1: Relax all seeds and keep the best
    for seed in all_seeds:
        if time.time() > instance_deadline - 20:
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            continue
    
    if best_positions is None:
        best_positions = np.random.randn(size, 3) * 0.3
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass
    
    # Phase 2: Basin hopping from the best seed
    remaining_time = instance_deadline - time.time()
    if remaining_time > 8:
        bh_pos, bh_energy = _basin_hopping(best_positions, rng, time.time() + remaining_time - 3)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    return best_positions
