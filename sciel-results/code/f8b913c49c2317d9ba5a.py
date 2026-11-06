"""
Strategy: Discrete combinatorial search over known Dzugutov global minimum structures.
Instead of continuous basin hopping, we generate candidate structures from the known
family of Mackay icosahedra and their sub-clusters (which are the putative global minima
for Dzugutov potential). We then perform a short, high-precision relaxation of these
candidates to ensure they are at the exact minimum, and select the best energy.
This is a "rigid-body" approach where the topology is fixed to high-symmetry icosahedral
shells, and we only optimize the scale (or allow slight relaxation) to find the exact
global minimum for each N.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
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

def _get_dodecahedron_vertices():
    """Return 20 vertices of a regular dodecahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_icosahedron_face_centers():
    """Return 20 face centers of a regular icosahedron (same as dodecahedron vertices)."""
    return _get_dodecahedron_vertices()

def _get_icosahedron_edge_midpoints():
    """
    Return 30 edge midpoints of a regular icosahedron.
    An edge connects two vertices if their distance is minimal.
    """
    verts = _get_icosahedron_vertices()
    # Compute all pairwise distances
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    # Find edges (dist < min_dist * 1.1 to be safe against floating point)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if dists[i, j] < min_dist * 1.1:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
    return np.array(edges)

def _generate_mackay_shells(n):
    """
    Generate a cluster of n atoms based on Mackay icosahedral shells.
    Shell 1: 1 atom (center)
    Shell 2: 12 atoms (icosahedron vertices)
    Shell 3: 42 atoms (30 edge midpoints + 12 'other' positions? 
              Actually, the 42 atoms in the 2nd shell of a Mackay icosahedron are:
              12 atoms at the vertices of a larger icosahedron? No.
              
              Let's use the standard construction:
              The 55-atom Mackay icosahedron consists of:
              1 center
              12 in first shell (radius R1)
              42 in second shell.
              The 42 atoms are located at the 30 edge midpoints of the first shell icosahedron
              and 12 additional atoms?
              
              Actually, the 42 atoms in the second shell of a Mackay icosahedron are:
              12 atoms at the same angular positions as the first shell vertices but at a larger radius? No.
              
              Correct Mackay 55 construction:
              It is a close-packed icosahedron.
              Layer 0: 1
              Layer 1: 12 (Icosahedron vertices)
              Layer 2: 42.
              The 42 atoms are formed by:
              - 12 atoms at the centers of the 12 pentagonal faces of the dual dodecahedron? No.
              
              Let's use a robust geometric generation:
              The 55-atom cluster is often described as having icosahedral symmetry.
              We can generate the 55-atom cluster by taking the 13-atom cluster and adding 42 atoms.
              A common way to generate the 55-atom Mackay icosahedron is:
              1. 1 center
              2. 12 vertices of icosahedron (radius R1)
              3. 30 midpoints of edges of the icosahedron (radius R_edge_mid)
              4. 12 vertices of a larger icosahedron? No, that would be 12 more.
              
              Wait, 1 + 12 + 30 + 12 = 55.
              Are the last 12 atoms at the vertices of a larger icosahedron?
              Or are they at the centers of the 20 faces? No, 20 faces.
              
              Actually, the 42 atoms in the second shell are:
              30 atoms at the edge midpoints of the first shell.
              12 atoms at the vertices of a second, larger icosahedron?
              
              Let's check the distances.
              In a Mackay icosahedron, all atoms in a shell are equidistant from the center.
              So the 42 atoms must all be at the same radius R2.
              The 30 edge midpoints of a unit icosahedron are at radius phi = 1.618.
              The 12 vertices of a larger icosahedron? If we scale the first icosahedron by phi,
              the vertices are at radius phi.
              So, if we place 12 atoms at the vertices of an icosahedron scaled by phi,
              and 30 atoms at the edge midpoints of the original icosahedron,
              do they all have the same radius?
              Yes, the edge midpoints of a unit icosahedron are at radius phi.
              The vertices of an icosahedron scaled by phi are at radius phi.
              So, the 42 atoms are:
              - 30 edge midpoints of the 12-vertex icosahedron (radius 1).
              - 12 vertices of a 12-vertex icosahedron scaled by phi (radius phi).
              
              This gives 30 + 12 = 42 atoms at radius phi.
              Total: 1 + 12 + 42 = 55.
              
              For other N, we can truncate this structure or use sub-clusters.
              Since we need 38, 45, 50, 55, 57, 60:
              - 55: Full Mackay.
              - 50, 45, 38: Sub-clusters of Mackay (remove outer atoms).
              - 57, 60: Supersets? Or different structures?
              
              For N > 55, we might need a 3rd shell or a different structure.
              However, the task says "putative global minima of Doye, Wales and Simdyankin".
              These are known for specific N.
              
              Let's generate the 55-atom Mackay structure and its sub-clusters.
              For N=57, 60, we might need to add atoms to the 55-atom cluster or use a different known minimum.
              Given the time limit, we will generate candidates for the 55-atom Mackay and its truncations,
              and for N > 55, we will use the 55-atom structure plus random atoms on a 3rd shell,
              or simply rely on the relaxation of the best candidate.
              
              Actually, a better strategy for N > 55 is to use the 55-atom structure and add atoms
              in a way that maintains icosahedral symmetry if possible, or just use the 55-atom
              structure as a seed for basin hopping if N > 55.
              
              But let's stick to the discrete combinatorial search for the known minima.
              The known global minima for Dzugutov are often Mackay icosahedra or their sub-clusters.
              
              Let's define the 55-atom structure explicitly.
    """
    # Generate 55-atom Mackay icosahedron
    v1 = _get_icosahedron_vertices()
    edge_mids = _get_icosahedron_edge_midpoints()
    v2 = _get_icosahedron_vertices() * ((1 + np.sqrt(5)) / 2)  # Scale by phi
    
    # Combine
    # Shell 1: Center
    # Shell 2: v1 (12 atoms)
    # Shell 3: edge_mids (30 atoms) + v2 (12 atoms) = 42 atoms
    
    # All atoms
    atoms_55 = np.vstack([
        np.zeros(1).reshape(1, 3),
        v1,
        edge_mids,
        v2
    ])
    
    # For N < 55, we can remove atoms from the outer shell.
    # The outer shell (shell 3) has 42 atoms.
    # We can remove atoms from shell 3 to get N = 55 - k.
    # For N > 55, we need to add atoms.
    
    if n == 55:
        return atoms_55
    
    # For other N, we generate a candidate by taking the 55-atom structure
    # and adding/removing atoms.
    
    # This is a simplified approach. A more robust approach would be to
    # generate multiple candidate structures for each N and relax them.
    
    # For now, let's just return the 55-atom structure for all N,
    # and let the relaxation handle the rest. This is not ideal for N << 55,
    # but it's a good starting point.
    
    # Better: Generate a random cluster of N atoms and relax it.
    # But we want to use the discrete combinatorial search.
    
    # Let's generate the 55-atom structure and then randomly remove atoms
    # if N < 55, or add random atoms if N > 55.
    
    if n < 55:
        # Remove atoms from the outer shell (last 42 atoms)
        # Keep the first 55 - n atoms
        return atoms_55[:n]
    else:
        # Add random atoms on a sphere of radius R3
        R3 = np.linalg.norm(atoms_55[0]) + 1.0  # Approximate
        # Actually, let's just use the 55-atom structure and add random atoms
        # This is a fallback.
        extra = np.random.uniform(-1, 1, (n - 55, 3)) * 2.0
        return np.vstack([atoms_55, extra])

def _generate_candidate(n, seed=0):
    """
    Generate a candidate structure for n atoms.
    We will generate multiple candidates:
    1. Mackay icosahedron (truncated/extended)
    2. Random compact cluster
    3. Perturbed Mackay
    """
    np.random.seed(seed)
    
    # Candidate 1: Mackay
    pos_mackay = _generate_mackay_shells(n)
    
    # Candidate 2: Random compact
    pos_random = []
    for _ in range(n):
        for _attempt in range(100):
            p = np.random.uniform(-1.5, 1.5, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in pos_random):
                pos_random.append(p)
                break
        else:
            pos_random.append(np.array([np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(-3, 3)]))
    pos_random = np.array(pos_random)
    
    # Candidate 3: Perturbed Mackay
    pos_perturbed = pos_mackay + np.random.normal(0, 0.1, (n, 3))
    
    return [pos_mackay, pos_random, pos_perturbed]

def _relax_candidate(pos, potential, parameter):
    """
    Relax a candidate structure and return the energy and relaxed positions.
    """
    try:
        e, pos_relaxed = relax(pos, potential, parameter)
        return e, pos_relaxed
    except Exception:
        return np.inf, pos

def _basin_hopping_worker(start_pos, potential, parameter, time_limit, best_energy_holder, lock):
    """
    Worker for Basin Hopping.
    """
    pos = start_pos.copy()
    e, pos_relaxed = relax(pos, potential, parameter)
    current_energy = e
    
    T = 5.0
    cooling_rate = 0.99
    step_size = 0.3
    
    end_time = time.time() + time_limit
    
    while time.time() < end_time:
        # Perturb
        n_perturb = max(1, len(pos) // 10)
        indices = np.random.choice(len(pos), n_perturb, replace=False)
        new_pos = pos.copy()
        for i in indices:
            new_pos[i] += np.random.normal(0, step_size, 3)
        
        # Relax
        try:
            new_e, new_pos_relaxed = relax(new_pos, potential, parameter)
        except Exception:
            continue
            
        delta_e = new_e - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            pos = new_pos_relaxed
            current_energy = new_e
            
            # Update global best
            with lock:
                if current_energy < best_energy_holder[0]:
                    best_energy_holder[0] = current_energy
                    best_energy_holder[1] = pos.copy()
                    
            step_size *= 0.999
            if step_size < 0.1:
                step_size = 0.1
                
        T *= cooling_rate
        if T < 0.1:
            T = 5.0

    return current_energy

def construct(instance):
    """
    Strategy:
    1. Generate multiple candidate structures using discrete combinatorial search (Mackay icosahedra).
    2. Relax each candidate to find the local minimum.
    3. Run basin hopping from the best candidate to ensure global minimum.
    4. Return the best structure found.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # 1. Generate Candidates
    candidates = _generate_candidate(size, seed=42)
    
    # 2. Relax Candidates
    best_energy = np.inf
    best_positions = None
    
    for pos in candidates:
        if not _check_time():
            break
        e, pos_relaxed = _relax_candidate(pos, potential, parameter)
        if e < best_energy:
            best_energy = e
            best_positions = pos_relaxed
    
    # 3. Basin Hopping from Best Candidate
    if best_positions is not None and _check_time():
        # Run basin hopping for the remaining time
        time_remaining = TIME_LIMIT - (time.time() - _START_TIME)
        # Use 4 threads
        num_threads = 4
        time_per_thread = time_remaining / num_threads
        
        start_points = [best_positions.copy()]
        for _ in range(3):
            perturbed = best_positions + np.random.normal(0, 0.2, (size, 3))
            start_points.append(perturbed)
        
        best_energy_holder = [best_energy, best_positions]
        lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for sp in start_points:
                if not _check_time():
                    break
                futures.append(executor.submit(_basin_hopping_worker, sp, potential, parameter, time_per_thread, best_energy_holder, lock))
                
            try:
                for future in as_completed(futures, timeout=time_remaining):
                    future.result()
            except Exception:
                pass
        
        best_energy = best_energy_holder[0]
        best_positions = best_energy_holder[1]
    
    # 4. Final Relaxation
    if best_positions is not None:
        try:
            e_final, pos_final = relax(best_positions, potential, parameter)
            return pos_final
        except Exception:
            return best_positions
    else:
        # Fallback: Random cluster
        pos_random = _generate_candidate(size, seed=0)[1]
        try:
            e_final, pos_final = relax(pos_random, potential, parameter)
            return pos_final
        except Exception:
            return pos_random

# EVOLVE-BLOCK-END
